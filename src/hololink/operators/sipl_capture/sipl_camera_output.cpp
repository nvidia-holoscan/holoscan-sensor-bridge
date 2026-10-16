/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "sipl_camera_output.hpp"

#include <hololink/core/hololink.hpp>
#include <hololink/core/logging_internal.hpp>

namespace hololink::operators {

namespace {

    class AcquiredFrameGuard {
    public:
        AcquiredFrameGuard(SIPLCaptureService& service, SIPLCaptureService::AcquiredFrame& frame)
            : service_(service)
            , frame_(frame)
        {
        }

        ~AcquiredFrameGuard()
        {
            if (active_) {
                service_.release_acquired_frame(frame_);
            }
        }

        void handoff_pending_output()
        {
            if (!active_) {
                return;
            }
            service_.release_raw_buffer_if_unused(frame_.buffer, frame_.buffer_raw);
            service_.register_pending_output(frame_.cuda_ptr, frame_.buffer);
            active_ = false;
        }

    private:
        SIPLCaptureService& service_;
        SIPLCaptureService::AcquiredFrame& frame_;
        bool active_ = true;
    };

} // namespace

void SIPLCameraOutputOp::setup(holoscan::OperatorSpec& spec)
{
    spec.output<holoscan::gxf::Entity>("output");

    // How long compute() waits for the next frame of this camera before the pipeline is
    // declared dead. A truncated sensor frame makes the capture engine skip and re-arm
    // its requests; that recovery has been observed to take more than one second, so
    // the default leaves room for it while still catching a stream that has stopped.
    spec.param(timeout_, "timeout", "Timeout",
        "Time to wait for the next frame before failing, in microseconds", 3000000U);
}

void SIPLCameraOutputOp::start()
{
    if (!service_) {
        throw std::runtime_error("SIPLCameraOutputOp requires a SIPLCaptureService");
    }
    if (camera_index_ >= service_->camera_count()) {
        throw std::runtime_error(fmt::format(
            "camera_index {} is out of range ({} cameras)", camera_index_, service_->camera_count()));
    }
    service_->add_operator_ref();
}

void SIPLCameraOutputOp::stop()
{
    if (service_) {
        service_->remove_operator_ref();
    }
}

void SIPLCameraOutputOp::compute(holoscan::InputContext& op_input,
    holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& context)
{
    const auto timeout_us = timeout_.get();
    auto frame = service_->acquire_frame(camera_index_, timeout_us);
    if (frame.status != SIPLCaptureService::AcquireStatus::Ok) {
        throw std::runtime_error(fmt::format(
            "SIPL frame acquire failed for camera {}", service_->output_name(camera_index_)));
    }

    AcquiredFrameGuard frame_guard(*service_, frame);

    auto nvm_buffer = dynamic_cast<nvsipl::INvSIPLClient::INvSIPLNvMBuffer*>(frame.buffer);
    if (nvm_buffer == nullptr) {
        throw std::runtime_error("Failed to get INvSIPLNvMBuffer");
    }

    const auto& output_name = service_->output_name(camera_index_);
    const auto name = output_name.c_str();

    const bool is_nv12 = frame.plane_color_format[0] == NvSciColor_Y8 && frame.plane_color_format[1] == NvSciColor_V8U8;
    const bool is_raw10 = frame.plane_color_format[0] == NvSciColor_X2Rc10Rb10Ra10_Bayer10RGGB
        || frame.plane_color_format[0] == NvSciColor_X2Rc10Rb10Ra10_Bayer10BGGR
        || frame.plane_color_format[0] == NvSciColor_X2Rc10Rb10Ra10_Bayer10GRBG
        || frame.plane_color_format[0] == NvSciColor_X2Rc10Rb10Ra10_Bayer10GBRG;

    // Everything that can fail is done before the Entity wraps the buffer, so a throw here
    // leaves nothing wrapped and the guard simply releases the frame.
    if (is_metadata_enabled()) {
        // The HSB metadata is written to the RAW image. The acquire thread copied it out of
        // the RAW buffer after its EOF wait (and, in ISP mode, matched it to this frame by
        // capture-start TSC), so no NvSci CPU mapping is needed here. The keys are omitted
        // when no record was matched (after a capture drop, or for a non-HSB camera).
        if (frame.hsb_metadata) {
            const auto& hsb_metadata = *frame.hsb_metadata;
            metadata()->set(output_name + "_frame_number", int64_t(hsb_metadata.frame_number));
            metadata()->set(output_name + "_timestamp_s", int64_t(hsb_metadata.timestamp_s));
            metadata()->set(output_name + "_timestamp_ns", int64_t(hsb_metadata.timestamp_ns));
            metadata()->set(output_name + "_metadata_s", int64_t(hsb_metadata.metadata_s));
            metadata()->set(output_name + "_metadata_ns", int64_t(hsb_metadata.metadata_ns));
            metadata()->set(output_name + "_crc", int64_t(hsb_metadata.crc));
            metadata()->set(output_name + "_bytes_written", int64_t(hsb_metadata.bytes_written));
        } else {
            HSB_LOG_DEBUG("No HSB metadata for {}", output_name);
        }

        // Output the SIPL ImageMetaData (e.g. frame timestamps and info), copied by value
        // while the buffer is still held.
        auto image_metadata = std::make_shared<holoscan::MetadataObject>();
        image_metadata->set_value<nvsipl::INvSIPLClient::ImageMetaData>(
            nvsipl::INvSIPLClient::ImageMetaData(nvm_buffer->GetImageData()));
        metadata()->set(output_name + "_metadata", image_metadata);

        // Output the ISP stats.
        if (auto isp_stats = service_->isp_stats(camera_index_)) {
            auto isp_stats_info = isp_stats->GetIspStatsInfo(nvm_buffer);
            if (!isp_stats_info) {
                throw std::runtime_error("Failed to get IspStatsInfo");
            }
            auto isp_stats_obj = std::make_shared<holoscan::MetadataObject>();
            isp_stats_obj->set_value<nvsipl::IspStatsInfo>(nvsipl::IspStatsInfo(*isp_stats_info));
            metadata()->set(output_name + "_isp_stats", isp_stats_obj);
        }
    }

    auto entity = holoscan::gxf::Entity::New(&context);

    if (is_nv12) {
        auto video_buffer = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::VideoBuffer>(name);
        if (!video_buffer) {
            throw std::runtime_error("Failed to add GXF VideoBuffer");
        }
        nvidia::gxf::VideoTypeTraits<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12> video_type;
        nvidia::gxf::VideoFormatSize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12> color_format;
        auto color_planes = color_format.getDefaultColorPlanes(frame.plane_width[0], frame.plane_height[0]);
        color_planes[0].stride = frame.plane_pitch[0];
        color_planes[1].offset = frame.plane_offset[1];
        color_planes[1].stride = frame.plane_pitch[1];
        nvidia::gxf::VideoBufferInfo info {
            frame.plane_width[0],
            frame.plane_height[0],
            video_type.value,
            std::move(color_planes),
            nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR
        };
        if (!video_buffer.value()->wrapMemory(info, frame.buffer_size, nvidia::gxf::MemoryStorageType::kDevice,
                frame.cuda_ptr, SIPLCaptureService::buffer_release_callback)) {
            throw std::runtime_error("Failed to add wrapped VideoBuffer memory");
        }
    } else if (is_raw10) {
        auto tensor = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>(name);
        if (!tensor) {
            throw std::runtime_error("Failed to add GXF Tensor");
        }

        // The whole RAW plane is wrapped as-is. That the image rows CameraInfo advertises to
        // the consumer fit inside it was checked when these buffers were allocated
        // (SIPLCaptureService::validate_raw_geometry), so nothing is re-checked per frame.
        const int raw_len = static_cast<int>(frame.plane_pitch[0] * frame.plane_height[0]);
        nvidia::gxf::Shape shape { raw_len };
        const auto element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
        const auto element_size = nvidia::gxf::PrimitiveTypeSize(element_type);

        if (!tensor.value()->wrapMemory(shape, element_type, element_size,
                nvidia::gxf::ComputeTrivialStrides(shape, element_size),
                nvidia::gxf::MemoryStorageType::kDevice,
                frame.cuda_ptr, SIPLCaptureService::buffer_release_callback)) {
            throw std::runtime_error("Failed to add wrapped memory");
        }
    } else {
        throw std::runtime_error(fmt::format("Buffer has unsupported color format: {}",
            static_cast<int>(frame.plane_color_format[0])));
    }

    frame_guard.handoff_pending_output();

    op_output.emit(entity, "output");
}

} // namespace hololink::operators
