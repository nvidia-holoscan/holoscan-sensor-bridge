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

    spec.param(timeout_, "timeout", "Timeout",
        "Timeout for capture requests, in microseconds", 1000000U);
}

void SIPLCameraOutputOp::start()
{
    if (!service_) {
        throw std::runtime_error("SIPLCameraOutputOp requires a SIPLCaptureService");
    }
    service_->get_camera_info();
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

    if (is_metadata_enabled()) {
        // The HSB metadata is written to the RAW image.
        const void* raw_ptr = nullptr;
        size_t hsb_metadata_offset = 0;
        if (frame.buffer == frame.buffer_raw) {
            // If we're already processing the RAW buffer, just get the pointer and offset.
            auto err = NvSciBufObjGetConstCpuPtr(frame.buf_obj, &raw_ptr);
            if (err != NvSciError_Success || raw_ptr == nullptr) {
                throw std::runtime_error("Failed to map buffer for metadata");
            }
            hsb_metadata_offset = frame.plane_pitch[0] * frame.plane_height[0];
        } else {
            // If we're not already processing the RAW buffer then we need to get the
            // buffer, attributes, and mapping required to read HSB metadata from it.
            auto nvm_buffer_raw = dynamic_cast<nvsipl::INvSIPLClient::INvSIPLNvMBuffer*>(frame.buffer_raw);
            if (nvm_buffer_raw == nullptr) {
                throw std::runtime_error("Failed to get INvSIPLNvMBuffer for metadata");
            }

            NvSciBufObj buf_obj_raw = nvm_buffer_raw->GetNvSciBufImage();
            if (buf_obj_raw == nullptr) {
                throw std::runtime_error("Failed to get NvSciBufObj for metadata");
            }

            NvSciBufAttrList buf_attr_list_raw;
            auto err = NvSciBufObjGetAttrList(buf_obj_raw, &buf_attr_list_raw);
            if (err != NvSciError_Success) {
                throw std::runtime_error("Failed to get buffer attribute list for metadata");
            }

            NvSciBufAttrKeyValuePair img_attrs_raw[] = {
                { NvSciBufImageAttrKey_PlanePitch, NULL, 0 },
                { NvSciBufImageAttrKey_PlaneHeight, NULL, 0 },
            };
            size_t num_attrs_raw = sizeof(img_attrs_raw) / sizeof(img_attrs_raw[0]);

            err = NvSciBufAttrListGetAttrs(buf_attr_list_raw, img_attrs_raw, num_attrs_raw);
            if (err != NvSciError_Success) {
                throw std::runtime_error("Failed to get buffer attributes for metadata");
            }

            const uint32_t plane_pitch_raw = *static_cast<const uint32_t*>(img_attrs_raw[0].value);
            const uint32_t plane_height_raw = *static_cast<const uint32_t*>(img_attrs_raw[1].value);

            err = NvSciBufObjGetConstCpuPtr(buf_obj_raw, &raw_ptr);
            if (err != NvSciError_Success || raw_ptr == nullptr) {
                throw std::runtime_error("Failed to map buffer");
            }
            hsb_metadata_offset = plane_pitch_raw * plane_height_raw;
        }

        // Deserialize and output the HSB metadata.
        auto hsb_metadata_ptr = static_cast<const uint8_t*>(raw_ptr) + hsb_metadata_offset;
        auto hsb_metadata = Hololink::deserialize_metadata(hsb_metadata_ptr, hololink::METADATA_SIZE);
        metadata()->set(output_name + "_frame_number", int64_t(hsb_metadata.frame_number));
        metadata()->set(output_name + "_timestamp_s", int64_t(hsb_metadata.timestamp_s));
        metadata()->set(output_name + "_timestamp_ns", int64_t(hsb_metadata.timestamp_ns));
        metadata()->set(output_name + "_metadata_s", int64_t(hsb_metadata.metadata_s));
        metadata()->set(output_name + "_metadata_ns", int64_t(hsb_metadata.metadata_ns));
        metadata()->set(output_name + "_crc", int64_t(hsb_metadata.crc));
        metadata()->set(output_name + "_bytes_written", int64_t(hsb_metadata.bytes_written));

        // Output the SIPL ImageMetaData (e.g. frame timestamps and info).
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

    frame_guard.handoff_pending_output();

    op_output.emit(entity, "output");
}

} // namespace hololink::operators
