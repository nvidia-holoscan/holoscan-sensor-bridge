/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "fusa_coe_capture.hpp"

#include <hololink/core/data_channel.hpp>
#include <hololink/core/hololink.hpp>

#include <gxf/std/tensor.hpp>
#include <yaml-cpp/yaml.h>

#define YAML_CONVERTER(TYPE)                                                                \
    template <>                                                                             \
    struct YAML::convert<TYPE> {                                                            \
        static Node encode(TYPE&) { throw std::runtime_error("Unsupported"); }              \
        static bool decode(const Node&, TYPE&) { throw std::runtime_error("Unsupported"); } \
    };

YAML_CONVERTER(hololink::DataChannel*);
YAML_CONVERTER(std::function<void()>);

namespace hololink::operators {

namespace {

    class LegacyCoeChannelConfig : public fusa_coe_capture::CoeChannelConfig {
    public:
        explicit LegacyCoeChannelConfig(hololink::DataChannel* channel)
            : channel_(channel)
        {
        }

        void set_packetizer_if_needed(
            bool csi_pixel_format, hololink::csi::PixelFormat pixel_format) override
        {
            if (csi_pixel_format) {
                auto packetizer_program = hololink::csi::get_packetizer_program(pixel_format);
                channel_->set_packetizer_program(packetizer_program);
            }
        }

        void configure_coe(
            uint8_t coe_channel, size_t frame_size, uint32_t bytes_per_line) override
        {
            channel_->configure_coe(coe_channel, frame_size, bytes_per_line);
        }

        void unconfigure() override
        {
            channel_->unconfigure();
        }

    private:
        hololink::DataChannel* channel_;
    };

    class LegacyCoeMetadataDecoder : public fusa_coe_capture::CoeMetadataDecoder {
    public:
        bool decode(
            const void* host_memory, size_t host_memory_size,
            fusa_coe_capture::CoeFrameMetadata& out) const override
        {
            auto frame_metadata = Hololink::deserialize_metadata(
                static_cast<const uint8_t*>(host_memory), host_memory_size);
            out.timestamp_s = static_cast<int64_t>(frame_metadata.timestamp_s);
            out.timestamp_ns = static_cast<int64_t>(frame_metadata.timestamp_ns);
            out.metadata_s = static_cast<int64_t>(frame_metadata.metadata_s);
            out.metadata_ns = static_cast<int64_t>(frame_metadata.metadata_ns);
            out.crc = static_cast<int64_t>(frame_metadata.crc);
            out.frame_number = static_cast<int64_t>(frame_metadata.frame_number);
            return true;
        }
    };

} // namespace

void FusaCoeCaptureOp::setup(holoscan::OperatorSpec& spec)
{
    spec.output<holoscan::gxf::Entity>("output");

    register_converter<hololink::DataChannel*>();
    register_converter<std::function<void()>>();

    spec.param(interface_, "interface", "Interface", "Interface for the CoE device.");
    spec.param(mac_addr_, "mac_addr", "MACAddr", "MAC Address for the CoE device.");
    spec.param(timeout_, "timeout", "Timeout", "Timeout for capture requests, in milliseconds");
    spec.param(out_tensor_name_, "out_tensor_name", "OutputTensorName",
        "Name of the output tensor", std::string(""));
    spec.param(hololink_channel_, "hololink_channel", "HololinkChannel",
        "Pointer to Hololink Datachannel object");
    spec.param(device_start_, "device_start", "DeviceStart",
        "Function to be called to start the device");
    spec.param(device_stop_, "device_stop", "DeviceStop",
        "Function to be called to stop the device");
    spec.param(cpu_output_, "cpu_output", "CpuOutput",
        "Wrap output tensor with CPU pointer (kHost) instead of GPU pointer (kDevice).",
        false);
}

void FusaCoeCaptureOp::start()
{
    LegacyCoeChannelConfig channel(hololink_channel_.get());
    core_.start(
        interface_.get(),
        mac_addr_.get(),
        timeout_.get(),
        channel,
        device_start_.get(),
        device_stop_.get());
}

void FusaCoeCaptureOp::stop()
{
    LegacyCoeChannelConfig channel(hololink_channel_.get());
    core_.stop(channel, device_stop_.get());
}

void FusaCoeCaptureOp::compute(holoscan::InputContext& /*op_input*/,
    holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& context)
{
    fusa_coe_capture::FusaCoeCaptureCore::BufferView buffer {};
    if (!core_.wait_for_acquired_buffer(
            timeout_.get(), out_tensor_name_.get().c_str(), buffer)) {
        // No frame within the capture timeout; skip this tick and let the
        // scheduler call us again. A genuine capture failure throws instead.
        return;
    }

    if (is_metadata_enabled()) {
        fusa_coe_capture::CoeFrameMetadata frame_metadata {};
        LegacyCoeMetadataDecoder decoder;
        if (core_.decode_metadata(buffer, decoder, frame_metadata)) {
            auto const& meta = metadata();
            meta->set("timestamp_s", frame_metadata.timestamp_s);
            meta->set("timestamp_ns", frame_metadata.timestamp_ns);
            meta->set("metadata_s", frame_metadata.metadata_s);
            meta->set("metadata_ns", frame_metadata.metadata_ns);
            meta->set("crc", frame_metadata.crc);
            meta->set("frame_number", frame_metadata.frame_number);
        }
    }

    auto entity = holoscan::gxf::Entity::New(&context);
    auto name = out_tensor_name_.get().c_str();
    auto tensor = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>(name);
    if (!tensor) {
        throw std::runtime_error("Failed to add GXF Tensor");
    }

    nvidia::gxf::Shape shape {
        static_cast<int>(core_.bytes_per_line() * core_.pixel_height())
    };
    const auto element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
    const auto element_size = nvidia::gxf::PrimitiveTypeSize(element_type);

    void* data_ptr;
    nvidia::gxf::MemoryStorageType storage_type;
    if (cpu_output_.get()) {
        data_ptr = static_cast<uint8_t*>(buffer.cpu_ptr) + core_.start_byte();
        storage_type = nvidia::gxf::MemoryStorageType::kHost;
    } else {
        data_ptr = static_cast<uint8_t*>(buffer.cuda_device_ptr) + core_.start_byte();
        storage_type = nvidia::gxf::MemoryStorageType::kDevice;
    }
    if (!tensor.value()->wrapMemory(shape, element_type, element_size,
            nvidia::gxf::ComputeTrivialStrides(shape, element_size),
            storage_type, data_ptr, buffer_release_callback)) {
        throw std::runtime_error("Failed to add wrapped memory");
    }

    core_.register_pending_output(data_ptr);

    op_output.emit(entity, "output");
}

uint32_t FusaCoeCaptureOp::receiver_start_byte()
{
    return fusa_coe_capture::FusaCoeCaptureCore::receiver_start_byte();
}

uint32_t FusaCoeCaptureOp::received_line_bytes(uint32_t line_bytes)
{
    return fusa_coe_capture::FusaCoeCaptureCore::received_line_bytes(line_bytes);
}

uint32_t FusaCoeCaptureOp::transmitted_line_bytes(
    csi::PixelFormat pixel_format, uint32_t pixel_width)
{
    return fusa_coe_capture::FusaCoeCaptureCore::transmitted_line_bytes(
        pixel_format, pixel_width);
}

void FusaCoeCaptureOp::configure(
    uint32_t start_byte, uint32_t received_bytes_per_line,
    uint32_t pixel_width, uint32_t pixel_height,
    csi::PixelFormat pixel_format,
    uint32_t trailing_bytes)
{
    core_.configure(
        start_byte, received_bytes_per_line, pixel_width, pixel_height,
        pixel_format, trailing_bytes);
}

void FusaCoeCaptureOp::configure_converter(csi::CsiConverter& converter)
{
    converter.configure(
        0, core_.bytes_per_line(), core_.pixel_width(), core_.pixel_height(),
        core_.pixel_format(), 0);
}

void FusaCoeCaptureOp::configure_frame_size(uint32_t frame_size_bytes)
{
    core_.configure_frame_size(frame_size_bytes);
}

nvidia::gxf::Expected<void> FusaCoeCaptureOp::buffer_release_callback(void* pointer)
{
    return fusa_coe_capture::FusaCoeCaptureCore::buffer_release_callback(pointer);
}

} // namespace hololink::operators
