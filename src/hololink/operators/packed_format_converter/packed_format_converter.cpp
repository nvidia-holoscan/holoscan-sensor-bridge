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

#include "packed_format_converter.hpp"

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/networking.hpp>
#include <holoscan/holoscan.hpp>

/**
 * @brief This macro defining a YAML converter which throws for unsupported types.
 *
 * Background: Holoscan supports setting parameters through YAML files. But for some parameters
 * accepted by the receiver operators like `DataChannel` class of functions it makes no sense
 * to specify them in YAML files. Therefore use a converter which throws for these types.
 *
 * @tparam TYPE
 */
#define YAML_CONVERTER(TYPE)                                                                \
    template <>                                                                             \
    struct YAML::convert<TYPE> {                                                            \
        static Node encode(TYPE&) { throw std::runtime_error("Unsupported"); }              \
        static bool decode(const Node&, TYPE&) { throw std::runtime_error("Unsupported"); } \
    };

YAML_CONVERTER(hololink::DataChannel*);

namespace {

const char* source = R"(
extern "C" {

// Converts packed 10-bit data, where each 4-byte dword contains 3 pixels, into 16-bit data.
__global__ void packed10bitTo16bit(unsigned short* output, const unsigned int* input,
                                   int bytes_per_line, int width, int height) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx_x >= width / 3) || (idx_y >= height))
        return;

    int input_index = (idx_y * bytes_per_line / 4) + idx_x;
    int output_index = (idx_y * width) + (idx_x * 3);

    output[output_index]     = (input[input_index] << 6) & 0xFFC0;
    output[output_index + 1] = (input[input_index] >> 4) & 0xFFC0;
    output[output_index + 2] = (input[input_index] >> 14) & 0xFFC0;
}

// Converts packed 12-bit data, where 2 pixels are stored in every 3 bytes.
__global__ void packed12bitTo16bit(unsigned short* output, const unsigned char* input,
                                   int bytes_per_line, int width, int height) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx_x >= width / 2) || (idx_y >= height))
        return;

    int input_index = (idx_y * bytes_per_line) + (idx_x * 3);
    int output_index = (idx_y * width) + (idx_x * 2);

    output[output_index]     = (input[input_index + 1] << 12 |
                                input[input_index + 2] << 4) & 0xFFC0;
    output[output_index + 1] = (input[input_index] << 8 |
                                input[input_index + 1]) & 0xFFC0;
}

})";

} // anonymous namespace

namespace hololink::operators {

void PackedFormatConverterOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input<holoscan::gxf::Entity>("input");
    spec.output<holoscan::gxf::Entity>("output");

    register_converter<hololink::DataChannel*>();

    spec.param(allocator_, "allocator", "Allocator",
        "Allocator used to allocate the output image, defaults to BlockMemoryPool");
    spec.param(cuda_device_ordinal_, "cuda_device_ordinal", "CudaDeviceOrdinal",
        "Device to use for CUDA operations", 0);
    spec.param(in_tensor_name_, "in_tensor_name", "InputTensorName",
        "Name of the input tensor", std::string(""));
    spec.param(out_tensor_name_, "out_tensor_name", "OutputTensorName",
        "Name of the output tensor", std::string(""));
    spec.param(hololink_channel_, "hololink_channel", "HololinkChannel",
        "Pointer to Hololink Datachannel object");
    cuda_stream_handler_.define_params(spec);
}

void PackedFormatConverterOp::start()
{
    if (!configured_) {
        throw std::runtime_error("PackedFormatConverterOp is not configured.");
    }

    CudaCheck(cuInit(0));
    CudaCheck(cuDeviceGet(&cuda_device_, cuda_device_ordinal_.get()));
    CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));
    int integrated = 0;
    CudaCheck(cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, cuda_device_));
    is_integrated_ = (integrated != 0);

    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);

    cuda_function_launcher_.reset(new hololink::common::CudaFunctionLauncher(
        source, { "packed10bitTo16bit", "packed12bitTo16bit" }));

    if (hololink_channel_.has_value() && hololink_channel_.get()) {
        switch (pixel_format_) {
        case hololink::csi::PixelFormat::RAW_10:
            hololink_channel_.get()->enable_packetizer_10();
            break;
        case hololink::csi::PixelFormat::RAW_12:
            hololink_channel_.get()->enable_packetizer_12();
            break;
        default:
            throw std::runtime_error("Unsupported bits per pixel value");
        }
    }
}

void PackedFormatConverterOp::stop()
{
    if (hololink_channel_.has_value() && hololink_channel_.get()) {
        hololink_channel_.get()->disable_packetizer();
    }

    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);

    cuda_function_launcher_.reset();

    CudaCheck(cuDevicePrimaryCtxRelease(cuda_device_));
    cuda_context_ = nullptr;
}

void PackedFormatConverterOp::compute(holoscan::InputContext& input, holoscan::OutputContext& output,
    holoscan::ExecutionContext& context)
{
    auto maybe_entity = input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) {
        throw std::runtime_error("Failed to receive input");
    }

    auto& entity = static_cast<nvidia::gxf::Entity&>(maybe_entity.value());

    // get the CUDA stream from the input message
    gxf_result_t stream_handler_result
        = cuda_stream_handler_.from_message(context.context(), entity);
    if (stream_handler_result != GXF_SUCCESS) {
        throw std::runtime_error(fmt::format(
            "Failed to get the CUDA stream from incoming messages: {}",
            GxfResultStr(stream_handler_result)));
    }

    const auto maybe_tensor = entity.get<nvidia::gxf::Tensor>(in_tensor_name_.get().c_str());
    if (!maybe_tensor) {
        throw std::runtime_error(fmt::format("Tensor not found in message ({})", in_tensor_name_.get()));
    }

    const auto input_tensor = maybe_tensor.value();

    if (input_tensor->storage_type() == nvidia::gxf::MemoryStorageType::kHost) {
        if (!is_integrated_ && !host_memory_warning_) {
            host_memory_warning_ = true;
            HSB_LOG_WARN(
                "The input tensor is stored in host memory, this will reduce performance of this "
                "operator. For best performance store the input tensor in device memory.");
        }
    } else if (input_tensor->storage_type() != nvidia::gxf::MemoryStorageType::kDevice) {
        throw std::runtime_error(
            fmt::format("Unsupported storage type {}", (int)input_tensor->storage_type()));
    }

    if (input_tensor->rank() != 1) {
        throw std::runtime_error("Tensor must be one dimensional");
    }

    // get handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        fragment()->executor().context(), allocator_->gxf_cid());

    // create the output
    nvidia::gxf::Shape shape { int(pixel_height_), int(pixel_width_), 1 };
    nvidia::gxf::Expected<nvidia::gxf::Entity> out_message
        = CreateTensorMap(context.context(), allocator.value(),
            { { out_tensor_name_.get(), nvidia::gxf::MemoryStorageType::kDevice, shape,
                nvidia::gxf::PrimitiveType::kUnsigned16, 0,
                nvidia::gxf::ComputeTrivialStrides(shape,
                    nvidia::gxf::PrimitiveTypeSize(nvidia::gxf::PrimitiveType::kUnsigned16)) } },
            false);

    if (!out_message) {
        throw std::runtime_error("failed to create out_message");
    }
    const auto tensor = out_message.value().get<nvidia::gxf::Tensor>(out_tensor_name_.get().c_str());
    if (!tensor) {
        throw std::runtime_error(
            fmt::format("failed to create out_tensor with name \"{}\"", out_tensor_name_.get()));
    }

    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);
    const cudaStream_t cuda_stream = cuda_stream_handler_.get_cuda_stream(context.context());

    switch (pixel_format_) {
    case hololink::csi::PixelFormat::RAW_10:
        cuda_function_launcher_->launch("packed10bitTo16bit",
            { pixel_width_ / 3, // outputs 3 pixels per shader invocation
                pixel_height_, 1 },
            cuda_stream, tensor.value()->pointer(),
            input_tensor->pointer() + start_byte_,
            bytes_per_line_, pixel_width_, pixel_height_);
        break;
    case hololink::csi::PixelFormat::RAW_12:
        cuda_function_launcher_->launch("packed12bitTo16bit",
            { pixel_width_ / 2, // outputs 2 pixels per shader invocation
                pixel_height_, 1 },
            cuda_stream, tensor.value()->pointer(),
            input_tensor->pointer() + start_byte_,
            bytes_per_line_, pixel_width_, pixel_height_);
        break;
    default:
        throw std::runtime_error("Unsupported bits per pixel value");
    }

    // pass the CUDA stream to the output message
    stream_handler_result = cuda_stream_handler_.to_message(out_message);
    if (stream_handler_result != GXF_SUCCESS) {
        throw std::runtime_error("Failed to add the CUDA stream to the outgoing messages");
    }

    // Emit the tensor
    auto result = holoscan::gxf::Entity(std::move(out_message.value()));
    output.emit(result);
}

uint32_t PackedFormatConverterOp::receiver_start_byte()
{
    return 0;
}

uint32_t PackedFormatConverterOp::received_line_bytes(uint32_t transmitted_line_bytes)
{
    return hololink::core::round_up(transmitted_line_bytes, 64);
}

uint32_t PackedFormatConverterOp::transmitted_line_bytes(hololink::csi::PixelFormat pixel_format,
    uint32_t pixel_width)
{
    switch (pixel_format) {
    case hololink::csi::PixelFormat::RAW_8:
        return pixel_width;
    case hololink::csi::PixelFormat::RAW_10:
        // 3 pixels per 4 bytes
        return ((pixel_width + 2) / 3) * 4;
    case hololink::csi::PixelFormat::RAW_12:
        // 2 pixels per 3 bytes
        return ((pixel_width + 1) / 2) * 3;
    default:
        throw std::runtime_error("Invalid bit depth");
    }
}

void PackedFormatConverterOp::configure(uint32_t start_byte, uint32_t bytes_per_line,
    uint32_t pixel_width, uint32_t pixel_height,
    hololink::csi::PixelFormat pixel_format,
    uint32_t trailing_bytes)
{
    HSB_LOG_INFO("start_byte={}, bytes_per_line={}, width={}, height={}, trailing_bytes={}",
        start_byte, bytes_per_line, pixel_width, pixel_height, trailing_bytes);
    start_byte_ = start_byte;
    bytes_per_line_ = bytes_per_line;
    pixel_width_ = pixel_width;
    pixel_height_ = pixel_height;
    pixel_format_ = pixel_format;
    trailing_bytes_ = trailing_bytes;
    configured_ = true;
}

size_t PackedFormatConverterOp::get_frame_size()
{
    return start_byte_ + (bytes_per_line_ * pixel_height_) + trailing_bytes_;
}

} // namespace hololink::operators
