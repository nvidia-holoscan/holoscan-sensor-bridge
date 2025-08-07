/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "csi_to_bayer.hpp"

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/networking.hpp>
#include <holoscan/holoscan.hpp>

namespace {

const char* source = R"(
extern "C" {

__global__ void frameReconstruction8(unsigned short * out,
                                     const unsigned char * in,
                                     int per_line_size,
                                     int width,
                                     int height)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx_x >= width) || (idx_y >= height))
        return;

    const int in_index = (per_line_size * idx_y) + idx_x;
    const int out_index = idx_y * width + idx_x;

    out[out_index] = in[in_index] << 8;
}

__global__ void frameReconstruction10(unsigned short * out,
                                      const unsigned char * in,
                                      int per_line_size,
                                      int width,
                                      int height)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx_x >= width) || (idx_y >= height))
        return;

    const int in_index = (per_line_size * idx_y) + idx_x * 5;
    const int out_index = idx_y * width + idx_x * 4;

    const unsigned short lsbs = in[in_index + 4];
    out[out_index + 0] = ((in[in_index + 0] << 2) | (lsbs & 0x03)) << 6;
    out[out_index + 1] = ((in[in_index + 1] << 4) | (lsbs & 0x0C)) << 4;
    out[out_index + 2] = ((in[in_index + 2] << 6) | (lsbs & 0x30)) << 2;
    out[out_index + 3] = ((in[in_index + 3] << 8) | (lsbs & 0xC0));
}

__global__ void frameReconstruction12(unsigned short * out,
                                      const unsigned char * in,
                                      int per_line_size,
                                      int width,
                                      int height)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx_x >= width) || (idx_y >= height))
        return;

    const int in_index = (per_line_size * idx_y) + idx_x * 3;
    const int out_index = idx_y * width + idx_x * 2;

    const unsigned short lsbs = in[in_index + 2];
    out[out_index + 0] = ((in[in_index + 0] << 4) | (lsbs & 0x0F)) << 4;
    out[out_index + 1] = ((in[in_index + 1] << 8) | (lsbs & 0xF0));
}

})";

} // anonymous namespace

namespace hololink::operators {

void CsiToBayerOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input<holoscan::gxf::Entity>("input");
    spec.output<holoscan::gxf::Entity>("output");

    spec.param(allocator_, "allocator", "Allocator",
        "Allocator used to allocate the output Bayer image, defaults to BlockMemoryPool");
    spec.param(cuda_device_ordinal_, "cuda_device_ordinal", "CudaDeviceOrdinal",
        "Device to use for CUDA operations", 0);
    spec.param(out_tensor_name_, "out_tensor_name", "OutputTensorName",
        "Name of the output tensor", std::string(""));
    cuda_stream_handler_.define_params(spec);
}

void CsiToBayerOp::start()
{
    if (!configured_) {
        throw std::runtime_error("CsiToBayerOp is not configured.");
    }

    CudaCheck(cuInit(0));
    CudaCheck(cuDeviceGet(&cuda_device_, cuda_device_ordinal_.get()));
    CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));
    int integrated = 0;
    CudaCheck(cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, cuda_device_));
    is_integrated_ = (integrated != 0);

    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);

    cuda_function_launcher_.reset(new hololink::common::CudaFunctionLauncher(
        source, { "frameReconstruction8", "frameReconstruction10", "frameReconstruction12" }));
}

void CsiToBayerOp::stop()
{
    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);

    cuda_function_launcher_.reset();

    CudaCheck(cuDevicePrimaryCtxRelease(cuda_device_));
    cuda_context_ = nullptr;
}

void CsiToBayerOp::compute(holoscan::InputContext& input, holoscan::OutputContext& output,
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
        throw std::runtime_error(fmt::format("Failed to get the CUDA stream from incoming messages: {}", GxfResultStr(stream_handler_result)));
    }

    const auto maybe_tensor = entity.get<nvidia::gxf::Tensor>();
    if (!maybe_tensor) {
        throw std::runtime_error("Tensor not found in message");
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
    case hololink::csi::PixelFormat::RAW_8:
        cuda_function_launcher_->launch("frameReconstruction8", { pixel_width_, pixel_height_, 1 }, cuda_stream,
            tensor.value()->pointer(),
            input_tensor->pointer() + start_byte_, bytes_per_line_, pixel_width_,
            pixel_height_);
        break;
    case hololink::csi::PixelFormat::RAW_10:
        cuda_function_launcher_->launch("frameReconstruction10",
            { pixel_width_ / 4, // outputs 4 pixels per shader invocation
                pixel_height_, 1 },
            cuda_stream, tensor.value()->pointer(),
            input_tensor->pointer() + start_byte_, bytes_per_line_, pixel_width_,
            pixel_height_);
        break;
    case hololink::csi::PixelFormat::RAW_12:
        cuda_function_launcher_->launch("frameReconstruction12",
            { pixel_width_ / 2, // outputs 2 pixels per shader invocation
                pixel_height_, 1 },
            cuda_stream, tensor.value()->pointer(),
            input_tensor->pointer() + start_byte_, bytes_per_line_, pixel_width_,
            pixel_height_);
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

uint32_t CsiToBayerOp::receiver_start_byte()
{
    // HSB, in this mode, doesn't insert any stuff in the front of received data.
    return 0;
}

uint32_t CsiToBayerOp::received_line_bytes(uint32_t transmitted_line_bytes)
{
    // Bytes are padded to 8.
    return hololink::core::round_up(transmitted_line_bytes, 8);
}

uint32_t CsiToBayerOp::transmitted_line_bytes(hololink::csi::PixelFormat pixel_format, uint32_t pixel_width)
{
    switch (pixel_format) {
    case hololink::csi::PixelFormat::RAW_8:
        return pixel_width;
    case hololink::csi::PixelFormat::RAW_10:
        return pixel_width * 5 / 4;
    case hololink::csi::PixelFormat::RAW_12:
        return pixel_width * 3 / 2;
    default:
        throw std::runtime_error(fmt::format("Unsupported pixel format {}", int(pixel_format)));
    }
}

void CsiToBayerOp::configure(uint32_t start_byte, uint32_t bytes_per_line, uint32_t pixel_width, uint32_t pixel_height, hololink::csi::PixelFormat pixel_format, uint32_t trailing_bytes)
{
    HSB_LOG_INFO("start_byte={}, bytes_per_line={}, pixel_width={}, pixel_height={}, pixel_format={}, trailing_bytes={}.",
        start_byte, bytes_per_line, pixel_width, pixel_height, static_cast<int>(pixel_format), trailing_bytes);
    start_byte_ = start_byte;
    bytes_per_line_ = bytes_per_line;
    pixel_width_ = pixel_width;
    pixel_height_ = pixel_height;
    pixel_format_ = pixel_format;
    csi_length_ = start_byte + bytes_per_line * pixel_height + trailing_bytes;
    configured_ = true;
}

size_t CsiToBayerOp::get_csi_length()
{
    if (!configured_) {
        throw std::runtime_error("CsiToBayerOp is not configured.");
    }

    return csi_length_;
}

} // namespace hololink::operators
