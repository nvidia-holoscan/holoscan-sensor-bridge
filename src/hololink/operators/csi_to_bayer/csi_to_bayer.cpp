/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <hololink/logging.hpp>
#include <hololink/native/cuda_helper.hpp>
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

static inline size_t align_8(size_t value) { return (value + 7) & ~7; }

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
    if (pixel_format_ == PixelFormat::INVALID) {
        throw std::runtime_error("CsiToBayerOp is not configured.");
    }

    CudaCheck(cuInit(0));
    CudaCheck(cuDeviceGet(&cuda_device_, cuda_device_ordinal_.get()));
    CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));
    int integrated = 0;
    CudaCheck(cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, cuda_device_));
    is_integrated_ = (integrated != 0);

    hololink::native::CudaContextScopedPush cur_cuda_context(cuda_context_);

    cuda_function_launcher_.reset(new hololink::native::CudaFunctionLauncher(
        source, { "frameReconstruction8", "frameReconstruction10", "frameReconstruction12" }));
}

void CsiToBayerOp::stop()
{
    hololink::native::CudaContextScopedPush cur_cuda_context(cuda_context_);

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

    const int32_t size = input_tensor->shape().dimension(0);

    // get handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        fragment()->executor().context(), allocator_->gxf_cid());

    // create the output
    nvidia::gxf::Shape shape { int(height_), int(width_), 1 };
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

    hololink::native::CudaContextScopedPush cur_cuda_context(cuda_context_);
    const cudaStream_t cuda_stream = cuda_stream_handler_.get_cuda_stream(context.context());

    const uint32_t per_line_size = line_start_size_ + bytes_per_line_ + line_end_size_;
    switch (pixel_format_) {
    case hololink::operators::CsiToBayerOp::PixelFormat::RAW_8:
        cuda_function_launcher_->launch("frameReconstruction8", { width_, height_, 1 }, cuda_stream,
            tensor.value()->pointer(),
            input_tensor->pointer() + frame_start_size_ + line_start_size_, per_line_size, width_,
            height_);
        break;
    case hololink::operators::CsiToBayerOp::PixelFormat::RAW_10:
        cuda_function_launcher_->launch("frameReconstruction10",
            { width_ / 4, // outputs 4 pixels per shader invocation
                height_, 1 },
            cuda_stream, tensor.value()->pointer(),
            input_tensor->pointer() + frame_start_size_ + line_start_size_, per_line_size, width_,
            height_);
        break;
    case hololink::operators::CsiToBayerOp::PixelFormat::RAW_12:
        cuda_function_launcher_->launch("frameReconstruction12",
            { width_ / 2, // outputs 2 pixels per shader invocation
                height_, 1 },
            cuda_stream, tensor.value()->pointer(),
            input_tensor->pointer() + frame_start_size_ + line_start_size_, per_line_size, width_,
            height_);
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

void CsiToBayerOp::configure(uint32_t width, uint32_t height, PixelFormat pixel_format,
    uint32_t frame_start_size, uint32_t frame_end_size, uint32_t line_start_size,
    uint32_t line_end_size, uint32_t margin_left, uint32_t margin_top, uint32_t margin_right,
    uint32_t margin_bottom)
{
    width_ = width;
    height_ = height;
    pixel_format_ = pixel_format;
    frame_start_size_ = frame_start_size;
    frame_end_size_ = frame_end_size;
    line_start_size_ = line_start_size;
    line_end_size_ = line_end_size;

    switch (pixel_format_) {
    case hololink::operators::CsiToBayerOp::PixelFormat::RAW_8:
        bytes_per_line_ = width_;
        line_start_size_ += margin_left;
        line_end_size_ += margin_right;
        break;
    case hololink::operators::CsiToBayerOp::PixelFormat::RAW_10:
        bytes_per_line_ = width_ * 5 / 4;
        line_start_size_ += margin_left * 5 / 4;
        line_end_size_ += margin_right * 5 / 4;
        break;
    case hololink::operators::CsiToBayerOp::PixelFormat::RAW_12:
        bytes_per_line_ = width_ * 3 / 2;
        line_start_size_ += margin_left * 3 / 2;
        line_end_size_ += margin_right * 3 / 2;
        break;
    default:
        throw std::runtime_error(fmt::format("Unsupported pixel format {}", int(pixel_format_)));
    }

    const uint32_t line_size = line_start_size + bytes_per_line_ + line_end_size;
    frame_start_size_ += margin_top * line_size;
    frame_end_size_ += margin_bottom * line_size;
    // NOTE that this align_8 is not a CSI specification; instead it comes
    // from the Sensor Bridge FPGA implementation.  When we convert Hololink
    // to C++, let's change this to a callback to that object, per
    // https://jirasw.nvidia.com/browse/BAJQ0XTT-137.
    csi_length_ = align_8(frame_start_size_ + line_size * height_ + frame_end_size_);
}

size_t CsiToBayerOp::get_csi_length()
{
    if (pixel_format_ == PixelFormat::INVALID) {
        throw std::runtime_error("CsiToBayerOp is not configured.");
    }

    return csi_length_;
}

} // namespace hololink::operators
