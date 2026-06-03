/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
                                      int quater_width,
                                      int height)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx_x >= quater_width) || (idx_y >= height))
        return;

    const int in_index = (per_line_size * idx_y) + idx_x * 5;
    const int out_index = (idx_y * quater_width + idx_x) * 4;

    const unsigned short lsbs = in[in_index + 4];
    out[out_index + 0] = ((in[in_index + 0] << 2) | (lsbs & 0x03)) << 6;
    out[out_index + 1] = ((in[in_index + 1] << 4) | (lsbs & 0x0C)) << 4;
    out[out_index + 2] = ((in[in_index + 2] << 6) | (lsbs & 0x30)) << 2;
    out[out_index + 3] = ((in[in_index + 3] << 8) | (lsbs & 0xC0));
}

__global__ void frameReconstruction12(unsigned short * out,
                                      const unsigned char * in,
                                      int per_line_size,
                                      int half_width,
                                      int height)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx_x >= half_width) || (idx_y >= height))
        return;

    const int in_index = (per_line_size * idx_y) + idx_x * 3;
    const int out_index = (idx_y * half_width + idx_x) * 2;

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
    spec.param(sub_frame_rows_, "sub_frame_rows", "SubFrameRows",
        "Number of rows in a sub-frame, if 0 (the default), the full frame will be used", 0U);
}

void CsiToBayerOp::start()
{
    if (!configured_) {
        throw std::runtime_error("CsiToBayerOp is not configured.");
    }

    if (sub_frame_rows_.get() > 0) {
        if (pixel_height_ % sub_frame_rows_.get() != 0) {
            throw std::runtime_error(fmt::format("Height of {} is not evenly divisible by sub_frame_rows of {}",
                pixel_height_, sub_frame_rows_.get()));
        }
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

    if (sub_frame_rows_.get() > 0) {
        // allocate memory for combined sub-frame data
        sub_frame_memory_size_ = bytes_per_line_ * sub_frame_rows_.get();
        sub_frame_memory_.reset([this]() {
            CUdeviceptr ptr;
            CudaCheck(cuMemAlloc(&ptr, sub_frame_memory_size_));
            return ptr;
        }());
    }
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

    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);
    // Get the CUDA stream from the input message if present, otherwise generate one.
    // This stream will also be transmitted on the output port.
    cudaStream_t cuda_stream = input.receive_cuda_stream();

    byte* input_pointer;
    uint32_t cur_pixel_height;
    size_t sub_frame_copy_size = 0;
    if (sub_frame_rows_.get() > 0) {
        // sub-frame handling
        input_pointer = input_tensor->pointer();

        const int64_t receiver_frame_number = metadata()->get<int64_t>("frame_number");
        const int64_t receiver_sub_frames_per_frame = (csi_length_ + sub_frame_memory_size_ - 1) / sub_frame_memory_size_;
        const int64_t receiver_sub_frame = receiver_frame_number % receiver_sub_frames_per_frame;
        const int64_t receiver_sub_frame_offset = int64_t(receiver_sub_frame * sub_frame_memory_size_);

        size_t input_size = input_tensor->size();
        int64_t sub_frame_offset;
        if (receiver_sub_frame == 0) {
            // skip the start offset of the first sub-frame
            if (input_size < start_byte_) {
                throw std::runtime_error(
                    fmt::format("Input tensor size {} is smaller than configured start_byte_ {}",
                        input_size, start_byte_));
            }
            input_pointer += start_byte_;
            input_size -= start_byte_;
            sub_frame_offset = 0;
        } else {
            sub_frame_offset = receiver_sub_frame_offset - start_byte_;
        }
        const size_t sub_frame_memory_offset = sub_frame_offset % sub_frame_memory_size_;
        const size_t remaining_size = sub_frame_memory_size_ - sub_frame_memory_offset;

        const size_t copy_size = std::min(input_size, remaining_size);
        CudaCheck(cuMemcpyAsync(sub_frame_memory_.get() + sub_frame_memory_offset, (CUdeviceptr)input_pointer, copy_size, cuda_stream));

        if (sub_frame_memory_offset + copy_size != sub_frame_memory_size_) {
            // partial sub-frame, wait for the rest
            return;
        }
        sub_frame_copy_size = input_size - copy_size;

        // replace the byte based sub-frame offset with the line based sub-frame offset
        metadata()->erase("sub_frame_offset");
        const int64_t sub_frame = sub_frame_offset / sub_frame_memory_size_;
        metadata()->set<int64_t>("sub_frame_offset", sub_frame * sub_frame_rows_);

        // replace the sub-frame number with the full frame number
        metadata()->erase("frame_number");
        metadata()->set<int64_t>("frame_number", receiver_frame_number / receiver_sub_frames_per_frame);

        input_pointer = reinterpret_cast<byte*>((CUdeviceptr)(sub_frame_memory_.get()));
        cur_pixel_height = sub_frame_rows_.get();
    } else {
        input_pointer = input_tensor->pointer() + start_byte_;
        cur_pixel_height = pixel_height_;
    }

    // get handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        fragment()->executor().context(), allocator_->gxf_cid());

    // create the output
    nvidia::gxf::Shape shape { int(cur_pixel_height), int(pixel_width_), 1 };
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

    switch (pixel_format_) {
    case hololink::csi::PixelFormat::RAW_8:
        cuda_function_launcher_->launch("frameReconstruction8", { pixel_width_, cur_pixel_height, 1 }, cuda_stream,
            tensor.value()->pointer(),
            input_pointer, bytes_per_line_, pixel_width_,
            cur_pixel_height);
        break;
    case hololink::csi::PixelFormat::RAW_10:
        cuda_function_launcher_->launch("frameReconstruction10",
            { pixel_width_ / 4, // outputs 4 pixels per shader invocation
                cur_pixel_height, 1 },
            cuda_stream, tensor.value()->pointer(),
            input_pointer, bytes_per_line_, pixel_width_ / 4,
            cur_pixel_height);
        break;
    case hololink::csi::PixelFormat::RAW_12:
        cuda_function_launcher_->launch("frameReconstruction12",
            { pixel_width_ / 2, // outputs 2 pixels per shader invocation
                cur_pixel_height, 1 },
            cuda_stream, tensor.value()->pointer(),
            input_pointer, bytes_per_line_, pixel_width_ / 2,
            cur_pixel_height);
        break;
    default:
        throw std::runtime_error("Unsupported bits per pixel value");
    }

    if (sub_frame_copy_size > 0) {
        // copy the remaining data to the beginning of the sub-frame memory
        CudaCheck(cuMemcpyAsync(sub_frame_memory_.get(),
            (CUdeviceptr)(input_tensor->pointer() + input_tensor->size() - sub_frame_copy_size),
            sub_frame_copy_size, cuda_stream));
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

    // the operator is not yet initialized when this function is called, so we need to find the
    // sub_frame_rows argument from the args() vector.
    uint32_t sub_frame_rows = sub_frame_rows_.default_value();
    for (auto& arg : args()) {
        if (arg.name() == "sub_frame_rows") {
            sub_frame_rows = std::any_cast<uint32_t>(arg.value());
        }
    }
    if (sub_frame_rows != 0) {
        frame_size_ = bytes_per_line_ * sub_frame_rows;
    } else {
        frame_size_ = csi_length_;
    }

    configured_ = true;
}

size_t CsiToBayerOp::get_csi_length()
{
    if (!configured_) {
        throw std::runtime_error("CsiToBayerOp is not configured.");
    }

    return csi_length_;
}

size_t CsiToBayerOp::get_sub_frame_size()
{
    if (!configured_) {
        throw std::runtime_error("CsiToBayerOp is not configured.");
    }

    return frame_size_;
}

} // namespace hololink::operators
