/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Native module packed-CSI -> 16-bit converter operator. Self-contained: it
 * carries its own unpack engine (the NVRTC packed8bitTo16bit /
 * packed10bitTo16bit / packed12bitTo16bit kernels), expressed against the
 * module hololink::module::csi::PixelFormat.
 */

#include "hololink/module/operators/packed_format_converter_op.hpp"

#include <stdexcept>

#include <fmt/format.h>
#include <gxf/std/tensor.hpp>
#include <holoscan/holoscan.hpp>

#include "hololink/module/cuda_unique.hpp" // HOLOLINK_MODULE_CUDA_CHECK
#include "hololink/module/logging.hpp" // HSB_LOG_*
#include "hololink/module/operators/cuda_function_launcher.hpp" // CudaFunctionLauncher, CudaContextScopedPush
#include "hololink/module/page_size.hpp" // round_up

namespace {

const char* source = R"(
extern "C" {

// Converts packed 8-bit data into 16-bit data.
__global__ void packed8bitTo16bit(unsigned short* output, const unsigned char* input,
                                  int bytes_per_line, int width, int height) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx_x >= width) || (idx_y >= height))
        return;

    int input_index = (idx_y * bytes_per_line) + idx_x;
    int output_index = (idx_y * width) + idx_x;

    output[output_index] = (((unsigned short)input[input_index]) << 8) & 0xFF00;
}

// Converts packed 10-bit data, where each 4-byte dword contains 3 pixels, into 16-bit data.
__global__ void packed10bitTo16bit(unsigned short* output, const unsigned int* input,
                                   int bytes_per_line, int width, int height) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    // 3 pixels per 4-byte group; round up so the final partial group (width
    // not a multiple of 3) is processed instead of dropped, and guard each
    // write so a partial group doesn't spill into the next row.
    if ((idx_x >= (width + 2) / 3) || (idx_y >= height))
        return;

    int input_index = (idx_y * bytes_per_line / 4) + idx_x;
    int output_index = (idx_y * width) + (idx_x * 3);
    int col = idx_x * 3;

    unsigned int packed = input[input_index];
    if (col + 0 < width) output[output_index]     = (packed << 6) & 0xFFC0;
    if (col + 1 < width) output[output_index + 1] = (packed >> 4) & 0xFFC0;
    if (col + 2 < width) output[output_index + 2] = (packed >> 14) & 0xFFC0;
}

// Converts packed 12-bit data, where 2 pixels are stored in every 3 bytes.
// Byte layout (matches the COE/FUSA RAW12 stream and is consistent with
// packed10bitTo16bit's pixel-0-first ordering):
//   byte[0]      = pixel0[7:0]
//   byte[1][3:0] = pixel0[11:8]
//   byte[1][7:4] = pixel1[3:0]
//   byte[2]      = pixel1[11:4]
__global__ void packed12bitTo16bit(unsigned short* output, const unsigned char* input,
                                   int bytes_per_line, int width, int height) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    // 2 pixels per 3-byte group; round up so the final partial group (odd
    // width) is processed instead of dropped, and guard each write so a
    // partial group doesn't spill into the next row.
    if ((idx_x >= (width + 1) / 2) || (idx_y >= height))
        return;

    int input_index = (idx_y * bytes_per_line) + (idx_x * 3);
    int output_index = (idx_y * width) + (idx_x * 2);
    int col = idx_x * 2;

    if (col + 0 < width) output[output_index]     = ((input[input_index + 1] & 0x0F) << 12 |
                                                    input[input_index]            <<  4) & 0xFFF0;
    if (col + 1 < width) output[output_index + 1] = (input[input_index + 2]        <<  8 |
                                                    (input[input_index + 1] & 0xF0)    ) & 0xFFF0;
}

})";

} // anonymous namespace

namespace hololink::module::operators {

void PackedFormatConverterOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input<holoscan::gxf::Entity>("input");
    spec.output<holoscan::gxf::Entity>("output");

    spec.param(allocator_, "allocator", "Allocator",
        "Allocator used to allocate the output image, defaults to BlockMemoryPool");
    spec.param(cuda_device_ordinal_, "cuda_device_ordinal", "CudaDeviceOrdinal",
        "Device to use for CUDA operations", 0);
    spec.param(in_tensor_name_, "in_tensor_name", "InputTensorName",
        "Name of the input tensor", std::string(""));
    spec.param(out_tensor_name_, "out_tensor_name", "OutputTensorName",
        "Name of the output tensor", std::string(""));
}

void PackedFormatConverterOp::start()
{
    if (!configured_) {
        throw std::runtime_error("PackedFormatConverterOp is not configured.");
    }

    HOLOLINK_MODULE_CUDA_CHECK(cuInit(0));
    HOLOLINK_MODULE_CUDA_CHECK(cuDeviceGet(&cuda_device_, cuda_device_ordinal_.get()));
    HOLOLINK_MODULE_CUDA_CHECK(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));
    int integrated = 0;
    HOLOLINK_MODULE_CUDA_CHECK(cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, cuda_device_));
    is_integrated_ = (integrated != 0);

    CudaContextScopedPush cur_cuda_context(cuda_context_);

    cuda_function_launcher_.reset(new CudaFunctionLauncher(
        source, { "packed8bitTo16bit", "packed10bitTo16bit", "packed12bitTo16bit" }));
}

void PackedFormatConverterOp::stop()
{
    CudaContextScopedPush cur_cuda_context(cuda_context_);

    cuda_function_launcher_.reset();

    HOLOLINK_MODULE_CUDA_CHECK(cuDevicePrimaryCtxRelease(cuda_device_));
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

    CudaContextScopedPush cur_cuda_context(cuda_context_);
    // Get the CUDA stream from the input message if present, otherwise generate one.
    // This stream will also be transmitted on the output port.
    const cudaStream_t cuda_stream = input.receive_cuda_stream();

    switch (pixel_format_) {
    case hololink::module::csi::PixelFormat::RAW_8:
        cuda_function_launcher_->launch("packed8bitTo16bit",
            { pixel_width_, pixel_height_, 1 },
            cuda_stream, tensor.value()->pointer(),
            input_tensor->pointer() + start_byte_,
            bytes_per_line_, pixel_width_, pixel_height_);
        break;
    case hololink::module::csi::PixelFormat::RAW_10:
        cuda_function_launcher_->launch("packed10bitTo16bit",
            { (pixel_width_ + 2) / 3, // outputs 3 pixels per shader invocation (round up for the tail)
                pixel_height_, 1 },
            cuda_stream, tensor.value()->pointer(),
            input_tensor->pointer() + start_byte_,
            bytes_per_line_, pixel_width_, pixel_height_);
        break;
    case hololink::module::csi::PixelFormat::RAW_12:
        cuda_function_launcher_->launch("packed12bitTo16bit",
            { (pixel_width_ + 1) / 2, // outputs 2 pixels per shader invocation (round up for the tail)
                pixel_height_, 1 },
            cuda_stream, tensor.value()->pointer(),
            input_tensor->pointer() + start_byte_,
            bytes_per_line_, pixel_width_, pixel_height_);
        break;
    default:
        throw std::runtime_error("Unsupported bits per pixel value");
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
    return hololink::module::round_up(transmitted_line_bytes, 64);
}

uint32_t PackedFormatConverterOp::transmitted_line_bytes(
    hololink::module::csi::PixelFormat pixel_format, uint32_t pixel_width)
{
    switch (pixel_format) {
    case hololink::module::csi::PixelFormat::RAW_8:
        return pixel_width;
    case hololink::module::csi::PixelFormat::RAW_10:
        // 3 pixels per 4 bytes
        return ((pixel_width + 2) / 3) * 4;
    case hololink::module::csi::PixelFormat::RAW_12:
        // 2 pixels per 3 bytes
        return ((pixel_width + 1) / 2) * 3;
    default:
        throw std::runtime_error("Invalid bit depth");
    }
}

void PackedFormatConverterOp::configure(uint32_t start_byte, uint32_t bytes_per_line,
    uint32_t pixel_width, uint32_t pixel_height,
    hololink::module::csi::PixelFormat pixel_format, uint32_t trailing_bytes)
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

} // namespace hololink::module::operators
