// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "image_decoder.hpp"

#include <hololink/logging.hpp>
#include <hololink/native/cuda_helper.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/utils/cuda_stream_handler.hpp>

namespace {
const char* source = R"(
extern "C" {
__global__ void frameReconstructionZ16(unsigned short* out,
                                       const unsigned char* in,
                                       int per_line_size,
                                       int width,
                                       int height)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((idx_x >= width) || (idx_y >= height)) return;
    int out_index = idx_y * width + idx_x;
    int in_index = (per_line_size * idx_y) + idx_x * 2;
    unsigned short val = static_cast<unsigned short>(in[in_index]) |
                         (static_cast<unsigned short>(in[in_index + 1]) << 8);
    out[out_index] = val;
}

__global__ void frameReconstructionYUYV(unsigned char* out_rgb,
                                        const unsigned char* in_yuyv,
                                        int per_line_size,
                                        int width,
                                        int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= width) || (y >= height)) return;
    int pixel_pair_idx = x / 2;
    int in_idx = y * per_line_size + pixel_pair_idx * 4;
    unsigned char Y0 = in_yuyv[in_idx + 0];
    unsigned char U  = in_yuyv[in_idx + 1];
    unsigned char Y1 = in_yuyv[in_idx + 2];
    unsigned char V  = in_yuyv[in_idx + 3];
    int c = x % 2;
    unsigned char Y = (c == 0) ? Y0 : Y1;
    int C = Y - 16, D = U - 128, E = V - 128;
    int R = (298 * C + 409 * E + 128) >> 8;
    int G = (298 * C - 100 * D - 208 * E + 128) >> 8;
    int B = (298 * C + 516 * D + 128) >> 8;
    auto clip = [](int val) -> unsigned char {
        return (val < 0) ? 0 : ((val > 255) ? 255 : val);
    };
    int out_idx = (y * width + x) * 3;
    out_rgb[out_idx + 0] = clip(B);
    out_rgb[out_idx + 1] = clip(G);
    out_rgb[out_idx + 2] = clip(R);
}
})";
} // namespace

namespace hololink::operators {

void ImageDecoder::setup(holoscan::OperatorSpec& spec) {
    spec.input<holoscan::gxf::Entity>("input");
    spec.output<holoscan::gxf::Entity>("output");
    spec.param(allocator_, "allocator", "Allocator", "Memory allocator");
    spec.param(cuda_device_ordinal_, "cuda_device_ordinal", "CudaDeviceOrdinal", "CUDA device");
    spec.param(out_tensor_name_, "out_tensor_name", "OutputTensorName", "Name of output tensor");
    cuda_stream_handler_.define_params(spec);
}

void ImageDecoder::start() {
    if (pixel_format_ == PixelFormat::INVALID) throw std::runtime_error("Decoder not configured");
    CudaCheck(cuInit(0));
    CudaCheck(cuDeviceGet(&cuda_device_, cuda_device_ordinal_.get()));
    CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));
    hololink::native::CudaContextScopedPush cur_cuda_context(cuda_context_);
    cuda_function_launcher_.reset(new hololink::native::CudaFunctionLauncher(
        source, {"frameReconstructionZ16", "frameReconstructionYUYV"}));
    colorizer_ = std::make_unique<rs2::utils::Colorizer>();
}

void ImageDecoder::stop() {
    hololink::native::CudaContextScopedPush cur_cuda_context(cuda_context_);
    cuda_function_launcher_.reset();
    CudaCheck(cuDevicePrimaryCtxRelease(cuda_device_));
    cuda_context_ = nullptr;
}

void ImageDecoder::compute(holoscan::InputContext& input, holoscan::OutputContext& output,
                           holoscan::ExecutionContext& context) {
    auto maybe_entity = input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) throw std::runtime_error("No input entity");
    auto& entity = static_cast<nvidia::gxf::Entity&>(maybe_entity.value());
    gxf_result_t stream_handler_result = cuda_stream_handler_.from_message(context.context(), entity);
    if (stream_handler_result != GXF_SUCCESS) throw std::runtime_error("Failed to get stream");
    auto input_tensor = entity.get<nvidia::gxf::Tensor>().value();
    if (input_tensor->rank() != 1) throw std::runtime_error("Tensor must be 1D");

    const int32_t size = input_tensor->shape().dimension(0);
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        fragment()->executor().context(), allocator_->gxf_cid());
    const uint32_t per_line_size = line_start_size_ + bytes_per_line_ + line_end_size_;

    switch (pixel_format_) {
    case PixelFormat::Z16: {
        // 1. Reconstruct Z16 tensor
        nvidia::gxf::Shape depth_shape{int(height_), int(width_), 1};
        auto depth_message = CreateTensorMap(context.context(), allocator.value(), {{
            "depth", nvidia::gxf::MemoryStorageType::kDevice, depth_shape,
            nvidia::gxf::PrimitiveType::kUnsigned16, 0,
            nvidia::gxf::ComputeTrivialStrides(depth_shape, 2)}});
        auto depth_tensor = depth_message.value().get<nvidia::gxf::Tensor>("depth");

        cuda_function_launcher_->launch("frameReconstructionZ16", {width_, height_, 1},
            cuda_stream_handler_.get_cuda_stream(context.context()),
            depth_tensor.value()->pointer(),
            input_tensor->pointer() + frame_start_size_ + line_start_size_,
            per_line_size, width_, height_);

        // 2. Copy to host
        std::vector<uint16_t> host_depth(width_ * height_);
        cudaMemcpy(host_depth.data(), depth_tensor.value()->pointer(),
                host_depth.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);

        // 3. Colorize
        std::vector<uint8_t> rgb_data(width_ * height_ * 3);
        colorizer_->set_equalize(true);
        colorizer_->set_depth_units(0.001f);  // mm to m
        colorizer_->colorize(host_depth.data(), rgb_data.data(), width_, height_);

        // 4. Allocate RGB tensor on GPU
        nvidia::gxf::Shape rgb_shape{int(height_), int(width_), 3};
        auto out_message = CreateTensorMap(context.context(), allocator.value(), {{
            out_tensor_name_.get(), nvidia::gxf::MemoryStorageType::kDevice, rgb_shape,
            nvidia::gxf::PrimitiveType::kUnsigned8, 0,
            nvidia::gxf::ComputeTrivialStrides(rgb_shape, 1)}});
        auto rgb_tensor = out_message.value().get<nvidia::gxf::Tensor>(out_tensor_name_.get().c_str());

        // 5. Copy RGB data to GPU
        cudaMemcpy(rgb_tensor.value()->pointer(), rgb_data.data(),
                rgb_data.size(), cudaMemcpyHostToDevice);

        // 6. Emit colorized RGB image
        stream_handler_result = cuda_stream_handler_.to_message(out_message);
        auto& out_entity = out_message.value();
        output.emit(out_entity);
        return;
    }
    case PixelFormat::YUYV: {
        nvidia::gxf::Shape shape{int(height_), int(width_), 3};
        auto out_message = CreateTensorMap(context.context(), allocator.value(), {{
            out_tensor_name_.get(), nvidia::gxf::MemoryStorageType::kDevice, shape,
            nvidia::gxf::PrimitiveType::kUnsigned8, 0,
            nvidia::gxf::ComputeTrivialStrides(shape, 1)}});
        auto tensor = out_message.value().get<nvidia::gxf::Tensor>(out_tensor_name_.get().c_str());
        cuda_function_launcher_->launch("frameReconstructionYUYV", {width_ / 2, height_, 1},
            cuda_stream_handler_.get_cuda_stream(context.context()),
            tensor.value()->pointer(),
            input_tensor->pointer() + frame_start_size_ + line_start_size_,
            per_line_size, width_, height_);
        stream_handler_result = cuda_stream_handler_.to_message(out_message);
        auto& out_entity = out_message.value();
        output.emit(out_entity);
        return;
    }
    default:
        throw std::runtime_error("Unsupported pixel format");
    }
}

void ImageDecoder::configure(uint32_t width, uint32_t height, PixelFormat pixel_format,
                             uint32_t frame_start_size, uint32_t frame_end_size,
                             uint32_t line_start_size, uint32_t line_end_size,
                             uint32_t margin_left, uint32_t margin_top,
                             uint32_t margin_right, uint32_t margin_bottom) {
    width_ = width;
    height_ = height;
    pixel_format_ = pixel_format;
    frame_start_size_ = frame_start_size;
    frame_end_size_ = frame_end_size;
    line_start_size_ = line_start_size;
    line_end_size_ = line_end_size;
    switch (pixel_format_) {
    case PixelFormat::Z16:
        bytes_per_line_ = width * 2;
        line_start_size_ += margin_left * 2;
        line_end_size_ += margin_right * 2;
        break;
    case PixelFormat::YUYV:
        bytes_per_line_ = width * 2;
        line_start_size_ += margin_left * 2;
        line_end_size_ += margin_right * 2;
        break;
    default:
        throw std::runtime_error("Unsupported pixel format");
    }
    const uint32_t line_size = line_start_size_ + bytes_per_line_ + line_end_size_;
    frame_start_size_ += margin_top * line_size;
    frame_end_size_ += margin_bottom * line_size;
    csi_length_ = (frame_start_size_ + line_size * height_ + frame_end_size_ + 7) & ~7;
}

size_t ImageDecoder::get_csi_length() {
    if (pixel_format_ == PixelFormat::INVALID) {
        throw std::runtime_error("ImageDecoder is not configured.");
    }
    return csi_length_;
}

} // namespace hololink::operators