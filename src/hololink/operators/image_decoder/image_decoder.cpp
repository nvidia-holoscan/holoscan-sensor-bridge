// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "image_decoder.hpp"

#include <hololink/core/logging_internal.hpp>
#include <hololink/common/cuda_helper.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/utils/cuda_stream_handler.hpp>

namespace {
const char* source = R"(
extern "C" {

typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;

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

__global__ void frameReconstructionYUYV(uint8_t* out_rgb,
                                        const uint8_t* in_yuyv,
                                        int per_line_size,
                                        int width,
                                        int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= width) || (y >= height)) return;
    int pixel_pair_idx = x / 2;
    int in_idx = y * per_line_size + pixel_pair_idx * 4;
    uint8_t Y0 = in_yuyv[in_idx + 0];
    uint8_t U  = in_yuyv[in_idx + 1];
    uint8_t Y1 = in_yuyv[in_idx + 2];
    uint8_t V  = in_yuyv[in_idx + 3];
    int c = x % 2;
    uint8_t Y = (c == 0) ? Y0 : Y1;
    int C = Y - 16, D = U - 128, E = V - 128;
    int R = (298 * C + 409 * E + 128) >> 8;
    int G = (298 * C - 100 * D - 208 * E + 128) >> 8;
    int B = (298 * C + 516 * D + 128) >> 8;
    R = R < 0 ? 0 : (R > 255 ? 255 : R);
    G = G < 0 ? 0 : (G > 255 ? 255 : G);
    B = B < 0 ? 0 : (B > 255 ? 255 : B);
    int out_idx = (y * width + x) * 3;
    out_rgb[out_idx + 0] = R;
    out_rgb[out_idx + 1] = G;
    out_rgb[out_idx + 2] = B;

}

__global__ void compute_histogram(const uint16_t* depth, int* hist, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    uint16_t d = depth[idx];
    if (d > 0 && d < 65536) atomicAdd(&hist[d], 1);
}

__global__ void prefix_sum_histogram(int* hist, int size) {
    for (int i = 1; i < size; ++i) {
        hist[i] += hist[i - 1];
    }
}

__device__ inline float3 interpolate_colormap(float value, const float3* colormap, int colormap_size) {
    float t = fminf(fmaxf(value, 0.f), 1.f) * (colormap_size - 1);
    int idx = (int)t;
    float frac = t - idx;
    float3 lo = colormap[idx];
    float3 hi = colormap[min(idx + 1, colormap_size - 1)];
    return make_float3(lo.x * (1.f - frac) + hi.x * frac,
                       lo.y * (1.f - frac) + hi.y * frac,
                       lo.z * (1.f - frac) + hi.z * frac);
}

__global__ void depthToRGB(uint8_t* out_rgb,
                           const uint16_t* depth,
                           const int* hist,
                           int width,
                           int height,
                           float depth_units,
                           float min_m,
                           float max_m,
                           bool equalize,
                           const float3* colormap,
                           int colormap_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    uint16_t d = depth[idx];

    if (d == 0) {
        out_rgb[3 * idx + 0] = 0;
        out_rgb[3 * idx + 1] = 0;
        out_rgb[3 * idx + 2] = 0;
        return;
    }

    float norm;
    if (equalize) {
        int total_hist = hist[65535];
        norm = (total_hist > 0) ? (float)(hist[d]) / total_hist : 0.f;
    } else {
        float depth_m = d * depth_units;
        norm = (depth_m - min_m) / (max_m - min_m);
        norm = fminf(fmaxf(norm, 0.f), 1.f);
    }

    float3 c = interpolate_colormap(norm, colormap, colormap_size);
    out_rgb[3 * idx + 0] = (uint8_t)(c.x);
    out_rgb[3 * idx + 1] = (uint8_t)(c.y);
    out_rgb[3 * idx + 2] = (uint8_t)(c.z);
}

}
)";
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
    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);
    cuda_function_launcher_.reset(new hololink::common::CudaFunctionLauncher(
        source, {"frameReconstructionZ16", "frameReconstructionYUYV","depthToRGB", "compute_histogram",
             "prefix_sum_histogram"}));
    
    // Allocate d_hist_ (256KB)
    cudaMalloc(&d_hist_, sizeof(int) * 0x10000);

    // Allocate and upload colormap
    std::vector<float3> colormap = {
        {0.f, 0.f, 255.f},    // Blue
        {0.f, 255.f, 255.f},  // Cyan
        {255.f, 255.f, 0.f},  // Yellow
        {255.f, 0.f, 0.f},    // Red
        {50.f, 0.f, 0.f}      // Dark red
    };
    colormap_size_ = colormap.size();
    cudaMalloc(&d_colormap_, colormap_size_ * sizeof(float3));
    cudaMemcpy(d_colormap_, colormap.data(), colormap_size_ * sizeof(float3), cudaMemcpyHostToDevice);
}

void ImageDecoder::stop() {
    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);
    cuda_function_launcher_.reset();

    if (d_hist_) {
        cudaFree(d_hist_);
        d_hist_ = nullptr;
    }

    if (d_colormap_) {
        cudaFree(d_colormap_);
        d_colormap_ = nullptr;
        colormap_size_ = 0;
    }

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

    if (input_tensor->rank() != 1) throw std::runtime_error("Tensor must be 1D");

    const int32_t size = input_tensor->shape().dimension(0);
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        fragment()->executor().context(), allocator_->gxf_cid());
    const uint32_t per_line_size = line_start_size_ + bytes_per_line_ + line_end_size_;

    switch (pixel_format_) {
    case PixelFormat::Z16: {
        // 1. Allocate depth tensor (device)
        nvidia::gxf::Shape depth_shape{int(height_), int(width_), 1};
        auto depth_message = CreateTensorMap(context.context(), allocator.value(), {{
            "depth", nvidia::gxf::MemoryStorageType::kDevice, depth_shape,
            nvidia::gxf::PrimitiveType::kUnsigned16, 0,
            nvidia::gxf::ComputeTrivialStrides(depth_shape, 2)}}, false);
        auto depth_tensor = depth_message.value().get<nvidia::gxf::Tensor>("depth");

        // 2. Reconstruct depth frame from CSI
        cuda_function_launcher_->launch("frameReconstructionZ16", {width_, height_, 1},
            cuda_stream_handler_.get_cuda_stream(context.context()),
            depth_tensor.value()->pointer(),
            input_tensor->pointer() + frame_start_size_ + line_start_size_,
            per_line_size, width_, height_);

        // 3. Allocate RGB tensor (device)
        nvidia::gxf::Shape rgb_shape{int(height_), int(width_), 3};
        auto out_message = CreateTensorMap(context.context(), allocator.value(), {{
            out_tensor_name_.get(), nvidia::gxf::MemoryStorageType::kDevice, rgb_shape,
            nvidia::gxf::PrimitiveType::kUnsigned8, 0,
            nvidia::gxf::ComputeTrivialStrides(rgb_shape, 1)}}, false);
        auto rgb_tensor = out_message.value().get<nvidia::gxf::Tensor>(out_tensor_name_.get().c_str());

        // 4. GPU histogram
        cudaMemsetAsync(d_hist_, 0, sizeof(int) * 0x10000, cuda_stream_handler_.get_cuda_stream(context.context()));

        cuda_function_launcher_->launch("compute_histogram", {width_, height_, 1},
            cuda_stream_handler_.get_cuda_stream(context.context()),
            depth_tensor.value()->pointer(), d_hist_, width_ * height_);

        cuda_function_launcher_->launch("prefix_sum_histogram", {1, 1, 1},
            cuda_stream_handler_.get_cuda_stream(context.context()),
            d_hist_,
            0x10000);

        // 5. Run GPU depth → RGB colorizer
        cuda_function_launcher_->launch("depthToRGB", {width_, height_, 1},
            cuda_stream_handler_.get_cuda_stream(context.context()),
            rgb_tensor.value()->pointer(),
            depth_tensor.value()->pointer(),
            d_hist_,
            width_, height_,
            0.001f,
            0.3f, 4.0f,
            false,
            d_colormap_,
            static_cast<int>(colormap_size_));


        // 6. Emit output
        stream_handler_result = cuda_stream_handler_.to_message(out_message);
        if (stream_handler_result != GXF_SUCCESS) {
            throw std::runtime_error("Failed to emit RGB image");
        }
        // auto& out_entity = out_message.value();
        auto out_entity = holoscan::gxf::Entity(std::move(out_message.value()));
        output.emit(out_entity);
        return;
    }
    case PixelFormat::YUYV: {
        nvidia::gxf::Shape shape{int(height_), int(width_), 3};
        auto out_message = CreateTensorMap(context.context(), allocator.value(), {{
            out_tensor_name_.get(), nvidia::gxf::MemoryStorageType::kDevice, shape,
            nvidia::gxf::PrimitiveType::kUnsigned8, 0,
            nvidia::gxf::ComputeTrivialStrides(shape, 1)}}, false);
        auto tensor = out_message.value().get<nvidia::gxf::Tensor>(out_tensor_name_.get().c_str());
        cuda_function_launcher_->launch("frameReconstructionYUYV", {width_, height_, 1},
            cuda_stream_handler_.get_cuda_stream(context.context()),
            tensor.value()->pointer(),
            input_tensor->pointer() + frame_start_size_ + line_start_size_,
            per_line_size, width_, height_);
        stream_handler_result = cuda_stream_handler_.to_message(out_message);
        auto out_entity = holoscan::gxf::Entity(std::move(out_message.value()));
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