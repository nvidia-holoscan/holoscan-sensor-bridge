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

#include "image_processor.hpp"

#include <hololink/native/cuda_helper.hpp>
#include <holoscan/holoscan.hpp>

namespace {

const char* source = R"(
#include <device_atomic_functions.h>
#include <cooperative_groups.h>

extern "C" {

// bayer component offsets
__inline__ __device__ unsigned int getBayerOffset(unsigned int x, unsigned int y)
{
    const unsigned int offsets[2][2]{{X0Y0_OFFSET, X1Y0_OFFSET}, {X0Y1_OFFSET, X1Y1_OFFSET}};
    return offsets[y & 1][x & 1];
}

/**
 * Apply black level correction.
 *
 * @param image [in] pointer to input image
 * @param components_per_line [in] componets per input image line (width * 3 for RGB)
 * @param height [in] height of the input image
 */
__global__ void applyBlackLevel(unsigned short *image,
                                int components_per_line,
                                int height)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx_x >= components_per_line) || (idx_y >= height))
        return;

    const int index = idx_y * components_per_line + idx_x;

    // subtract optical black and clamp
    float value = max(float(image[index]) - OPTICAL_BLACK, 0.f);
    // fix white level
    const float range = (1 << (sizeof(unsigned short) * 8)) - 1;
    value *= range / (range - float(OPTICAL_BLACK));
    image[index] = (unsigned short)(value + 0.5f);
}

/**
 * Calculate the histogram of an image.
 *
 * Based on the Cuda SDK histogram256 sample.
 *
 * First each warp of a thread builds a sub-histogram in shared memory. Then the per-warp
 * sub-histograms are merged per block and written to global memory using atomics.
 *
 * Note, this kernel needs HISTOGRAM_THREADBLOCK_MEMORY bytes of shared
 * memory.
 *
 * @param in [in] pointer to image data
 * @param histogram [in] pointer to the histogram data (must be able to hold HISTOGRAM_BIN_COUNT values)
 * @param width [in] width of the image
 * @param height [in] height of the image
 */
__global__ void histogram(const unsigned short *in,
                          unsigned int *histogram,
                          unsigned int width,
                          unsigned int height)
{
    uint2 index = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (index.y >= height)
    {
        return;
    }

    // per-warp subhistogram storage
    __shared__ unsigned int s_hist[HISTOGRAM_THREADBLOCK_MEMORY / sizeof(unsigned int)];

    // clear shared memory storage for current threadblock before processing
    if (threadIdx.y == 0)
    {
#pragma unroll
        for (int i = 0; i < ((HISTOGRAM_THREADBLOCK_MEMORY / sizeof(unsigned int)) / HISTOGRAM_THREADBLOCK_SIZE); ++i)
        {
            s_hist[threadIdx.x + i * HISTOGRAM_THREADBLOCK_SIZE] = 0;
        }
    }

    // handle to thread block group
    cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();

    cooperative_groups::sync(cta);

    // cycle through the entire data set, update subhistograms for each warp
    unsigned int *const s_warp_hist = s_hist + (threadIdx.x >> LOG2_WARP_SIZE) * HISTOGRAM_BIN_COUNT * CHANNELS;
    while (index.x < width)
    {
        // take the upper 8 bits
        const unsigned char bin = ((unsigned char*)&in[index.y * width + index.x])[1];
        atomicAdd(s_warp_hist + bin + getBayerOffset(index.x, index.y) * HISTOGRAM_BIN_COUNT, 1u);
        index.x += blockDim.x * gridDim.x;
    }

    // Merge per-warp histograms into per-block and write to global memory
    cooperative_groups::sync(cta);

    if (threadIdx.y == 0)
    {
        for (int bin = threadIdx.x; bin < HISTOGRAM_BIN_COUNT * CHANNELS; bin += HISTOGRAM_THREADBLOCK_SIZE)
        {
            unsigned int sum = 0;

#pragma unroll
            for (int i = 0; i < HISTOGRAM_WARP_COUNT; ++i)
            {
                sum += s_hist[bin + i * HISTOGRAM_BIN_COUNT * CHANNELS];
            }

            atomicAdd(&histogram[bin], sum);
        }
    }
}

/**
 * Calculate the white balance gains using the per channel histograms
 *
 * @param histogram [in] pointer to histogram data (HISTOGRAM_BIN_COUNT * CHANNELS values)
 * @param gains [in] pointer to the white balance gains (CHANNELS values)
 */
__global__ void calcWBGains(const unsigned int *histogram,
                            float *gains)
{
    unsigned long long int average[CHANNELS];
    unsigned long long int max_gain = 0;
    for (int channel = 0; channel < CHANNELS; ++channel)
    {
        unsigned long long int value = 0.f;
        for (int bin = 1; bin < HISTOGRAM_BIN_COUNT; ++bin)
        {
            value += histogram[channel * HISTOGRAM_BIN_COUNT + bin] * bin;
        }
        if (channel == 1)
        {
            // there are two green channels in the image which both are counted
            // in one histogram therfore divide green channel by 2
            value /= 2;
        }
        max_gain = max(max_gain, value);
        average[channel] = max(value, 1ull);
    }

    for (int channel = 0; channel < CHANNELS; ++channel)
    {
        gains[channel] = float(max_gain) / float(average[channel]);
    }
}

/**
 * Apply white balance gains.
 *
 * @param in [in] pointer to image
 * @param width [in] width of the image
 * @param height [in] height of the image
 * @param gains [in] pointer to the white balance gains (CHANNELS values)
 */
__global__ void applyOperations(unsigned short *image,
                             int width,
                             int height,
                             const float *gains)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx_x >= width) || (idx_y >= height))
        return;

    const int index = idx_y * width + idx_x;

    float value = (float)(image[index]);

    // apply gain
    const unsigned int channel = getBayerOffset(idx_x, idx_y);
    value *= gains[channel];

    const float range = (1 << (sizeof(unsigned short) * 8)) - 1;

    // clamp
    value = max(min(value, range), 0.f);

    image[index] = (unsigned short)(value + 0.5f);
}

})";

// 3 channels (RGB)
constexpr auto CHANNELS = 3;
// histogram bin's
constexpr auto HISTOGRAM_BIN_COUNT = 256;

} // anonymous namespace

namespace hololink::operators {

void ImageProcessorOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input<holoscan::gxf::Entity>("input");
    spec.output<holoscan::gxf::Entity>("output");

    spec.param(bayer_format_, "bayer_format", "BayerFormat", "Bayer format (one of hololink::operators::ImageProcessorOp::BayerFormat)");
    spec.param(pixel_format_, "pixel_format", "PixelFormat", "Pixel format (one of hololink::operators::CsiToBayerOp::PixelFormat)");
    spec.param(optical_black_, "optical_black", "Optical Black", "optical black value", 0);
    spec.param(
        cuda_device_ordinal_, "cuda_device_ordinal", "CudaDeviceOrdinal", "Device to use for CUDA operations", 0);
    cuda_stream_handler_.define_params(spec);
}

void ImageProcessorOp::start()
{
    CudaCheck(cuInit(0));
    CUdevice device;
    CudaCheck(cuDeviceGet(&cuda_device_, cuda_device_ordinal_.get()));
    CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));
    int integrated = 0;
    CudaCheck(cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, cuda_device_));
    is_integrated_ = (integrated != 0);

    hololink::native::CudaContextScopedPush cur_cuda_context(cuda_context_);

    // histogram setup
    const auto log2_warp_size = 5;
    const auto warp_size = 1 << log2_warp_size;

    // size of histogram memory
    const auto histogram_warp_memory = HISTOGRAM_BIN_COUNT * sizeof(uint32_t) * CHANNELS;
    histogram_memory_.reset([histogram_warp_memory] {
        CUdeviceptr mem = 0;
        CudaCheck(cuMemAlloc(&mem, histogram_warp_memory));
        return mem;
    }());

    // calculate the maximum warp count supported by the available shared memory
    // size (warps == subhistograms per threadblock)
    CUdevice cuda_device = 0;
    CudaCheck(cuDeviceGet(&cuda_device, cuda_device_ordinal_));
    int shm_size = 0;
    CudaCheck(cuDeviceGetAttribute(&shm_size, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, cuda_device));

    // round down since we can't exceed the available size
    const auto histogram_warp_count = shm_size / histogram_warp_memory;
    // the shared memory per threadblock is the memory needed by one warp
    // multiplied by the warps we launch
    const auto histogram_threadblock_memory = histogram_warp_memory * histogram_warp_count;
    // threadblock size
    histogram_threadblock_size_ = histogram_warp_count * warp_size;

    uint32_t least_significant_bit;
    switch (hololink::operators::CsiToBayerOp::PixelFormat(pixel_format_.get())) {
    case hololink::operators::CsiToBayerOp::PixelFormat::RAW_8:
        least_significant_bit = 0;
        break;
    case hololink::operators::CsiToBayerOp::PixelFormat::RAW_10:
        // data is stored in the upper 10 bits of a 16 bit value
        least_significant_bit = 16 - 10;
        break;
    case hololink::operators::CsiToBayerOp::PixelFormat::RAW_12:
        // data is stored in the upper 12 bits of a 16 bit value
        least_significant_bit = 16 - 12;
        break;
    default:
        throw std::runtime_error(fmt::format("Camera pixel format {} not supported.", int(pixel_format_.get())));
    }

    uint32_t x0y0_offset, x1y0_offset, x0y1_offset, x1y1_offset;
    switch (BayerFormat(bayer_format_.get())) {
    case BayerFormat::RGGB:
        x0y0_offset = 0; // R
        x1y0_offset = 1; // G
        x0y1_offset = 1; // G
        x1y1_offset = 2; // B
        break;
    case BayerFormat::GBRG:
        x0y0_offset = 1; // G
        x1y0_offset = 2; // B
        x0y1_offset = 0; // R
        x1y1_offset = 1; // G
        break;
    default:
        throw std::runtime_error(fmt::format("Camera bayer format {} not supported.", int(bayer_format_.get())));
    }

    cuda_function_launcher_.reset(new hololink::native::CudaFunctionLauncher(
        source, { "applyBlackLevel", "histogram", "calcWBGains", "applyOperations" },
        { fmt::format("-D CHANNELS={}", CHANNELS),
            fmt::format("-D X0Y0_OFFSET={}", x0y0_offset),
            fmt::format("-D X1Y0_OFFSET={}", x1y0_offset),
            fmt::format("-D X0Y1_OFFSET={}", x0y1_offset),
            fmt::format("-D X1Y1_OFFSET={}", x1y1_offset),
            fmt::format("-D HISTOGRAM_BIN_COUNT={}", HISTOGRAM_BIN_COUNT),
            fmt::format("-D LOG2_WARP_SIZE={}", log2_warp_size),
            fmt::format("-D HISTOGRAM_WARP_COUNT={}", histogram_warp_count),
            fmt::format("-D HISTOGRAM_THREADBLOCK_SIZE={}", histogram_threadblock_size_),
            fmt::format("-D HISTOGRAM_THREADBLOCK_MEMORY={}", histogram_threadblock_memory),
            fmt::format("-D OPTICAL_BLACK={}", optical_black_.get() * (1 << least_significant_bit)) }));

    white_balance_gains_memory_.reset([] {
        CUdeviceptr mem = 0;
        CudaCheck(cuMemAlloc(&mem, CHANNELS * sizeof(float)));
        return mem;
    }());
}

void ImageProcessorOp::stop()
{
    hololink::native::CudaContextScopedPush cur_cuda_context(cuda_context_);

    cuda_function_launcher_.reset();
    histogram_memory_.reset();

    CudaCheck(cuDevicePrimaryCtxRelease(cuda_device_));
    cuda_context_ = nullptr;
}

void ImageProcessorOp::compute(holoscan::InputContext& input, holoscan::OutputContext& output, holoscan::ExecutionContext& context)
{
    auto maybe_entity = input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) {
        throw std::runtime_error("Failed to receive input");
    }

    auto& entity = static_cast<nvidia::gxf::Entity&>(maybe_entity.value());

    // get the CUDA stream from the input message
    gxf_result_t stream_handler_result = cuda_stream_handler_.from_message(context.context(), entity);
    if (stream_handler_result != GXF_SUCCESS) {
        throw std::runtime_error("Failed to get the CUDA stream from incoming messages");
    }

    const auto maybe_tensor = entity.get<nvidia::gxf::Tensor>();
    if (!maybe_tensor) {
        throw std::runtime_error("Tensor not found in message");
    }

    const auto input_tensor = maybe_tensor.value();

    if (input_tensor->storage_type() == nvidia::gxf::MemoryStorageType::kHost) {
        if (!is_integrated_ && !host_memory_warning_) {
            host_memory_warning_ = true;
            HOLOSCAN_LOG_WARN(
                "The input tensor is stored in host memory, this will reduce performance of this "
                "operator. For best performance store the input tensor in device memory.");
        }
    } else if (input_tensor->storage_type() != nvidia::gxf::MemoryStorageType::kDevice) {
        throw std::runtime_error(
            fmt::format("Unsupported storage type {}", (int)input_tensor->storage_type()));
    }

    if (input_tensor->rank() != 3) {
        throw std::runtime_error("Tensor must be an image");
    }
    if (input_tensor->element_type() != nvidia::gxf::PrimitiveType::kUnsigned16) {
        throw std::runtime_error(fmt::format("Unexpected image data type '{}', expected '{}'", int(input_tensor->element_type()), int(nvidia::gxf::PrimitiveType::kUnsigned16)));
    }

    const uint32_t height = input_tensor->shape().dimension(0);
    const uint32_t width = input_tensor->shape().dimension(1);
    const uint32_t components = input_tensor->shape().dimension(2);
    if (components != 1) {
        throw std::runtime_error(fmt::format("Unexpected component count {}, expected '1'", components));
    }

    hololink::native::CudaContextScopedPush cur_cuda_context(cuda_context_);
    const cudaStream_t cuda_stream = cuda_stream_handler_.get_cuda_stream(context.context());

    // apply optical black if set
    if (optical_black_ != 0.f) {
        cuda_function_launcher_->launch(
            "applyBlackLevel",
            { width, height, 1 },
            cuda_stream,
            input_tensor->pointer(), width, height);
    }

    // apply Grey World White Balance algorithm
    CudaCheck(cuMemsetD32Async(histogram_memory_.get(), 0, CHANNELS * HISTOGRAM_BIN_COUNT, cuda_stream));
    cuda_function_launcher_->launch(
        "histogram",
        { width, height, 1 },
        { histogram_threadblock_size_, 2, 1 },
        cuda_stream,
        input_tensor->pointer(), histogram_memory_.get(), width, height);

    // calculate white balance gains
    cuda_function_launcher_->launch(
        "calcWBGains",
        { 1, 1, 1 },
        { 1, 1, 1 },
        cuda_stream,
        histogram_memory_.get(), white_balance_gains_memory_.get());

    cuda_function_launcher_->launch(
        "applyOperations",
        { width, height, 1 },
        cuda_stream,
        input_tensor->pointer(), width, height, white_balance_gains_memory_.get());

    // pass the CUDA stream to the output message
    auto out_message = nvidia::gxf::Expected<nvidia::gxf::Entity>(entity);
    stream_handler_result
        = cuda_stream_handler_.to_message(out_message);
    if (stream_handler_result != GXF_SUCCESS) {
        throw std::runtime_error("Failed to add the CUDA stream to the outgoing messages");
    }

    // Emit the tensor
    output.emit(entity);
}

} // namespace hololink::operators
