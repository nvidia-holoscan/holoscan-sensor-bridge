/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Native module image-processing operator. Self-contained: it carries its own engine (the
 * NVRTC applyBlackLevel / histogram / calcWBGains / applyOperations kernels, the histogram
 * sizing in start(), and the sub-frame-aware white-balance accumulation in compute()). The
 * pixel_format / bayer_format parameters are interpreted against the module
 * hololink::module::csi::PixelFormat / BayerFormat.
 */

#include "hololink/module/operators/image_processor_op.hpp"

#include <array>
#include <stdexcept>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <hololink/operators/capture_metadata_utils.hpp>
#include <hololink/operators/image_encoding_utils.hpp>

#include "hololink/module/csi_converter.hpp" // PixelFormat / BayerFormat
#include "hololink/module/cuda_unique.hpp" // HOLOLINK_MODULE_CUDA_CHECK
#include "hololink/module/logging.hpp" // HSB_LOG_*
#include "hololink/module/operators/cuda_function_launcher.hpp" // CudaFunctionLauncher, CudaContextScopedPush

namespace {

const char* source = R"(
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
 * @param components_per_line [in] components per input image line (width * 3 for RGB)
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

    __syncthreads();

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
    __syncthreads();

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
            // in one histogram therefore divide green channel by 2
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

namespace hololink::module::operators {

namespace {

    // The output tensor pool's slot size: one image of 16-bit samples when the application gave the
    // image's size, the ceiling of earlier releases otherwise. The compiler evaluates it once, against
    // the Params given to graph.op(). (The bound can't come from the input: a pass-through operator
    // between CsiToBayerOp and this one, ComputeCrcOp for one, has no finite span of its own.)
    holoscan::expected<std::uint64_t, holoscan::Error> output_frame_bytes(
        const ImageProcessorParams& params) noexcept
    {
        static constexpr std::uint64_t kDefaultMaxOutputBytes = 64ULL * 1024 * 1024;
        if (params.pixel_width == 0 || params.pixel_height == 0) {
            return kDefaultMaxOutputBytes;
        }
        return std::uint64_t { params.pixel_width } * params.pixel_height * sizeof(uint16_t);
    }

} // namespace

void ImageProcessorOp::setup(holoscan::OperatorSpec& spec)
{
    static constexpr DLDataType kUInt16 { static_cast<uint8_t>(kDLUInt), 16, 1 };

    (void)spec.input(input_tensor, "in_tensor")
        .tensor_field(holoscan::schema::ImageTensorFields::data);
    spec.input(input_metadata, "in_metadata");
    spec.output(output_tensor, "tensor")
        .max_emits_per_compute(1U)
        .tensor_field(holoscan::schema::ImageTensorFields::data)
        .produces_tensor(holoscan::TensorOutputSpec {
                             .representation = holoscan::TensorRepresentation {
                                 .memory_kind = holoscan::MemoryKind::kCudaDevice,
                                 .dtype = kUInt16,
                             },
                             .bounds = holoscan::TensorBounds {
                                 .max_rank = 3U,
                             },
                             .storage = holoscan::TensorOutputStorage::kRuntimePool,
                         },
            // One slot holds one image of pixel_width x pixel_height 16-bit samples (see
            // output_frame_bytes() above).
            holoscan::parameter_projected_byte_span<&output_frame_bytes>("com.nvidia.hololink.module.operators.ImageProcessorOp.output_frame_bytes", holoscan::param_ref<&ImageProcessorParams::pixel_width>(), holoscan::param_ref<&ImageProcessorParams::pixel_height>()));
    spec.output(output_metadata, "metadata").max_emits_per_compute(1U);
}

holoscan::Contract ImageProcessorOp::contract() const
{
    holoscan::Contract result;
    result
        .trigger(holoscan::OnAligned {
            .ports = holoscan::ports(input_tensor, input_metadata),
            .window = hololink::operators::kMetadataPairingWindow,
        })
        .input(input_tensor, holoscan::Exact {})
        .input(input_metadata, holoscan::Exact {});
    return result;
}

void ImageProcessorOp::start()
{
    const auto& p = params();
    pixel_format_ = p.pixel_format;
    bayer_format_ = p.bayer_format;
    optical_black_ = p.optical_black;
    cuda_device_ordinal_ = p.cuda_device_ordinal;

    HOLOLINK_MODULE_CUDA_CHECK(cuInit(0));
    HOLOLINK_MODULE_CUDA_CHECK(cuDeviceGet(&cuda_device_, cuda_device_ordinal_));
    HOLOLINK_MODULE_CUDA_CHECK(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));
    int integrated = 0;
    HOLOLINK_MODULE_CUDA_CHECK(cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, cuda_device_));
    is_integrated_ = (integrated != 0);

    CudaContextScopedPush cur_cuda_context(cuda_context_);

    // histogram setup
    const auto log2_warp_size = 5;
    const auto warp_size = 1 << log2_warp_size;

    // size of histogram memory
    constexpr auto histogram_warp_memory = HISTOGRAM_BIN_COUNT * sizeof(uint32_t) * CHANNELS;
    histogram_memory_.reset([] {
        CUdeviceptr mem = 0;
        HOLOLINK_MODULE_CUDA_CHECK(cuMemAlloc(&mem, histogram_warp_memory));
        return mem;
    }());

    // calculate the maximum warp count supported by the available shared memory
    // size (warps == subhistograms per threadblock)
    CUdevice cuda_device = 0;
    HOLOLINK_MODULE_CUDA_CHECK(cuDeviceGet(&cuda_device, cuda_device_ordinal_));
    int shm_size = 0;
    HOLOLINK_MODULE_CUDA_CHECK(cuDeviceGetAttribute(&shm_size, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, cuda_device));

    // round down since we can't exceed the available size
    const auto histogram_warp_count = shm_size / histogram_warp_memory;
    // the shared memory per threadblock is the memory needed by one warp
    // multiplied by the warps we launch
    const auto histogram_threadblock_memory = histogram_warp_memory * histogram_warp_count;
    // threadblock size
    histogram_threadblock_size_ = histogram_warp_count * warp_size;

    uint32_t least_significant_bit;
    switch (hololink::module::csi::PixelFormat(pixel_format_)) {
    case hololink::module::csi::PixelFormat::RAW_8:
        least_significant_bit = 0;
        break;
    case hololink::module::csi::PixelFormat::RAW_10:
        // data is stored in the upper 10 bits of a 16 bit value
        least_significant_bit = 16 - 10;
        break;
    case hololink::module::csi::PixelFormat::RAW_12:
        // data is stored in the upper 12 bits of a 16 bit value
        least_significant_bit = 16 - 12;
        break;
    default:
        throw std::runtime_error(fmt::format("Camera pixel format {} not supported.", pixel_format_));
    }

    uint32_t x0y0_offset, x1y0_offset, x0y1_offset, x1y1_offset;
    switch (hololink::module::csi::BayerFormat(bayer_format_)) {
    case hololink::module::csi::BayerFormat::RGGB:
        x0y0_offset = 0; // R
        x1y0_offset = 1; // G
        x0y1_offset = 1; // G
        x1y1_offset = 2; // B
        break;
    case hololink::module::csi::BayerFormat::GBRG:
        x0y0_offset = 1; // G
        x1y0_offset = 2; // B
        x0y1_offset = 0; // R
        x1y1_offset = 1; // G
        break;
    case hololink::module::csi::BayerFormat::BGGR:
        x0y0_offset = 2; // B
        x1y0_offset = 1; // G
        x0y1_offset = 1; // G
        x1y1_offset = 0; // R
        break;
    case hololink::module::csi::BayerFormat::GRBG:
        x0y0_offset = 1; // G
        x1y0_offset = 0; // R
        x0y1_offset = 2; // B
        x1y1_offset = 1; // G
        break;
    default:
        throw std::runtime_error(fmt::format("Camera bayer format {} not supported.", bayer_format_));
    }

    cuda_function_launcher_.reset(new CudaFunctionLauncher(
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
            fmt::format("-D OPTICAL_BLACK={}", optical_black_ * (1 << least_significant_bit)) }));

    white_balance_gains_memory_.reset([] {
        CUdeviceptr mem = 0;
        HOLOLINK_MODULE_CUDA_CHECK(cuMemAlloc(&mem, CHANNELS * sizeof(float)));
        return mem;
    }());

    // Initialize white balance gains to 1.0 (no correction) to avoid using uninitialized values
    float initial_gains[CHANNELS] = { 1.0f, 1.0f, 1.0f };
    HOLOLINK_MODULE_CUDA_CHECK(cuMemcpyHtoD(white_balance_gains_memory_.get(), initial_gains, CHANNELS * sizeof(float)));
    expected_frame_number_ = 0;
    accumulating_histogram_ = false;
}

void ImageProcessorOp::stop()
{
    CudaContextScopedPush cur_cuda_context(cuda_context_);

    cuda_function_launcher_.reset();
    histogram_memory_.reset();
    white_balance_gains_memory_.reset();

    HOLOLINK_MODULE_CUDA_CHECK(cuDevicePrimaryCtxRelease(cuda_device_));
    cuda_context_ = nullptr;
}

holoscan::expected<void, holoscan::Error> ImageProcessorOp::compute(holoscan::ExecutionContext& context)
{
    auto result = compute_impl(context);
    if (!result && result.error().code == holoscan::ErrorCode::kNotReady) {
        return {};
    }
    return result;
}

holoscan::expected<void, holoscan::Error> ImageProcessorOp::compute_impl(holoscan::ExecutionContext& context)
{
    auto maybe_paired = hololink::operators::receive_paired_tensor_metadata(input_tensor, input_metadata);
    if (!maybe_paired) {
        return holoscan::make_unexpected(std::move(maybe_paired).error());
    }
    if (!*maybe_paired) {
        if (const uint64_t dropped = mismatched_frames_.count()) {
            HSB_LOG_WARN("ImageProcessorOp: dropped {} frame(s) in the last second whose payload and metadata "
                         "were stamped as different frames",
                dropped);
        }
        return {};
    }
    auto& paired = **maybe_paired;
    const holoscan::schema::ImageT& in_image = paired.tensor.data;
    if (!in_image.data) {
        return holoscan::make_unexpected(holoscan::Error { holoscan::ErrorCode::kInvalidArgument,
            "received frame has null tensor" });
    }
    const holoscan::Tensor& in_tensor = *in_image.data;
    const hololink::CaptureMetadataT& in_frame = paired.metadata.data;

    const DLDeviceType device_type = in_tensor.device().device_type;
    if (device_type == kDLCPU || device_type == kDLCUDAHost) {
        if (!is_integrated_ && !host_memory_warning_) {
            host_memory_warning_ = true;
            HSB_LOG_WARN(
                "The input tensor is stored in host memory, this will reduce performance of this "
                "operator. For best performance store the input tensor in device memory.");
        }
    } else if (device_type != kDLCUDA) {
        return holoscan::make_unexpected(holoscan::Error { holoscan::ErrorCode::kInvalidArgument,
            fmt::format("Unsupported device type {}", static_cast<int>(device_type)) });
    }

    if (in_tensor.shape_span().size() != 3) {
        return holoscan::make_unexpected(holoscan::Error { holoscan::ErrorCode::kInvalidArgument,
            "Tensor must be an image (rank 3)" });
    }

    const DLDataType dtype = in_tensor.dtype();
    if (dtype.code != kDLUInt || dtype.bits != 16) {
        return holoscan::make_unexpected(holoscan::Error { holoscan::ErrorCode::kInvalidArgument,
            "Unexpected image data type, expected uint16" });
    }

    const uint32_t height = static_cast<uint32_t>(in_tensor.shape_span()[0]);
    const uint32_t width = static_cast<uint32_t>(in_tensor.shape_span()[1]);
    const uint32_t components = static_cast<uint32_t>(in_tensor.shape_span()[2]);
    if (components != 1) {
        return holoscan::make_unexpected(holoscan::Error { holoscan::ErrorCode::kInvalidArgument,
            fmt::format("Unexpected component count {}, expected '1'", components) });
    }

    CudaContextScopedPush cur_cuda_context(cuda_context_);
    const CUstream cuda_stream = context.cuda_stream().get();

    // Allocate an owned output tensor -- input pool tensors are read-only under the receive lease.
    static constexpr DLDataType kUInt16 { static_cast<uint8_t>(kDLUInt), 16, 1 };
    const std::array<int64_t, 3> out_shape { int64_t(height), int64_t(width), int64_t(components) };
    auto loan = output_tensor.allocate_tensor(holoscan::TensorLoanRequest { .shape = out_shape, .dtype = kUInt16 });
    if (!loan) {
        if (loan.error().code == holoscan::ErrorCode::kResourceExhausted) {
            // Every pool slot is still held downstream: drop this frame, like a full downstream
            // queue (below).
            count_dropped_frame(in_frame.sub_frame_offset);
            return {};
        }
        return holoscan::make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
        return holoscan::make_unexpected(std::move(writer).error());
    }

    uint16_t* const image_ptr = static_cast<uint16_t*>(writer->data());
    // Copy the input (read-leased) into the writable output slot, densely. The input's rows may
    // be padded (its byte strides say so); its pixels must be packed.
    const std::vector<int64_t> strides = in_tensor.strides();
    const auto row_bytes = static_cast<int64_t>(width) * int64_t(sizeof(uint16_t));
    if (strides.size() != 3 || strides[2] != int64_t(sizeof(uint16_t))
        || strides[1] != int64_t(sizeof(uint16_t)) || strides[0] < row_bytes) {
        return holoscan::make_unexpected(holoscan::Error { holoscan::ErrorCode::kInvalidArgument,
            fmt::format("Unsupported image layout: byte strides [{}]", fmt::join(strides, ", ")) });
    }
    CUDA_MEMCPY2D copy {};
    copy.srcMemoryType = CU_MEMORYTYPE_UNIFIED;
    copy.srcDevice = reinterpret_cast<CUdeviceptr>(in_tensor.data());
    copy.srcPitch = static_cast<size_t>(strides[0]);
    copy.dstMemoryType = CU_MEMORYTYPE_UNIFIED;
    copy.dstDevice = reinterpret_cast<CUdeviceptr>(image_ptr);
    copy.dstPitch = static_cast<size_t>(row_bytes);
    copy.WidthInBytes = static_cast<size_t>(row_bytes);
    copy.Height = height;
    if (cuMemcpy2DAsync(&copy, cuda_stream) != CUDA_SUCCESS) {
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kFailure, "cuMemcpy2DAsync failed in ImageProcessorOp::compute" });
    }

    // apply optical black if set
    if (optical_black_ != 0) {
        cuda_function_launcher_->launch(
            "applyBlackLevel",
            { width, height, 1 },
            cuda_stream,
            image_ptr, width, height);
    }

    bool clear_histogram = true;

    const bool has_sub_frame_offset = (in_frame.sub_frame_rows > 0);
    if (has_sub_frame_offset) {
        if (accumulating_histogram_ && in_frame.frame_number == expected_frame_number_) {
            // for full-frame processing we have a full histogram for each frame, but for sub-frame processing
            // we need to accumulate the histogram for all sub-frames before calculating the white balance gains
            clear_histogram = false;
        } else {
            // when we have sub-frames the white-balance gains of the previous frame will be used
            // for the current frame; the first frame after start() has no previous frame and
            // keeps the initial gains (an empty or never-cleared histogram would give zero or
            // garbage gains)
            if (accumulating_histogram_) {
                cuda_function_launcher_->launch(
                    "calcWBGains",
                    { 1, 1, 1 },
                    { 1, 1, 1 },
                    cuda_stream,
                    histogram_memory_.get(), white_balance_gains_memory_.get());
            }
            expected_frame_number_ = in_frame.frame_number;
            accumulating_histogram_ = true;
        }
    }

    if (clear_histogram) {
        HOLOLINK_MODULE_CUDA_CHECK(cuMemsetD32Async(histogram_memory_.get(), 0, CHANNELS * HISTOGRAM_BIN_COUNT, cuda_stream));
    }

    // apply Grey World White Balance algorithm
    cuda_function_launcher_->launch(
        "histogram",
        { width, height, 1 },
        { histogram_threadblock_size_, 2, 1 },
        cuda_stream,
        image_ptr, histogram_memory_.get(), width, height);

    // if we don't have sub-frames, calculate the white balance gains for the current frame and apply them
    if (!has_sub_frame_offset) {
        cuda_function_launcher_->launch(
            "calcWBGains",
            { 1, 1, 1 },
            { 1, 1, 1 },
            cuda_stream,
            histogram_memory_.get(), white_balance_gains_memory_.get());
    }

    cuda_function_launcher_->launch(
        "applyOperations",
        { width, height, 1 },
        cuda_stream,
        image_ptr, width, height, white_balance_gains_memory_.get());

    auto committed = std::move(*writer).commit();
    if (!committed) {
        return holoscan::make_unexpected(std::move(committed).error());
    }

    // Pass metadata through unchanged -- this operator's white-balance/optical-black
    // processing doesn't change frame identity or hardware metadata.
    hololink::CaptureMetadataT out_metadata = hololink::operators::forward_metadata(in_frame);
    const holoscan::EmitOptions emit_opts = hololink::operators::propagate_emit_options(
        paired.tensor.metadata, static_cast<uint64_t>(in_frame.frame_number));
    // Pass every non-tensor ImageT field through unchanged -- white-balance/optical-black
    // processing doesn't change the frame's geometry, CFA phase, or color metadata.
    holoscan::schema::ImageT out_descriptor = hololink::operators::forward_image_descriptor(in_image);
    auto emitted = hololink::operators::emit_tensor_and_metadata(
        [&] { return output_tensor.emit_tensor(std::move(*loan), out_descriptor, emit_opts); },
        output_metadata, out_metadata, emit_opts);
    if (!emitted) {
        return holoscan::make_unexpected(std::move(emitted).error());
    }
    if (*emitted == hololink::operators::PairedEmit::kDropped) {
        // A downstream queue is full: the frame is dropped rather than failing the session.
        count_dropped_frame(out_metadata.sub_frame_offset);
    }
    return {};
}

void ImageProcessorOp::count_dropped_frame(int64_t sub_frame_offset)
{
    if (const uint64_t dropped = dropped_frames_.count()) {
        HSB_LOG_WARN("ImageProcessorOp: dropped {} frame(s)/sub-frame(s) in the last second because "
                     "downstream was behind (latest sub_frame_offset={})",
            dropped, sub_frame_offset);
    }
}

} // namespace hololink::module::operators
