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

#include <chrono>
#include <time.h>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/networking.hpp>
#include <hololink/operators/capture_metadata_utils.hpp>
#include <hololink/operators/image_encoding_utils.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

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

holoscan::Contract CsiToBayerOp::contract() const
{
    // Upstream stamps both ports with an identical capture_time, so OnAligned pairs them on an
    // exact match (see kMetadataPairingWindow).
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

namespace {

    // The bytes of one full frame of 16-bit samples: the output tensor pool's slot size. The compiler
    // evaluates it once, against the Params given to graph.op(); HSDK5 wants the geometry's nested
    // fields read through a projection like this one, not as parameter_byte_span() operands.
    holoscan::expected<std::uint64_t, holoscan::Error> output_frame_bytes(
        const CsiToBayerParams& params) noexcept
    {
        // Without a geometry, keep the ceiling of earlier releases so that the run compiles and
        // start() reports the missing geometry, as it always has.
        static constexpr std::uint64_t kDefaultMaxOutputBytes = 64ULL * 1024 * 1024;
        const auto& geometry = params.geometry;
        if (geometry.pixel_width == 0 || geometry.pixel_height == 0) {
            return kDefaultMaxOutputBytes;
        }
        return std::uint64_t { geometry.pixel_width } * geometry.pixel_height * sizeof(uint16_t);
    }

} // namespace

void CsiToBayerOp::setup(holoscan::OperatorSpec& spec)
{
    static constexpr DLDataType kUInt8 { static_cast<uint8_t>(kDLUInt), 8, 1 };
    static constexpr DLDataType kUInt16 { static_cast<uint8_t>(kDLUInt), 16, 1 };

    // Input and output declaration names share one namespace per operator (Holoscan rejects
    // reusing a name across inputs/outputs/readiness sources), so the input ports here use an
    // "input_" prefix while the outputs keep the bare "tensor"/"metadata" names that downstream
    // consumers (add_flow, bind_tensor_output_device) wire against.
    spec.input(input_tensor, "in_tensor")
        .expects_tensor(holoscan::TensorInputSpec {
            .representation = holoscan::TensorRepresentation {
                .memory_kind = holoscan::MemoryKind::kCudaDevice,
                .dtype = kUInt8,
            },
        });
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
            // One slot holds one full frame of 16-bit samples (a sub-frame band is smaller), so
            // the pool is sized by the configured geometry rather than by a fixed ceiling: the
            // compiler resolves this expression from the Params each graph.op() call supplies.
            holoscan::parameter_projected_byte_span<&output_frame_bytes>("com.nvidia.hololink.operators.CsiToBayerOp.output_frame_bytes", holoscan::param_ref<&CsiToBayerParams::geometry, &hololink::csi::CsiGeometry::pixel_width>(), holoscan::param_ref<&CsiToBayerParams::geometry, &hololink::csi::CsiGeometry::pixel_height>()));
    spec.output(output_metadata, "metadata").max_emits_per_compute(1U);
}

void CsiToBayerOp::start()
{
    const auto& p = params();
    cuda_device_ordinal_ = p.cuda_device_ordinal;
    sub_frame_rows_ = p.sub_frame_rows;
    bayer_format_ = p.bayer_format;
    header_included_ = p.header_included;
    drop_incomplete_ = p.drop_incomplete;

    // Apply geometry from params if provided (HSDK5 path via configure_converter(CsiGeometry&)).
    if (p.geometry.pixel_width != 0) {
        const auto& g = p.geometry;
        configure(g.start_byte, g.bytes_per_line, g.pixel_width, g.pixel_height,
            g.pixel_format, g.trailing_bytes);
    }

    if (!configured_) {
        throw std::runtime_error("CsiToBayerOp is not configured.");
    }
    // configure() may have run before the parameters were bound (legacy configure_converter());
    // size the frame from the final sub_frame_rows_ either way.
    frame_size_ = sub_frame_rows_ > 0 ? bytes_per_line_ * sub_frame_rows_ : csi_length_;

    sub_frame_bands_.reset();
    if (sub_frame_rows_ > 0) {
        if (pixel_height_ % sub_frame_rows_ != 0) {
            throw std::runtime_error(fmt::format("Height of {} is not evenly divisible by sub_frame_rows of {}",
                pixel_height_, sub_frame_rows_));
        }
        if (sub_frame_rows_ % 2 != 0) {
            // Every band is stamped with the frame's Bayer phase, which a band that starts on an
            // odd row doesn't have.
            throw std::runtime_error(fmt::format("sub_frame_rows of {} is odd; it must be even, so that "
                                                 "every band starts on the frame's Bayer phase",
                sub_frame_rows_));
        }
        sub_frame_bands_.emplace(SubFrameBands::Geometry {
            .csi_length = csi_length_,
            .band_bytes = size_t(bytes_per_line_) * sub_frame_rows_,
            .start_byte = start_byte_,
            .band_rows = sub_frame_rows_,
            .pixel_height = pixel_height_,
        });
    }

    CudaCheck(cuInit(0));
    CudaCheck(cuDeviceGet(&cuda_device_, cuda_device_ordinal_));
    CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));
    int integrated = 0;
    CudaCheck(cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, cuda_device_));
    is_integrated_ = (integrated != 0);

    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);

    cuda_function_launcher_.reset(new hololink::common::CudaFunctionLauncher(
        source, { "frameReconstruction8", "frameReconstruction10", "frameReconstruction12" }));

    if (sub_frame_rows_ > 0) {
        sub_frame_memory_size_ = bytes_per_line_ * sub_frame_rows_;
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

void CsiToBayerOp::carry_over_sub_frame_tail(const uint8_t* in_base, size_t in_total_size,
    size_t tail_size, CUstream cuda_stream)
{
    // The bytes of the chunk past the band just completed start the next band.
    if (tail_size > 0) {
        CudaCheck(cuMemcpyAsync(sub_frame_memory_.get(),
            (CUdeviceptr)(uintptr_t)(in_base + in_total_size - tail_size), tail_size, cuda_stream));
    }
}

void CsiToBayerOp::count_incomplete_frame(int64_t sub_frame_offset)
{
    if (const uint64_t count = incomplete_frames_.count()) {
        HSB_LOG_WARN("CsiToBayerOp: {} {} frame(s)/sub-frame(s) in the last second whose data was "
                     "partly lost upstream (latest sub_frame_offset={})",
            drop_incomplete_ ? "withheld" : "passed on, marked degraded,", count, sub_frame_offset);
    }
}

void CsiToBayerOp::count_oversized_chunk(size_t chunk_size)
{
    if (const uint64_t count = oversized_chunks_.count()) {
        HSB_LOG_ERROR("CsiToBayerOp: dropped {} received chunk(s) in the last second larger than a "
                      "sub-frame band ({} > {} bytes): the receiver's frame_size must be "
                      "bytes_per_line * sub_frame_rows",
            count, chunk_size, sub_frame_memory_size_);
    }
}

void CsiToBayerOp::count_backpressure_drop(int64_t sub_frame_offset)
{
    // Report once per second so sustained drops are visible.
    ++backpressure_drops_;
    const auto now = std::chrono::steady_clock::now();
    if (now - backpressure_report_time_ >= std::chrono::seconds(1)) {
        HSB_LOG_WARN("CsiToBayerOp: dropped {} frame(s)/sub-frame(s) in the last second because "
                     "downstream was behind (latest sub_frame_offset={})",
            backpressure_drops_, sub_frame_offset);
        backpressure_drops_ = 0;
        backpressure_report_time_ = now;
    }
}

holoscan::expected<void, holoscan::Error> CsiToBayerOp::compute(
    holoscan::ExecutionContext& context)
{
    // kNotReady is a normal, expected outcome from any call in compute_impl() below (a
    // clock-driven downstream consumer, e.g. HolovizOp, can trigger this operator's activation
    // -- or contend for its output pool -- before new data/capacity has actually arrived) --
    // drop this activation instead of failing the whole pipeline over it.
    auto result = compute_impl(context);
    if (!result && result.error().code == holoscan::ErrorCode::kNotReady) {
        return {};
    }
    return result;
}

holoscan::expected<void, holoscan::Error> CsiToBayerOp::compute_impl(
    holoscan::ExecutionContext& context)
{
    // Capture wall-clock immediately on operator entry — measures scheduler dispatch latency.
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    const int64_t operator_entry_ns = int64_t(ts.tv_sec) * 1'000'000'000LL + ts.tv_nsec;

    auto maybe_paired = hololink::operators::receive_paired_tensor_metadata(input_tensor, input_metadata);
    if (!maybe_paired) {
        return holoscan::make_unexpected(std::move(maybe_paired).error());
    }
    if (!*maybe_paired) {
        if (const uint64_t dropped = mismatched_frames_.count()) {
            HSB_LOG_WARN("CsiToBayerOp: dropped {} frame(s) in the last second whose payload and metadata "
                         "were stamped as different frames",
                dropped);
        }
        return {};
    }
    auto& paired = **maybe_paired;
    const holoscan::Tensor& in_tensor = paired.tensor.data;
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

    if (in_tensor.shape_span().size() != 1) {
        return holoscan::make_unexpected(holoscan::Error { holoscan::ErrorCode::kInvalidArgument,
            "Tensor must be one dimensional" });
    }

    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);
    const CUstream cuda_stream = context.cuda_stream().get();

    const uint8_t* const in_base = static_cast<const uint8_t*>(in_tensor.data());
    const size_t in_total_size = static_cast<size_t>(in_tensor.nbytes());

    const uint8_t* input_pointer = nullptr;
    uint32_t cur_pixel_height = 0;
    size_t sub_frame_copy_size = 0;
    uint64_t output_frame_id = static_cast<uint64_t>(in_frame.frame_number);
    int64_t sub_frame_row_offset = 0;
    // Rows of the output that the input actually holds, and whether part of it was lost.
    uint32_t valid_rows = 0;
    bool incomplete = false;

    if (sub_frame_rows_ > 0) {
        const SubFrameBands::Step step = sub_frame_bands_->next(SubFrameBands::Chunk {
            .frame_number = in_frame.frame_number,
            .size = in_total_size,
            .bytes_written = in_frame.bytes_written,
            .bytes_received = in_frame.frame_bytes_received,
        });
        if (step.oversized) {
            // Larger than a band: the receiver's frame_size isn't the band size. Dropped.
            count_oversized_chunk(in_total_size);
            return {};
        }
        if (step.band == SubFrameBands::Band::kHeaderOnly) {
            // Entire chunk is still within the header — nothing to copy yet.
            return {};
        }
        CudaCheck(cuMemcpyAsync(sub_frame_memory_.get() + step.band_offset,
            (CUdeviceptr)(uintptr_t)(in_base + step.payload_offset), step.copy_size, cuda_stream));
        if (step.band == SubFrameBands::Band::kPending) {
            return {};
        }
        sub_frame_copy_size = step.tail_size;
        sub_frame_row_offset = step.row_offset;

        if (step.band == SubFrameBands::Band::kTrailingOnly) {
            // A band made only of the frame's trailing bytes (the sensor's status lines after
            // the image, when they add up to a whole band): no image rows in it.
            carry_over_sub_frame_tail(in_base, in_total_size, sub_frame_copy_size, cuda_stream);
            return {};
        }
        if (step.incomplete) {
            // Part of this band was lost: its head with an earlier chunk (dropped, short or with
            // a hole), so sub_frame_memory_ holds stale bytes ahead of this chunk's, or a packet
            // of this chunk itself. Pass it on marked degraded, or with drop_incomplete withhold
            // it -- SubFrameCombinerOp then discards its frame, SubFrameVisualizerOp keeps the
            // previous frame's band -- and only carry this chunk's tail over to the next band
            // (which the hole also marks incomplete).
            count_incomplete_frame(sub_frame_row_offset);
            if (drop_incomplete_) {
                carry_over_sub_frame_tail(in_base, in_total_size, sub_frame_copy_size, cuda_stream);
                return {};
            }
            incomplete = true;
        }

        output_frame_id = step.frame_id;

        input_pointer = reinterpret_cast<const uint8_t*>((uintptr_t)(sub_frame_memory_.get()));
        cur_pixel_height = sub_frame_rows_;
        valid_rows = sub_frame_rows_;
    } else {
        // This operator is fed by two different kinds of upstream: BaseReceiverOp-derived
        // receivers (LinuxReceiverOp/RoceReceiverOp) emit the full raw buffer including the
        // start_byte_ header region, so it must be skipped here; FusaCoeCaptureOp already
        // strips it before emitting its tensor (see fusa_coe_capture.cpp's
        // `src = buffer.cuda_device_ptr + core_.start_byte()` and
        // `frame_data_size = bytes_per_line * pixel_height`, with no header region included) --
        // skipping it again here would double-skip into an already-stripped buffer, shifting
        // every row's read position and running past the true end of valid data (visible as a
        // wrapped/shifted, corrupted image). Which case this is is a fixed fact about the
        // upstream producer wired to this instance (params().header_included, set by the app at
        // compose time), not something to infer per frame from the buffer's size: a
        // BaseReceiverOp-derived receiver trims its emitted tensor to the frame's actual
        // bytes_written (see BaseReceiverOp::compute()), so an ordinary dropped-packet frame can
        // legitimately be smaller than start_byte_ + payload size while still including the
        // header, which a size-based guess would misclassify.
        input_pointer = header_included_ ? (in_base + start_byte_) : in_base;
        cur_pixel_height = pixel_height_;

        // A frame the HSB wrote only part of (the receiver trims it to bytes_written) or that
        // lost packets on the way (fewer bytes received than written) is incomplete. Only the
        // lines that arrived are converted -- never reading past the tensor -- and the rest of
        // the image is zeroed.
        const size_t header = header_included_ ? start_byte_ : 0;
        const size_t written = in_frame.bytes_written > 0
            ? std::min(in_total_size, static_cast<size_t>(in_frame.bytes_written))
            : in_total_size;
        const size_t line_bytes = transmitted_line_bytes(pixel_format_, pixel_width_);
        valid_rows = written >= header + line_bytes
            ? static_cast<uint32_t>(std::min<size_t>(pixel_height_, (written - header - line_bytes) / bytes_per_line_ + 1))
            : 0;
        const bool has_hole = in_frame.frame_bytes_received != 0
            && in_frame.frame_bytes_received < in_frame.bytes_written;
        if (valid_rows < pixel_height_ || has_hole) {
            count_incomplete_frame(0);
            if (drop_incomplete_) {
                return {};
            }
            incomplete = true;
        }
    }

    // Allocate output tensor via the HSDK5 pool declared in setup().
    static constexpr DLDataType kUInt16 { static_cast<uint8_t>(kDLUInt), 16, 1 };
    const std::array<int64_t, 3> shape_arr { int64_t(cur_pixel_height), int64_t(pixel_width_), 1 };
    auto loan = output_tensor.allocate_tensor(
        holoscan::TensorLoanRequest { .shape = shape_arr, .dtype = kUInt16 });
    if (!loan) {
        if (loan.error().code == holoscan::ErrorCode::kResourceExhausted) {
            // Every pool slot is still held downstream: drop this frame/sub-frame like a full
            // downstream queue (below), keeping the next band's carried-over bytes.
            carry_over_sub_frame_tail(in_base, in_total_size, sub_frame_copy_size, cuda_stream);
            count_backpressure_drop(sub_frame_row_offset);
            return {};
        }
        return holoscan::make_unexpected(std::move(loan).error());
    }

    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
        return holoscan::make_unexpected(std::move(writer).error());
    }
    void* const out_ptr = writer->data();

    // Convert the rows that arrived (all of them, unless the frame is incomplete).
    if (valid_rows > 0) {
        switch (pixel_format_) {
        case hololink::csi::PixelFormat::RAW_8:
            cuda_function_launcher_->launch("frameReconstruction8", { pixel_width_, valid_rows, 1 }, cuda_stream,
                out_ptr, input_pointer, bytes_per_line_, pixel_width_, valid_rows);
            break;
        case hololink::csi::PixelFormat::RAW_10:
            cuda_function_launcher_->launch("frameReconstruction10",
                { pixel_width_ / 4, valid_rows, 1 },
                cuda_stream, out_ptr, input_pointer, bytes_per_line_, pixel_width_ / 4, valid_rows);
            break;
        case hololink::csi::PixelFormat::RAW_12:
            cuda_function_launcher_->launch("frameReconstruction12",
                { pixel_width_ / 2, valid_rows, 1 },
                cuda_stream, out_ptr, input_pointer, bytes_per_line_, pixel_width_ / 2, valid_rows);
            break;
        default:
            return holoscan::make_unexpected(holoscan::Error { holoscan::ErrorCode::kInvalidArgument,
                "Unsupported bits per pixel value" });
        }
    }
    if (valid_rows < cur_pixel_height) {
        // The rows that never arrived are zeroed rather than left as stale pool memory.
        CudaCheck(cuMemsetD16Async(reinterpret_cast<CUdeviceptr>(out_ptr)
                + std::size_t(valid_rows) * pixel_width_ * sizeof(uint16_t),
            0, std::size_t(cur_pixel_height - valid_rows) * pixel_width_, cuda_stream));
    }

    // After the reconstruction above, which reads sub_frame_memory_ on the same stream.
    carry_over_sub_frame_tail(in_base, in_total_size, sub_frame_copy_size, cuda_stream);

    auto committed = std::move(*writer).commit();
    if (!committed) {
        return holoscan::make_unexpected(std::move(committed).error());
    }

    // Pass most metadata through unchanged; frame_number/sub_frame_offset/
    // sub_frame_rows/operator_entry_ns are this operator's own recomputed values,
    // not copies of the input's (see comments above: sub-frame division,
    // per-sub-frame offset, this operator's configured sub_frame_rows_, and the
    // wall-clock captured at compute() entry).
    hololink::CaptureMetadataT out_metadata = hololink::operators::forward_metadata(in_frame);
    out_metadata.frame_number = static_cast<int64_t>(output_frame_id);
    out_metadata.sub_frame_offset = sub_frame_row_offset;
    out_metadata.sub_frame_rows = sub_frame_rows_;
    out_metadata.operator_entry_ns = operator_entry_ns;

    // Both emits below share the ORIGINAL capture_time this frame arrived with (propagated,
    // not re-minted -- see propagate_emit_options()), so downstream consumers' own bounded
    // OnAligned pairing windows stay meaningful and capture_time keeps reflecting the real
    // capture instant all the way through the pipeline.
    holoscan::EmitOptions emit_opts = hololink::operators::propagate_emit_options(
        paired.tensor.metadata, output_frame_id);
    if (incomplete) {
        emit_opts.flags = emit_opts.flags | holoscan::SampleFlags::kDegraded;
    }
    holoscan::schema::ImageT out_descriptor;
    out_descriptor.height = static_cast<int32_t>(cur_pixel_height);
    out_descriptor.width = static_cast<int32_t>(pixel_width_);
    out_descriptor.encoding = hololink::operators::bayer_image_encoding(bayer_format_);
    auto emitted = hololink::operators::emit_tensor_and_metadata(
        [&] { return output_tensor.emit_tensor(std::move(*loan), out_descriptor, emit_opts); },
        output_metadata, out_metadata, emit_opts);
    if (!emitted) {
        return holoscan::make_unexpected(std::move(emitted).error());
    }
    if (*emitted == hololink::operators::PairedEmit::kDropped) {
        // A downstream queue is full: the frame is dropped rather than failing the session.
        count_backpressure_drop(out_metadata.sub_frame_offset);
    }
    return {};
}

uint32_t CsiToBayerOp::receiver_start_byte()
{
    return 0;
}

uint32_t CsiToBayerOp::received_line_bytes(uint32_t transmitted_line_bytes)
{
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

    // sub_frame_rows_ is latched in start(); before that (a camera driver's configure_converter()
    // at graph-construction time, when get_sub_frame_size() already sizes the receiver's pages)
    // read the authored parameter directly, so legacy and geometry-struct callers both see the
    // per-band size.
    const uint32_t sub_frame_rows = sub_frame_rows_ != 0 ? sub_frame_rows_ : params().sub_frame_rows;
    if (sub_frame_rows != 0) {
        frame_size_ = bytes_per_line_ * sub_frame_rows;
    } else {
        frame_size_ = csi_length_;
    }

    configured_ = true;
}

size_t CsiToBayerOp::get_csi_length() const
{
    if (!configured_) {
        throw std::runtime_error("CsiToBayerOp is not configured.");
    }

    return csi_length_;
}

size_t CsiToBayerOp::get_sub_frame_size() const
{
    if (!configured_) {
        throw std::runtime_error("CsiToBayerOp is not configured.");
    }

    return frame_size_;
}

} // namespace hololink::operators
