/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "sub_frame_combiner.hpp"

#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/operators/capture_metadata_utils.hpp>
#include <hololink/operators/image_encoding_utils.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cuda.h>

namespace hololink::operators {

void SubFrameCombinerOp::setup(holoscan::OperatorSpec& spec)
{
    // Room for a 4K RGBA16 frame (3840 x 2160 x 4 x 2 bytes); the same ceiling as
    // CsiToBayerOp's and ImageProcessorOp's outputs.
    static constexpr std::size_t kMaxOutputBytes = 64ULL * 1024 * 1024;

    // Input and output declaration names share one namespace per operator (Holoscan rejects
    // reusing a name across inputs/outputs/readiness sources), so the input ports here use an
    // "in_" prefix while the outputs keep the bare "tensor"/"metadata" names that downstream
    // consumers (add_flow, bind_tensor_output_device) wire against.
    (void)spec.input(input_tensor, "in_tensor")
        .tensor_field(holoscan::schema::ImageTensorFields::data);
    spec.input(input_metadata, "in_metadata");
    // A runtime-pool output (bound to a device by the app) rather than caller-allocated storage:
    // the pool gives the outgoing channel a device, which is what provisions the consumer's CUDA
    // stream (HolovizOp and BayerDemosaicOp can consume the combined frame directly), and its
    // slots are recycled instead of allocated per frame.
    spec.output(output_tensor, "tensor")
        .max_emits_per_compute(1U)
        .tensor_field(holoscan::schema::ImageTensorFields::data)
        .produces_tensor(holoscan::TensorOutputSpec {
            .representation = holoscan::TensorRepresentation {
                .memory_kind = holoscan::MemoryKind::kCudaDevice,
            },
            .bounds = holoscan::TensorBounds {
                .max_rank = 3U,
                .max_byte_span = kMaxOutputBytes,
            },
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        });
    spec.output(output_metadata, "metadata").max_emits_per_compute(1U);
}

holoscan::Contract SubFrameCombinerOp::contract() const
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

void SubFrameCombinerOp::start()
{
    const auto& p = params();
    if (p.height == 0) {
        throw std::invalid_argument("SubFrameCombinerOp: height must be > 0");
    }
    cuda_device_ordinal_ = p.cuda_device_ordinal;
    height_ = p.height;
    drop_incomplete_ = p.drop_incomplete;
    expected_frame_number_ = 0;
    reset_pending_frame();
    dropped_frames_ = 0;

    CudaCheck(cuInit(0));
    CudaCheck(cuDeviceGet(&cuda_device_, cuda_device_ordinal_));
    CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));
}

void SubFrameCombinerOp::stop()
{
    reset_pending_frame();
    {
        hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);
        frame_memory_.reset();
        frame_memory_size_ = 0;
    }

    CudaCheck(cuDevicePrimaryCtxRelease(cuda_device_));
    cuda_context_ = nullptr;
}

void SubFrameCombinerOp::reset_pending_frame()
{
    pending_output_metadata_.reset();
    pending_image_descriptor_.reset();
    pending_sample_metadata_.reset();
    received_rows_ = 0;
    frame_has_gap_ = false;
    pending_flags_ = holoscan::SampleFlags::kNone;
}

void SubFrameCombinerOp::count_dropped_frame(const char* reason)
{
    ++dropped_frames_;
    const auto now = std::chrono::steady_clock::now();
    if (now - drop_report_time_ >= std::chrono::seconds(1)) {
        HOLOSCAN_LOG_WARN("SubFrameCombinerOp: dropped {} assembled frame(s) in the last second ({})",
            dropped_frames_, reason);
        dropped_frames_ = 0;
        drop_report_time_ = now;
    }
}

holoscan::expected<void, holoscan::Error> SubFrameCombinerOp::emit_frame(holoscan::ExecutionContext& context)
{
    // A band never arrived (its rows still hold an earlier frame's content), or one arrived
    // marked degraded (part of its data was lost upstream).
    const bool incomplete = frame_has_gap_ || received_rows_ != height_
        || (pending_flags_ & holoscan::SampleFlags::kDegraded) != holoscan::SampleFlags::kNone;
    if (incomplete && drop_incomplete_) {
        HOLOSCAN_LOG_WARN("Discarding frame {}: part of it was lost upstream",
            pending_output_metadata_->frame_number);
        reset_pending_frame();
        return {};
    }

    auto loan = output_tensor.allocate_tensor(
        holoscan::TensorLoanRequest { .shape = pending_shape_, .dtype = pending_dtype_ });
    if (!loan) {
        reset_pending_frame();
        if (loan.error().code == holoscan::ErrorCode::kResourceExhausted) {
            // Every pool slot is still held downstream (the consumer is behind): drop this
            // frame, like the receiver does, rather than failing the session.
            count_dropped_frame("output pool exhausted");
            return {};
        }
        return holoscan::make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
        reset_pending_frame();
        return holoscan::make_unexpected(std::move(writer).error());
    }
    // Ordered on the same stream as the band copies, and before the next frame's first band
    // overwrites the assembly buffer.
    CudaCheck(cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(writer->data()), frame_memory_.get(),
        static_cast<size_t>(height_) * row_bytes_, context.cuda_stream().get()));
    auto committed = std::move(*writer).commit();
    if (!committed) {
        reset_pending_frame();
        return holoscan::make_unexpected(std::move(committed).error());
    }

    // Both emits below share the ORIGINAL capture_time the first sub-frame arrived with
    // (propagated, not re-minted -- see propagate_emit_options()), so downstream consumers'
    // own bounded OnAligned pairing windows stay meaningful and capture_time keeps reflecting
    // the real capture instant all the way through the pipeline.
    holoscan::EmitOptions emit_opts = hololink::operators::propagate_emit_options(
        *pending_sample_metadata_, static_cast<uint64_t>(pending_output_metadata_->frame_number));
    emit_opts.flags = emit_opts.flags | pending_flags_;
    if (incomplete) {
        emit_opts.flags = emit_opts.flags | holoscan::SampleFlags::kDegraded;
    }
    hololink::CaptureMetadataT out_metadata = std::move(*pending_output_metadata_);
    const holoscan::schema::ImageT out_descriptor = std::move(*pending_image_descriptor_);
    reset_pending_frame();

    auto emitted = hololink::operators::emit_tensor_and_metadata(
        [&] { return output_tensor.emit_tensor(std::move(*loan), out_descriptor, emit_opts); },
        output_metadata, out_metadata, emit_opts);
    if (!emitted) {
        return holoscan::make_unexpected(std::move(emitted).error());
    }
    if (*emitted == hololink::operators::PairedEmit::kDropped) {
        // A downstream queue is full (the display is behind): the assembled frame is dropped,
        // like the receiver and the other operators do, rather than failing the session.
        count_dropped_frame("downstream queue full");
    }
    return {};
}

holoscan::expected<void, holoscan::Error> SubFrameCombinerOp::compute(holoscan::ExecutionContext& context)
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

holoscan::expected<void, holoscan::Error> SubFrameCombinerOp::compute_impl(holoscan::ExecutionContext& context)
{
    auto maybe_paired = hololink::operators::receive_paired_tensor_metadata(input_tensor, input_metadata);
    if (!maybe_paired) {
        return holoscan::make_unexpected(std::move(maybe_paired).error());
    }
    if (!*maybe_paired) {
        if (const uint64_t dropped = mismatched_frames_.count()) {
            HOLOSCAN_LOG_WARN("SubFrameCombinerOp: dropped {} frame(s) in the last second whose payload and metadata "
                              "were stamped as different frames",
                dropped);
        }
        return {};
    }
    auto& paired = **maybe_paired;
    const holoscan::schema::ImageT& in_image = paired.tensor.data;
    if (!in_image.data) {
        return holoscan::make_unexpected(holoscan::Error { holoscan::ErrorCode::kInvalidArgument,
            "Sub-frame image has no tensor" });
    }
    const holoscan::Tensor& in_tensor = *in_image.data;
    const hololink::CaptureMetadataT& in_frame = paired.metadata.data;

    const auto shape = in_tensor.shape_span();
    if (shape.size() != 3) {
        return holoscan::make_unexpected(holoscan::Error { holoscan::ErrorCode::kInvalidArgument,
            "Tensor must be rank 3" });
    }
    const int64_t band_rows = shape[0];
    const int64_t width = shape[1];
    const int64_t components = shape[2];
    const DLDataType dtype = in_tensor.dtype();
    const auto item_size = static_cast<int64_t>(in_tensor.itemsize());
    const std::vector<int64_t> strides = in_tensor.strides();
    // The pixels of a row must be packed; the rows themselves may be padded.
    if (band_rows <= 0 || width <= 0 || components <= 0 || item_size <= 0 || strides.size() != 3
        || strides[2] != item_size || strides[1] != components * item_size
        || strides[0] < width * strides[1]) {
        return holoscan::make_unexpected(holoscan::Error { holoscan::ErrorCode::kInvalidArgument,
            fmt::format("Unsupported sub-frame layout: shape [{}, {}, {}], byte strides [{}]",
                band_rows, width, components, fmt::join(strides, ", ")) });
    }
    const auto band_row_bytes = static_cast<size_t>(width * components * item_size);

    const int64_t sub_frame_offset = in_frame.sub_frame_offset;
    const int64_t frame_number = in_frame.frame_number;

    if (frame_number != expected_frame_number_) {
        if (pending_output_metadata_) {
            // The trailing sub-frame(s) of the pending frame never arrived. Emit it, incomplete
            // (as HSDK4 did) -- unless drop_incomplete is set, or this sub-frame completes its own
            // frame too: one compute() emits at most one frame.
            const bool completes_own_frame = sub_frame_offset + band_rows == static_cast<int64_t>(height_);
            HOLOSCAN_LOG_WARN(
                "Partial frame detected, sub-frame(s) dropped (expected frame {}, new frame {}); "
                "{} frame {} with {} of {} rows received",
                expected_frame_number_, frame_number,
                drop_incomplete_ || completes_own_frame ? "discarding" : "emitting", expected_frame_number_,
                received_rows_, height_);
            if (drop_incomplete_ || completes_own_frame) {
                reset_pending_frame();
            } else {
                auto emitted = emit_frame(context);
                if (!emitted) {
                    return emitted;
                }
            }
        }
        expected_frame_number_ = frame_number;
    }

    if (sub_frame_offset < 0 || sub_frame_offset >= static_cast<int64_t>(height_)) {
        return holoscan::make_unexpected(holoscan::Error { holoscan::ErrorCode::kInvalidArgument,
            fmt::format("sub_frame_offset {} is out of range [0, {})", sub_frame_offset, height_) });
    }
    if (sub_frame_offset + band_rows > static_cast<int64_t>(height_)) {
        return holoscan::make_unexpected(holoscan::Error { holoscan::ErrorCode::kInvalidArgument,
            fmt::format("sub_frame_offset {} + sub_height {} exceeds frame height {}",
                sub_frame_offset, band_rows, height_) });
    }

    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);
    const CUstream cuda_stream = context.cuda_stream().get();

    if (!pending_output_metadata_) {
        // First sub-frame of a frame: size the assembly buffer, and record what the combined
        // frame inherits from it. sub_frame_offset/sub_frame_rows reset to 0 -- once combined,
        // this is a full frame, not a sub-frame, and downstream consumers should see it that way.
        const size_t frame_bytes = static_cast<size_t>(height_) * band_row_bytes;
        if (frame_bytes > frame_memory_size_) {
            frame_memory_.reset();
            frame_memory_size_ = 0;
            frame_memory_.reset([frame_bytes] {
                CUdeviceptr ptr = 0;
                CudaCheck(cuMemAlloc(&ptr, frame_bytes));
                return ptr;
            }());
            frame_memory_size_ = frame_bytes;
        }
        pending_shape_ = { static_cast<int64_t>(height_), width, components };
        pending_dtype_ = dtype;
        row_bytes_ = band_row_bytes;
        pending_output_metadata_ = hololink::operators::forward_metadata(in_frame);
        pending_output_metadata_->sub_frame_offset = 0;
        pending_output_metadata_->sub_frame_rows = 0;
        // The combined frame's true capture instant is when its first sub-frame arrived.
        pending_sample_metadata_ = paired.tensor.metadata;
        // Every sub-frame of one frame shares the same geometry/encoding; height is the only
        // field that differs between a sub-frame and the combined frame.
        pending_image_descriptor_ = hololink::operators::forward_image_descriptor(in_image);
        pending_image_descriptor_->height = static_cast<int32_t>(height_);
        received_rows_ = 0;
        frame_has_gap_ = false;
    } else if (width != pending_shape_[1] || components != pending_shape_[2]
        || dtype.code != pending_dtype_.code || dtype.bits != pending_dtype_.bits
        || dtype.lanes != pending_dtype_.lanes) {
        return holoscan::make_unexpected(holoscan::Error { holoscan::ErrorCode::kInvalidArgument,
            fmt::format("Sub-frame geometry changed within frame {}: [{}, {}] after [{}, {}]",
                frame_number, width, components, pending_shape_[1], pending_shape_[2]) });
    }

    // A gap here means an upstream sub-frame was dropped, which is a transient under load
    // rather than a programming error: the destination is derived from sub_frame_offset, so
    // the slices that did arrive still land correctly. Resync so the remaining slices do not
    // each warn, but mark the frame: the skipped rows hold stale content, so emit_frame()
    // discards it instead of presenting it as complete.
    if (sub_frame_offset != static_cast<int64_t>(received_rows_)) {
        HOLOSCAN_LOG_WARN(
            "Non-contiguous sub-frame: got offset {}, expected row {} (sub-frame dropped; the "
            "frame is incomplete)",
            sub_frame_offset, received_rows_);
        frame_has_gap_ = true;
    }
    pending_flags_ = pending_flags_ | paired.tensor.metadata.flags;

    CUDA_MEMCPY2D copy {};
    copy.srcMemoryType = CU_MEMORYTYPE_UNIFIED;
    copy.srcDevice = reinterpret_cast<CUdeviceptr>(in_tensor.data());
    copy.srcPitch = static_cast<size_t>(strides[0]);
    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.dstDevice = frame_memory_.get() + static_cast<size_t>(sub_frame_offset) * row_bytes_;
    copy.dstPitch = row_bytes_;
    copy.WidthInBytes = row_bytes_;
    copy.Height = static_cast<size_t>(band_rows);
    CudaCheck(cuMemcpy2DAsync(&copy, cuda_stream));

    received_rows_ = static_cast<uint64_t>(sub_frame_offset + band_rows);
    if (received_rows_ == height_) {
        return emit_frame(context);
    }
    return {};
}

} // namespace hololink::operators
