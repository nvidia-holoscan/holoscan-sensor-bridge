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

// Template method bodies for BaseReceiverOp<Derived, ParamsT>.
//
// This file is NOT included from base_receiver_op.hpp.  It is included by
// each concrete derived-class .cpp file (LinuxReceiverOp, RoceReceiverOp,
// GpuRoceTransceiverOp) so that the explicit template instantiation for that
// type can be placed in the same translation unit as the bodies:
//
//   #include "linux_receiver_op.hpp"
//   #include "../base_receiver_op.inl"
//   // ...
//   template class BaseReceiverOp<LinuxReceiverOp, LinuxReceiverParams>;
//
// This keeps base_receiver_op decoupled from its derived types — adding a new
// receiver requires no changes to base_receiver_op.*.

#pragma once

#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <netinet/in.h>
#include <pthread.h>
#include <sys/socket.h>

#include <array>
#include <climits>
#include <functional>
#include <stdexcept>

#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>

#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/logging_internal.hpp>

namespace hololink::operators {

template <typename Derived, typename ParamsT>
BaseReceiverOp<Derived, ParamsT>::~BaseReceiverOp()
{
    // stop() has normally joined these threads already; a joinable thread left behind would
    // abort the process when destroyed.
    running_ = false;
    join_startup_post();
    stop_ingress_watch();
}

template <typename Derived, typename ParamsT>
holoscan::expected<std::uint64_t, holoscan::Error> BaseReceiverOp<Derived, ParamsT>::pool_slot_bytes(
    const ParamsT& params) noexcept
{
    if (params.frame_size == 0 || params.frame_size > kMaxFrameBytes) {
        return std::uint64_t { kMaxFrameBytes };
    }
    return std::uint64_t { params.frame_size };
}

template <typename Derived, typename ParamsT>
void BaseReceiverOp<Derived, ParamsT>::setup(holoscan::OperatorSpec& spec)
{
    static constexpr DLDataType kUInt8 { static_cast<uint8_t>(kDLUInt), 8, 1 };

    // Tensor output port: the raw CSI pixel data. produces_tensor() declares the
    // tensor memory kind and element type so HSDK5 can validate connected sink
    // ports and set up the transport channel. tensor_allocation() creates a
    // pre-allocated pool of CUDA device memory slots; compute() borrows a slot via
    // allocate_tensor(), copies the RDMA frame into it, and publishes it via
    // emit() — all without any heap allocation on the hot path.
    spec.output(tensor, "tensor")
        .max_emits_per_compute(1U) // at most one frame emitted per compute()
        .produces_tensor(holoscan::TensorOutputSpec {
            .representation = holoscan::TensorRepresentation {
                .memory_kind = holoscan::MemoryKind::kCudaDevice,
                .dtype = kUInt8, // raw byte buffer
            },
            .bounds = holoscan::TensorBounds {
                .max_rank = 1U, // 1-D tensor (byte array)
            },
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        },
            // One slot holds one frame_size frame (see pool_slot_bytes()), so the pool is sized
            // by the configured frame rather than by a fixed ceiling. The projection is named
            // after the receiver's Params schema, as each receiver type has its own.
            holoscan::parameter_projected_byte_span<&BaseReceiverOp::pool_slot_bytes>(
                std::string(holoscan::param_schema<ParamsT>().header.stable_id())
                    + ".pool_slot_bytes",
                holoscan::param_ref<&ParamsT::frame_size>()));

    // Metadata output port: per-capture hardware metadata (timestamps, CRC, frame
    // counter, etc.) as its own typed FlatBuffer sample, correlated with the tensor
    // above via the shared capture_time/frame_id stamped on both in compute().
    spec.output(metadata, "metadata")
        .max_emits_per_compute(1U);

    // Notification source: HSDK5 replacement for HSDK4's AsynchronousCondition.
    // The background receiver thread calls frame_ready() → sender_.post() to
    // signal that a new frame has arrived.  The OnNotified contract in contract()
    // tells the scheduler to call compute() when the notification fires.
    // capacity(): notifications accepted from post() through *retirement* of the
    //   activation they triggered. One is outstanding at a time (notification_outstanding_),
    //   so at most two count at once: that of the compute() running, and the one its re-arm,
    //   or a frame landing meanwhile, posts. With capacity(1) that second post would be
    //   rejected, so that two pages completing back to back -- e.g. the last full page and
    //   the short trailing page of every sub-frame-sliced sensor frame -- would leave one
    //   page per frame unconsumed. The rest is headroom.
    // sender_reference_capacity(1): only one sender handle is live at a time
    //   (acquired in on_arm(), invalidated in stop()).
    static constexpr uint32_t kFrameReadyNotificationCapacity = 8;
    spec.notification_source(source_, "frame-ready")
        .capacity(kFrameReadyNotificationCapacity)
        .sender_reference_capacity(1);

    // kArm lifecycle callback: runs after graph compilation but before start().
    // This is the correct moment to acquire the generation-bound NotificationSender
    // because the run session exists but compute() has not yet been called.
    // The static_cast changes the member-function-pointer owner type from
    // BaseReceiverOp<Derived,ParamsT> to Derived so HSDK5's lifecycle system
    // sees a match with the operator's actual runtime type, avoiding
    // GRAPH_LIFECYCLE_CALLBACK_OWNER_MISMATCH.  The cast is safe because
    // Derived inherits from BaseReceiverOp<Derived,ParamsT>.
    spec.lifecycle()
        .stage(holoscan::LifecycleStage::kArm,
            static_cast<holoscan::LifecycleStatus (Derived::*)(holoscan::LifecycleContext&) noexcept>(
                &BaseReceiverOp<Derived, ParamsT>::on_arm));

    frame_count_ = 0;
}

template <typename Derived, typename ParamsT>
holoscan::Contract BaseReceiverOp<Derived, ParamsT>::contract() const
{
    holoscan::Contract result;
    result.trigger(holoscan::OnNotified { .event = source_ });
    if (const int64_t frame_limit = this->params().frame_limit; frame_limit > 0) {
        result.complete_after(holoscan::ComputeCalls { .limit = static_cast<uint64_t>(frame_limit) });
    }
    return result;
}

template <typename Derived, typename ParamsT>
void BaseReceiverOp<Derived, ParamsT>::create_data_socket()
{
    data_socket_.reset(socket(AF_INET, SOCK_DGRAM, 0));
    if (!data_socket_) {
        throw std::runtime_error("Failed to create socket");
    }

    hololink_channel_->configure_socket(data_socket_.get());
}

template <typename Derived, typename ParamsT>
void BaseReceiverOp<Derived, ParamsT>::start()
{
    create_data_socket();
    // A reused operator reports its first stall again.
    ok_ = true;
    notification_outstanding_ = false;
    running_ = true;
    // A start() that fails gets no stop(), so undo what was started before rethrowing.
    try {
        // Cleans up after itself when it throws.
        start_receiver();
    } catch (...) {
        running_ = false;
        join_startup_post();
        data_socket_.reset();
        throw;
    }
    try {
        if (device_start_) {
            device_start_();
        }
    } catch (...) {
        running_ = false;
        join_startup_post();
        try {
            stop_receiver();
        } catch (const std::exception& e) {
            HSB_LOG_ERROR("Stopping the receiver after a failed device start failed: {}", e.what());
        }
        sender_ = holoscan::NotificationSender {};
        data_socket_.reset();
        throw;
    }
    // Frames that landed while the receiver and the device started were posted before the run
    // took notifications; post_notification() retries that until the run starts. In case none
    // was posted, post here: that does nothing while one is outstanding.
    if (frames_ready()) {
        post_notification();
    }
    if (watch_ingress()) {
        start_ingress_watch();
    }
    frame_limit_run_end_.started(this->params().frame_limit, this->params().frame_limit_ends_run);
}

template <typename Derived, typename ParamsT>
void BaseReceiverOp<Derived, ParamsT>::stop()
{
    frame_limit_run_end_.stopped();
    running_ = false;
    join_startup_post();
    stop_ingress_watch();
    // The receiver is torn down even when stopping the device fails: a receiver thread
    // that is never joined aborts the process when it is destroyed. The first failure
    // is rethrown.
    std::exception_ptr failure;
    if (device_stop_) {
        try {
            device_stop_();
        } catch (...) {
            failure = std::current_exception();
        }
    }
    try {
        stop_receiver();
    } catch (...) {
        if (!failure) {
            failure = std::current_exception();
        }
    }
    // Invalidate the sender so late frame_ready() calls from the (now joined)
    // receiver thread are silently rejected.
    sender_ = holoscan::NotificationSender {};
    if (failure) {
        std::rethrow_exception(failure);
    }
}

template <typename Derived, typename ParamsT>
holoscan::expected<void, holoscan::Error> BaseReceiverOp<Derived, ParamsT>::compute(
    holoscan::ExecutionContext& context)
{
    // With frame_limit_ends_run, the call that reaches frame_limit may end the whole run.
    const FrameLimitRunEnd::Call frame_limit_call(frame_limit_run_end_, context);
    if (frame_limit_call.reached()) {
        HSB_LOG_INFO("Frame limit {} reached{}", this->params().frame_limit,
            frame_limit_call.ends_run() ? ": ending the run" : "; the run ends once the other receivers reach theirs");
    }

    // get the CUDA stream for this operator
    CUstream cuda_stream = reinterpret_cast<CUstream>(context.cuda_stream().get());

    static const bool trace = std::getenv("HOLOLINK_RX_TRACE") != nullptr;
    const auto compute_begin = std::chrono::steady_clock::now();
    if (trace && trace_last_compute_end_.time_since_epoch().count() != 0) {
        trace_max_gap_ = std::max(trace_max_gap_,
            std::chrono::duration_cast<std::chrono::microseconds>(compute_begin - trace_last_compute_end_));
    }
    struct TraceScope {
        BaseReceiverOp& op;
        std::chrono::steady_clock::time_point begin;
        ~TraceScope()
        {
            const auto end = std::chrono::steady_clock::now();
            op.trace_last_compute_end_ = end;
            if (!trace) {
                return;
            }
            op.trace_max_total_ = std::max(op.trace_max_total_,
                std::chrono::duration_cast<std::chrono::microseconds>(end - begin));
            ++op.trace_frames_;
            if (end - op.trace_report_time_ >= std::chrono::seconds(1)) {
                HSB_LOG_WARN("receiver op trace: {} computes/s, longest gap between computes {:.2f} ms, "
                             "longest get_next_frame {:.2f} ms, longest allocate_tensor {:.2f} ms, longest compute {:.2f} ms",
                    op.trace_frames_, op.trace_max_gap_.count() / 1000.0, op.trace_max_get_.count() / 1000.0,
                    op.trace_max_alloc_.count() / 1000.0, op.trace_max_total_.count() / 1000.0);
                op.trace_frames_ = 0;
                op.trace_max_gap_ = op.trace_max_get_ = op.trace_max_alloc_ = op.trace_max_total_ = std::chrono::microseconds { 0 };
                op.trace_report_time_ = end;
            }
        }
    } trace_scope { *this, compute_begin };

    // Never waits: a frame is queued for each notification, except the ingress watch's (see
    // notification_outstanding_). Whether one was reported ready decides, should none come,
    // whether to re-arm (below).
    const bool reported_ready = frames_ready();
    auto [frame_memory, frame_metadata] = get_next_frame(0.0, cuda_stream);
    // This activation's notification has served. Clearing it only now, after the take, accounts
    // for every frame: one that landed before found it outstanding and posted nothing, and the
    // re-arm below takes care of it; one that lands from here on posts its own.
    notification_outstanding_ = false;
    if (trace) {
        trace_max_get_ = std::max(trace_max_get_,
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - compute_begin));
    }
    if (!frame_metadata) {
        if (receive_path_failed()) {
            return holoscan::make_unexpected(holoscan::Error {
                holoscan::ErrorCode::kFailure,
                "The receive path failed; no further frames will arrive" });
        }
        timeout(context);
        // No frame data to emit; return without producing output, and succeed: an error
        // would fail the run.  Downstream operators will not be activated (no input), and
        // the scheduler will call compute() again on the next frame_ready() notification,
        // or the ingress watch's while no frame comes (see watch_ingress()), or at once for a
        // frame that landed since get_next_frame() -- but not for a receiver that reported a
        // frame ready and yielded none, which asking again at once would only spin on.
        if (!reported_ready && frames_ready()) {
            frame_ready();
        }
        return {};
    }

    ok_ = true;
    frame_count_ += 1;

    const auto& p = this->params();
    // Reject frame sizes that exceed the pool slot size declared in setup().
    // allocate_tensor() would fail on every frame — better to catch it here.
    if (p.frame_size > kMaxFrameBytes) {
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kInvalidArgument,
            fmt::format("frame_size {:#x} exceeds pool ceiling kMaxFrameBytes {:#x}",
                p.frame_size, kMaxFrameBytes) });
    }
    // frame_size is size_t (unsigned) but tensor shape uses int64_t (signed).
    // Guard against overflow before the cast below.
    if (p.frame_size > static_cast<size_t>(INT_MAX)) {
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kFailure,
            "frame_size exceeds INT_MAX" });
    }
    int64_t tensor_size = static_cast<int64_t>(p.frame_size);
    // Trim: shrink the output tensor to the bytes the HSB actually wrote.
    // Sensors with variable-length CSI headers may produce frames smaller than
    // frame_size; bytes_written carries the exact count for this frame.
    if (p.trim) {
        // bytes_written is absent for receivers that do not support trim
        // (e.g. GpuRoceTransceiverOp); skip silently in that case.
        if (auto bw_opt = frame_metadata->template get<int64_t>("bytes_written")) {
            int64_t bytes_written = *bw_opt;
            static constexpr unsigned kMaxLoggedErrors = 5;
            static unsigned flood_guard = 0;
            if (bytes_written < 0) {
                // Negative bytes_written is a hardware or protocol error; the value
                // would wrap to a huge size_t after the signed→unsigned cast in
                // cudaMemcpyAsync.  Ignore and keep full frame_size.
                if (flood_guard < kMaxLoggedErrors) {
                    HSB_LOG_ERROR(
                        "Unexpected negative bytes_written={}; ignoring trim.", bytes_written);
                    flood_guard++;
                }
            } else if (bytes_written > tensor_size) {
                // Sanity check: bytes_written must never exceed the allocated frame
                // buffer.  If it does, the HSB firmware has written past the end of
                // the DMA region — a serious error.  Ignore the trim value and keep
                // the full frame_size so the pool copy stays within bounds.
                if (flood_guard < kMaxLoggedErrors) {
                    HSB_LOG_ERROR(
                        "Unexpected bytes_written={:#x} is larger than frame_size={:#x}; ignoring trim.",
                        bytes_written, tensor_size);
                    flood_guard++;
                }
            } else {
                // Trim the tensor to the actual received byte count so downstream
                // operators see exactly the data written by the HSB, not the
                // zero-padded tail up to frame_size.
                tensor_size = bytes_written;
            }
        }
    }

    // Allocate a pool slot and copy the RDMA frame into it.
    static constexpr DLDataType kUInt8 { static_cast<uint8_t>(kDLUInt), 8, 1 };
    const std::array<int64_t, 1> shape { tensor_size };

    // Borrow a pre-allocated slot from the tensor pool declared in setup().
    // The loan is non-blocking: if all pool slots are in use by downstream
    // consumers that have not yet released them, allocate_tensor() returns
    // kResourceExhausted immediately rather than blocking.
    const auto alloc_begin = std::chrono::steady_clock::now();
    auto loan = tensor.allocate_tensor(
        holoscan::TensorLoanRequest { .shape = shape, .dtype = kUInt8 });
    if (trace) {
        trace_max_alloc_ = std::max(trace_max_alloc_,
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - alloc_begin));
    }
    if (!loan) { // pool allocation failed — distinguish back-pressure from fatal errors
        const auto code = loan.error().code;
        if (code == holoscan::ErrorCode::kResourceExhausted) {
            // Normal back-pressure: downstream is slower than the camera.
            // Drop this frame and re-arm so compute() is called again as
            // soon as the next frame arrives from the receiver thread.
            count_dropped_frame("tensor pool exhausted");
            if (frames_ready()) {
                frame_ready();
            }
            return {};
        }
        // Any other error is unexpected and propagated as a compute failure.
        HSB_LOG_ERROR("allocate_tensor failed: {}", holoscan::error_code_name(code));
        return holoscan::make_unexpected(std::move(loan).error());
    }

    // Enter the write phase on the HSDK5 compute stream.  This records the
    // stream so the runtime can insert a CUDA readiness event after commit(),
    // allowing downstream consumers to pre-order their work on the same stream
    // without any explicit synchronisation calls.
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
        HSB_LOG_ERROR("loan->write() failed: {}",
            holoscan::error_code_name(writer.error().code));
        return holoscan::make_unexpected(std::move(writer).error());
    }

    // Copy the RDMA frame from the receiver's DMA buffer into the pool slot.
    // Device-to-Device on the HSDK5 compute stream: no CPU involvement and no
    // synchronisation barrier.  The copy completes before any downstream kernel
    // that is enqueued on the same stream after commit() returns.
    //
    // This copy replaces HSDK4's zero-copy path (which wrapped the RDMA buffer
    // directly in a GXF tensor).  It is necessary because HSDK5 provides no
    // public API to supply a custom DeviceBufferAllocator to a tensor output pool
    // (holomq::Context::create_device_buffer_pool accepts one internally but
    // TensorOutputSpecBuilder::tensor_allocation() does not expose it).  Once
    // HSDK5 exposes that hook, the receiver can allocate pool slots from
    // RDMA-registered memory and have the NIC DMA directly into them,
    // eliminating this copy entirely.
    if (cudaMemcpyAsync(
            writer->data(), // destination: pool slot
            reinterpret_cast<const void*>(frame_memory), // source: RDMA DMA buffer
            static_cast<std::size_t>(tensor_size),
            cudaMemcpyDeviceToDevice,
            context.cuda_stream().get())
        != cudaSuccess) {
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kFailure,
            "cudaMemcpyAsync failed in BaseReceiverOp::compute" });
    }

    // Close the write phase: records a CUDA readiness event so downstream
    // consumers can enqueue device work without explicit synchronisation.
    auto committed = std::move(*writer).commit();
    if (!committed) {
        HSB_LOG_ERROR("commit() failed: {}",
            holoscan::error_code_name(committed.error().code));
        return holoscan::make_unexpected(std::move(committed).error());
    }

    // Fill the schema descriptor with all hardware metadata fields.
    // Fields absent from a given receiver (e.g. Linux-only psn on RoCE) stay 0.
    const auto meta_get = [&](const char* key) -> int64_t {
        auto v = frame_metadata->template get<int64_t>(key);
        return v ? *v : int64_t { 0 };
    };

    // Populate the FlatBuffer schema descriptor from the intermediate metadata map.
    // Fields absent for a given receiver type (e.g. "psn" is Linux-only, "dropped"
    // is RoCE-only) return 0 from meta_get and are harmlessly left at zero.
    hololink::CaptureMetadataT frame_metadata_out;
    frame_metadata_out.frame_number = meta_get("frame_number");
    frame_metadata_out.received_frame_number = meta_get("received_frame_number");
    frame_metadata_out.timestamp_s = meta_get("timestamp_s");
    frame_metadata_out.timestamp_ns = meta_get("timestamp_ns");
    frame_metadata_out.metadata_s = meta_get("metadata_s");
    frame_metadata_out.metadata_ns = meta_get("metadata_ns");
    frame_metadata_out.received_s = meta_get("received_s");
    frame_metadata_out.received_ns = meta_get("received_ns");
    frame_metadata_out.crc = meta_get("crc");
    frame_metadata_out.bytes_written = meta_get("bytes_written");
    // "packets_dropped" (Linux) and "dropped" (RoCE) map to the same schema field.
    frame_metadata_out.packets_dropped = meta_get("packets_dropped")
        + meta_get("dropped");
    frame_metadata_out.psn = meta_get("psn");
    frame_metadata_out.frame_packets_received = meta_get("frame_packets_received");
    frame_metadata_out.frame_bytes_received = meta_get("frame_bytes_received");
    frame_metadata_out.imm_data = meta_get("imm_data");
    frame_metadata_out.page_number = meta_get("page_number");

    // Publish the pool slot (pixels) and the metadata sample as two correlated
    // emits, both stamped with the same capture_time/frame_id so a downstream
    // consumer can pair them on an exact capture_time match (see
    // kMetadataPairingWindow in capture_metadata_utils.hpp) and check that they
    // describe the same frame (receive_paired_tensor_metadata()). The tensor loan
    // is consumed on the first emit() regardless of outcome; the pool slot is
    // reclaimed once all downstream consumers release the sample.
    const auto emit_opts = propagate_metadata(*frame_metadata, context);
    auto emitted = hololink::operators::emit_tensor_and_metadata(
        [&] { return tensor.emit(std::move(*loan), emit_opts); }, metadata, frame_metadata_out,
        emit_opts);
    if (!emitted) {
        HSB_LOG_ERROR("Publishing frame {} failed: {}", frame_metadata_out.frame_number,
            holoscan::runtime_error_message(emitted.error()));
        return holoscan::make_unexpected(std::move(emitted).error());
    }
    if (*emitted == hololink::operators::PairedEmit::kDropped) {
        // A downstream queue is full: the frame is dropped rather than failing the run.
        count_dropped_frame("downstream queue full");
    }

    // Re-arm if another frame is queued: it found this activation's notification outstanding
    // and posted none. Posting only now, after the emits, lets the consumers on this thread take
    // this frame before the next activation publishes another.
    if (frames_ready()) {
        frame_ready();
    }

    return {};
}

template <typename Derived, typename ParamsT>
void BaseReceiverOp<Derived, ParamsT>::count_dropped_frame(const char* reason)
{
    if (const uint64_t dropped = dropped_frames_.count()) {
        HSB_LOG_WARN("Dropped {} frame(s) in the last second ({}): the pipeline is behind the "
                     "camera",
            dropped, reason);
    }
}

template <typename Derived, typename ParamsT>
void BaseReceiverOp<Derived, ParamsT>::timeout(holoscan::ExecutionContext& /*context*/)
{
    if (ok_) {
        ok_ = false;
        HSB_LOG_ERROR("Ingress frame timeout; ignoring.");
    }
}

template <typename Derived, typename ParamsT>
void BaseReceiverOp<Derived, ParamsT>::frame_ready()
{
    frame_ready_calls_.fetch_add(1, std::memory_order_relaxed);
    post_notification();
}

template <typename Derived, typename ParamsT>
void BaseReceiverOp<Derived, ParamsT>::start_ingress_watch()
{
    // The ingress watch is a diagnostic: without its thread the run goes on, with no timeout
    // reports.
    try {
        ingress_watch_thread_ = std::thread([this] {
            pthread_setname_np(pthread_self(), "ingress_watch");
            const std::chrono::milliseconds interval { static_cast<int64_t>(kIngressTimeoutMs) };
            uint64_t calls = frame_ready_calls_.load(std::memory_order_relaxed);
            std::unique_lock<std::mutex> lock(ingress_watch_mutex_);
            while (!ingress_watch_cv_.wait_for(lock, interval, [this] { return !running_.load(); })) {
                const uint64_t latest = frame_ready_calls_.load(std::memory_order_relaxed);
                if (latest == calls) {
                    // A whole interval without a frame: let compute() find none and report it.
                    post_notification();
                }
                calls = latest;
            }
        });
    } catch (const std::exception& e) {
        HSB_LOG_WARN("Not watching for ingress frame timeouts: {}", e.what());
    }
}

template <typename Derived, typename ParamsT>
void BaseReceiverOp<Derived, ParamsT>::stop_ingress_watch()
{
    // running_ is already false; taking the mutex makes sure the thread sees that before it
    // waits again, so the notification below isn't lost.
    {
        std::lock_guard<std::mutex> lock(ingress_watch_mutex_);
    }
    ingress_watch_cv_.notify_all();
    if (ingress_watch_thread_.joinable()) {
        ingress_watch_thread_.join();
    }
}

template <typename Derived, typename ParamsT>
void BaseReceiverOp<Derived, ParamsT>::post_notification()
{
    if (!running_.load(std::memory_order_relaxed)) {
        return;
    }
    // The activation of the one outstanding takes this frame too, or re-arms for it.
    if (notification_outstanding_.exchange(true)) {
        return;
    }
    auto result = sender_.post();
    if (!result) {
        const auto code = result.error().code;
        if (code == holoscan::ErrorCode::kNotReady) {
            post_after_start();
            return;
        }
        notification_outstanding_ = false;
        if (code != holoscan::ErrorCode::kBackpressured
            && code != holoscan::ErrorCode::kClosed
            && code != holoscan::ErrorCode::kStaleEndpoint) {
            HSB_LOG_ERROR("frame_ready: sender_.post() failed: {}",
                holoscan::error_code_name(code));
        }
    }
}

template <typename Derived, typename ParamsT>
void BaseReceiverOp<Derived, ParamsT>::post_after_start()
{
    // Called by post_notification() for the outstanding notification, from start() or from the
    // receive thread; join_startup_post() holds the lock while it joins, so a thread is never
    // started after stop() has waited for it.
    std::lock_guard<std::mutex> lock(startup_post_mutex_);
    if (!running_.load()) {
        notification_outstanding_ = false;
        return;
    }
    // Only the post that made the notification outstanding gets here, so an earlier thread has
    // posted its own and returned.
    if (startup_post_thread_.joinable()) {
        startup_post_thread_.join();
    }
    startup_post_thread_ = std::thread([this] {
        while (running_.load()) {
            const auto posted = sender_.post();
            if (posted) {
                return;
            }
            if (posted.error().code != holoscan::ErrorCode::kNotReady) {
                notification_outstanding_ = false;
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
}

template <typename Derived, typename ParamsT>
void BaseReceiverOp<Derived, ParamsT>::join_startup_post()
{
    std::lock_guard<std::mutex> lock(startup_post_mutex_);
    if (startup_post_thread_.joinable()) {
        startup_post_thread_.join();
    }
}

template <typename Derived, typename ParamsT>
std::tuple<std::string, uint32_t> BaseReceiverOp<Derived, ParamsT>::local_ip_and_port()
{
    sockaddr_in ip {};
    ip.sin_family = AF_UNSPEC;
    socklen_t ip_len = sizeof(ip);
    if (getsockname(data_socket_.get(), reinterpret_cast<sockaddr*>(&ip), &ip_len) < 0) {
        throw std::runtime_error(
            fmt::format("getsockname failed with errno={}: \"{}\"", errno, strerror(errno)));
    }
    return { inet_ntoa(ip.sin_addr), ntohs(ip.sin_port) };
}

template <typename Derived, typename ParamsT>
holoscan::LifecycleStatus BaseReceiverOp<Derived, ParamsT>::on_arm(
    holoscan::LifecycleContext& context) noexcept
{
    auto sender = context.notification_sender(source_);
    if (!sender) {
        HSB_LOG_ERROR("on_arm: failed to acquire NotificationSender: {}",
            holoscan::error_code_name(sender.error().code));
        return holoscan::LifecycleStatus::kFatalFailure;
    }
    sender_ = std::move(*sender);
    return holoscan::LifecycleStatus::kOk;
}

template <typename Derived, typename ParamsT>
holoscan::EmitOptions BaseReceiverOp<Derived, ParamsT>::propagate_metadata(
    const hololink::Metadata& frame_metadata,
    holoscan::ExecutionContext& context)
{
    holoscan::EmitOptions opts;
    // Stamp the time the frame is taken, not the activation's time: when compute() still waited
    // in get_next_frame() for one or more frame periods, activation_time() added that wait to
    // each frame's age, so a consumer with a LatestWithin/ReuseLast bound (HolovizOp: 100 ms)
    // could reject every frame. Both emits of a frame get these same options, so OnAligned
    // pairing does not change.
    auto now = context.clock().now();
    opts.capture_time = now ? *now : context.activation_time();
    if (auto frame_num = frame_metadata.template get<int64_t>("frame_number")) {
        opts.frame_id = static_cast<uint64_t>(*frame_num);
    }
    return opts;
}

} // namespace hololink::operators
