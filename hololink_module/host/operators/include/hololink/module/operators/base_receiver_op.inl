/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Template method bodies for BaseReceiverOp<Derived, ParamsT>.
//
// This file is NOT included from base_receiver_op.hpp. It is included by each concrete
// derived-class .cpp file (RoceReceiverOp, LinuxReceiverOp, HsbControllerOp) so that the
// explicit template instantiation for that type can be placed in the same translation unit as
// the bodies:
//
//   #include "hololink/module/operators/roce_receiver_op.hpp"
//   #include "hololink/module/operators/base_receiver_op.inl"
//   // ...
//   template class BaseReceiverOp<RoceReceiverOp, RoceReceiverParams>;
//
// Mirrors src/hololink/operators/base_receiver_op.inl's structure and Holoscan-facing API
// usage exactly; see that file for the native reference this was modeled on.

#pragma once

#include <array>
#include <chrono>
#include <climits>
#include <exception>
#include <functional>
#include <stdexcept>
#include <thread>

#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>

#include <fmt/format.h>

#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "hololink/module/cuda_unique.hpp" // HOLOLINK_MODULE_CUDA_CHECK
#include "hololink/module/logging.hpp"
#include "hololink/module/status.h"

namespace hololink::module::operators {

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

    // Tensor output port: the raw received pixel data. produces_tensor() declares the
    // tensor memory kind and element type so HSDK5 can validate connected sink ports and
    // set up the transport channel. tensor_allocation() creates a pre-allocated pool of
    // CUDA device memory slots; compute() borrows a slot, copies the received frame into
    // it, and publishes it via emit() -- all without any heap allocation on the hot path.
    spec.output(tensor, "tensor")
        .max_emits_per_compute(1U)
        .produces_tensor(holoscan::TensorOutputSpec {
            .representation = holoscan::TensorRepresentation {
                .memory_kind = holoscan::MemoryKind::kCudaDevice,
                .dtype = kUInt8,
            },
            .bounds = holoscan::TensorBounds {
                .max_rank = 1U,
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

    // Notification source: HSDK5 replacement for HSDK4's AsynchronousCondition. Each
    // derived class's receiver invokes frame_ready() (via its set_frame_ready callback)
    // when a new frame lands, which posts to sender_ to signal the scheduler. The
    // OnNotified contract in contract() tells the scheduler to call compute() when the
    // notification fires.
    // capacity(): notifications accepted from post() through *retirement* of the
    //   activation they triggered. One is outstanding at a time (notification_outstanding_),
    //   so at most two count at once: that of the compute() running, and the one its re-arm,
    //   or a frame landing meanwhile, posts. capacity(1) would reject that second post; see
    //   the native BaseReceiverOp. The rest is headroom.
    // sender_reference_capacity(1): only one sender handle is live at a time (acquired
    // in on_arm(), invalidated in stop()).
    static constexpr uint32_t kFrameReadyNotificationCapacity = 8;
    spec.notification_source(source_, "frame-ready")
        .capacity(kFrameReadyNotificationCapacity)
        .sender_reference_capacity(1);

    // kArm lifecycle callback: runs after graph compilation but before start(). This is
    // the correct moment to acquire the generation-bound NotificationSender because the
    // run session exists but compute() has not yet been called. The static_cast changes
    // the member-function-pointer owner type from BaseReceiverOp<Derived,ParamsT> to
    // Derived so HSDK5's lifecycle system sees a match with the operator's actual
    // runtime type, avoiding GRAPH_LIFECYCLE_CALLBACK_OWNER_MISMATCH. Safe because
    // Derived inherits from BaseReceiverOp<Derived,ParamsT>.
    spec.lifecycle()
        .stage(holoscan::LifecycleStage::kArm,
            static_cast<holoscan::LifecycleStatus (Derived::*)(holoscan::LifecycleContext&) noexcept>(
                &BaseReceiverOp<Derived, ParamsT>::on_arm));
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
void BaseReceiverOp<Derived, ParamsT>::start()
{
    notification_outstanding_ = false;
    running_ = true;
    try {
        start_receiver();
        if (device_start_) {
            device_start_();
        }
    } catch (...) {
        // A start() that fails gets no stop(): tear the receiver down here, or its thread,
        // socket and queue pair outlive the run. stop_receiver() copes with a partial start.
        running_ = false;
        join_startup_post();
        try {
            stop_receiver();
        } catch (const std::exception& e) {
            HSB_LOG_ERROR("Stopping the receiver after a failed start failed: {}", e.what());
        }
        throw;
    }
    // Don't wait for the next signal if a frame is already queued and none was posted (this does
    // nothing while one is outstanding).
    if (frames_ready()) {
        frame_ready();
    }
    frame_limit_run_end_.started(this->params().frame_limit, this->params().frame_limit_ends_run);
}

template <typename Derived, typename ParamsT>
void BaseReceiverOp<Derived, ParamsT>::stop()
{
    frame_limit_run_end_.stopped();
    // Stop signaling before tearing the receiver down.
    running_ = false;
    join_startup_post();
    // The receiver is torn down even when stopping the device fails: a receiver thread that
    // is never joined aborts the process when it is destroyed. The first failure is rethrown.
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
    // Invalidate the sender so late frame_ready() calls from the (now joined) receiver
    // thread are silently rejected.
    sender_ = holoscan::NotificationSender {};
    if (failure) {
        std::rethrow_exception(failure);
    }
}

template <typename Derived, typename ParamsT>
holoscan::expected<void, holoscan::Error> BaseReceiverOp<Derived, ParamsT>::compute(
    holoscan::ExecutionContext& context)
{
    const auto& p = this->params();
    // With frame_limit_ends_run, the call that reaches frame_limit may end the whole run.
    const hololink::operators::FrameLimitRunEnd::Call frame_limit_call(frame_limit_run_end_, context);
    if (frame_limit_call.reached()) {
        HSB_LOG_INFO("Frame limit {} reached{}", p.frame_limit,
            frame_limit_call.ends_run() ? ": ending the run" : "; the run ends once the other receivers reach theirs");
    }

    // cuMemcpyDtoH (RoceReceiverOp) / cuMemcpyHtoDAsync (LinuxReceiverOp, inside its
    // receiver) are context-sensitive driver calls. Holoscan worker threads don't carry
    // frame_context, so bind it here every compute -- the monitor thread spawned by
    // start_receiver() already does the same.
    HOLOLINK_MODULE_CUDA_CHECK(cuCtxSetCurrent(p.frame_context));

    CUstream cuda_stream = reinterpret_cast<CUstream>(context.cuda_stream().get());

    // Never waits: a frame is queued for each notification (see notification_outstanding_).
    // Whether one was reported ready decides, should none come, whether to re-arm (below).
    const bool reported_ready = frames_ready();
    auto [frame_memory, frame_metadata] = get_next_frame(0.0, cuda_stream);
    // This activation's notification has served. Clearing it only now, after the take, accounts
    // for every frame: one that landed before found it outstanding and posted nothing, and the
    // re-arm below takes care of it; one that lands from here on posts its own.
    notification_outstanding_ = false;
    if (!frame_metadata) {
        if (receive_path_failed()) {
            return holoscan::make_unexpected(holoscan::Error {
                holoscan::ErrorCode::kFailure,
                "The receive path failed; no further frames will arrive" });
        }
        // No frame (HsbControllerOp connected, before its first frame): produce no output; the
        // scheduler revisits us on the next frame-ready notification, or at once for a frame
        // that landed since get_next_frame() -- but not for a receiver that reported a frame
        // ready and yielded none (a RoCE receiver whose receive path failed, behind a
        // HsbControllerOp waiting for its watchdog), which asking again at once would only
        // spin on.
        if (!reported_ready && frames_ready()) {
            frame_ready();
        }
        return {};
    }

    // Reject frame sizes that exceed the pool slot size declared in setup(). Same guard
    // module compute() already performs today, kept verbatim.
    if (p.frame_size > kMaxFrameBytes) {
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kInvalidArgument,
            fmt::format("frame_size {:#x} exceeds pool ceiling kMaxFrameBytes {:#x}",
                p.frame_size, kMaxFrameBytes) });
    }
    if (p.frame_size > static_cast<size_t>(INT_MAX)) {
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kFailure,
            "frame_size exceeds INT_MAX" });
    }
    const int64_t tensor_size = static_cast<int64_t>(p.frame_size);

    static constexpr DLDataType kUInt8 { static_cast<uint8_t>(kDLUInt), 8, 1 };
    const std::array<int64_t, 1> shape { tensor_size };

    // Borrow a pre-allocated slot from the tensor pool declared in setup(). The loan is
    // non-blocking: if all pool slots are in use by downstream consumers that have not
    // yet released them, allocate_tensor() returns kResourceExhausted immediately
    // rather than blocking.
    auto loan = tensor.allocate_tensor(
        holoscan::TensorLoanRequest { .shape = shape, .dtype = kUInt8 });
    if (!loan) {
        const auto code = loan.error().code;
        if (code == holoscan::ErrorCode::kResourceExhausted) {
            // Normal back-pressure: downstream is slower than the sensor. Drop this
            // frame and re-arm so compute() is called again as soon as the next frame
            // arrives from the receiver thread.
            count_dropped_frame("tensor pool exhausted");
            if (frames_ready()) {
                frame_ready();
            }
            return {};
        }
        HSB_LOG_ERROR("allocate_tensor failed: {}", holoscan::error_code_name(code));
        return holoscan::make_unexpected(std::move(loan).error());
    }

    // Enter the write phase on the HSDK5 compute stream. This records the stream so the
    // runtime can insert a CUDA readiness event after commit(), allowing downstream
    // consumers to pre-order their work on the same stream without any explicit
    // synchronisation calls.
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
        HSB_LOG_ERROR("loan->write() failed: {}",
            holoscan::error_code_name(writer.error().code));
        return holoscan::make_unexpected(std::move(writer).error());
    }

    // Copy the received frame from the receiver's DMA/reassembly buffer into the pool
    // slot. Device-to-Device on the HSDK5 compute stream: no CPU involvement and no
    // synchronisation barrier. This copy replaces the module shell's previous zero-copy
    // wrapMemory path -- HSDK5 provides no public API to back a runtime-pool tensor
    // with externally-owned memory directly (the same forced tradeoff the already-
    // ported native BaseReceiverOp made for the identical reason).
    if (cudaMemcpyAsync(
            writer->data(),
            reinterpret_cast<const void*>(frame_memory),
            static_cast<std::size_t>(tensor_size),
            cudaMemcpyDeviceToDevice,
            context.cuda_stream().get())
        != cudaSuccess) {
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kFailure,
            "cudaMemcpyAsync failed in BaseReceiverOp::compute" });
    }
    frame_copy_enqueued(reinterpret_cast<CUstream>(context.cuda_stream().get()));

    // Close the write phase: records a CUDA readiness event so downstream consumers can
    // enqueue device work without explicit synchronisation.
    auto committed = std::move(*writer).commit();
    if (!committed) {
        HSB_LOG_ERROR("commit() failed: {}",
            holoscan::error_code_name(committed.error().code));
        return holoscan::make_unexpected(std::move(committed).error());
    }

    // Fill the schema descriptor from the intermediate metadata map each derived class's
    // get_next_frame() populated. Fields absent for a given receiver type (e.g.
    // "frame_packets_received" is Linux-only, "dropped" RoCE-only) return 0 from meta_get and
    // are harmlessly left at zero.
    const auto meta_get = [&](const char* key) -> int64_t {
        auto v = frame_metadata->template get<int64_t>(key);
        return v ? *v : int64_t { 0 };
    };

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
    // "packets_dropped" (Linux) and "dropped" (RoCE) map to the same schema field, as in the
    // native BaseReceiverOp.
    frame_metadata_out.packets_dropped = meta_get("packets_dropped") + meta_get("dropped");
    frame_metadata_out.psn = meta_get("psn");
    frame_metadata_out.frame_packets_received = meta_get("frame_packets_received");
    frame_metadata_out.frame_bytes_received = meta_get("frame_bytes_received");
    frame_metadata_out.imm_data = meta_get("imm_data");
    frame_metadata_out.page_number = meta_get("page_number");

    // Publish the pool slot (pixels) and the metadata sample as two correlated emits,
    // both stamped with the same capture_time/frame_id so a downstream consumer can
    // pair them (e.g. native CsiToBayerOp's OnAligned contract).
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
void BaseReceiverOp<Derived, ParamsT>::post_after_start()
{
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
void BaseReceiverOp<Derived, ParamsT>::count_dropped_frame(const char* reason)
{
    if (const uint64_t dropped = dropped_frames_.count()) {
        HSB_LOG_WARN("Dropped {} frame(s) in the last second ({}): the pipeline is behind the "
                     "sensor",
            dropped, reason);
    }
}

template <typename Derived, typename ParamsT>
void BaseReceiverOp<Derived, ParamsT>::frame_ready()
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
    // Stamp the time the frame is taken, not the activation start: when compute() still waited
    // in get_next_frame() for one or more frame periods, that wait was added to each frame's age
    // (a ClockTrigger HolovizOp, max_input_age 100 ms, then rejected every frame).
    // Both emits of a frame get these same options, so OnAligned pairing does not change.
    auto now = context.clock().now();
    opts.capture_time = now ? *now : context.activation_time();
    if (auto frame_num = frame_metadata.template get<int64_t>("frame_number")) {
        opts.frame_id = static_cast<uint64_t>(*frame_num);
    }
    return opts;
}

} // namespace hololink::module::operators
