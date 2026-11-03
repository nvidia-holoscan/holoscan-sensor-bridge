/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hololink/module/operators/hsb_controller_op.hpp"
#include "hololink/module/operators/base_receiver_op.inl"

#include <stdexcept>
#include <utility>

#include <cuda_runtime_api.h>

#include "hololink/module/cuda_unique.hpp" // HOLOLINK_MODULE_CUDA_CHECK
#include "hololink/module/logging.hpp" // HSB_LOG_*

namespace hololink::module::operators {

void HsbControllerOp::start_receiver()
{
    const auto& p = this->params();

    HOLOLINK_MODULE_CUDA_CHECK(cuCtxSetCurrent(p.frame_context));
    if (cudaEventCreateWithFlags(&last_copy_done_, cudaEventDisableTiming) != cudaSuccess) {
        throw std::runtime_error("HsbControllerOp: cudaEventCreateWithFlags failed");
    }
    live_frame_pending_copy_ = false;
    fallback_shown_ = false;

    NetworkReceiver::Config config;
    config.frame_context = p.frame_context;
    config.frame_size = p.frame_size;
    config.page_size = p.page_size;
    config.pages = p.pages;
    config.queue_size = p.queue_size;
    config.metadata_offset = p.metadata_offset;
    // config.frame_ready is set by HsbController::start() from the wake callback below.

    controller_ = std::make_unique<HsbController>(
        sensor_factory_, network_receiver_factory_, std::move(config),
        metadata_, p.watchdog_timeout_s);
    controller_->start([this] { this->frame_ready(); });

    // Tick compute() once at startup (before any device has connected) so it emits the
    // fallback test image; the controller's own found/lost wake callback then drives it,
    // replacing the test image with live frames (and back, on disconnect). The run accepts
    // posts only once every operator has started; frame_ready() retries until it does.
    this->frame_ready();
}

void HsbControllerOp::frame_copy_enqueued(CUstream cuda_stream)
{
    if (live_frame_pending_copy_) {
        live_frame_pending_copy_ = false;
        if (cudaEventRecord(last_copy_done_, reinterpret_cast<cudaStream_t>(cuda_stream)) != cudaSuccess) {
            HSB_LOG_ERROR("HsbControllerOp: cudaEventRecord failed; waiting for the stream instead");
            cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream));
        }
    }
}

void HsbControllerOp::wait_for_last_copy()
{
    if (!last_owner_ || !last_copy_done_) {
        return;
    }
    const cudaError_t status = cudaEventSynchronize(last_copy_done_);
    if (status != cudaSuccess) {
        HSB_LOG_ERROR("HsbControllerOp: waiting for the last frame's copy failed: {}",
            cudaGetErrorString(status));
    }
}

void HsbControllerOp::stop_receiver()
{
    if (controller_) {
        controller_->stop();
        controller_.reset();
    }
    // Same race get_next_frame() guards against: the last frame's async copy (issued by
    // BaseReceiverOp::compute() after get_next_frame() returned) may still be in flight when
    // stop() runs, and dropping the last reference here can free device memory that copy is
    // still reading from.
    wait_for_last_copy();
    last_owner_.reset();
    if (last_copy_done_) {
        cudaEventDestroy(last_copy_done_);
        last_copy_done_ = nullptr;
    }
}

bool HsbControllerOp::frames_ready()
{
    // Not "|| !controller_->connected()" alone -- that would make every disconnected compute()
    // re-arm itself forever after emitting one fallback frame (see class comment) -- but only
    // until the fallback has gone out once.
    return controller_ && (controller_->frames_ready() || (!controller_->connected() && !fallback_shown_));
}

std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>> HsbControllerOp::get_next_frame(
    double timeout_ms, CUstream cuda_stream)
{
    auto metadata = std::make_shared<hololink::Metadata>();
    CUdeviceptr frame = 0;
    size_t size = 0;
    std::shared_ptr<void> owner;
    if (controller_
        && controller_->get_next_frame(static_cast<unsigned>(timeout_ms), *metadata,
            cuda_stream, frame, size, owner)) {
        // The previous frame's async copy (issued by BaseReceiverOp::compute(), which runs
        // after this hook returns) may still be in flight; a disconnect racing on the reactor
        // thread could otherwise free and reuse that memory before the copy finishes reading
        // it. A full compute() round-trip has elapsed since it was issued, so this is a
        // non-blocking check in the overwhelming common case, not a real stall.
        wait_for_last_copy();
        last_owner_ = std::move(owner);
        live_frame_pending_copy_ = true;
        fallback_shown_ = false;
        return { frame, metadata };
    }

    if (controller_ && !controller_->connected()) {
        CUdeviceptr fallback_ptr = 0;
        size_t fallback_size = 0;
        controller_->fallback_frame(fallback_ptr, fallback_size);
        if (fallback_ptr != 0) {
            // BaseReceiverOp::compute() always copies exactly params().frame_size bytes from
            // whatever pointer this hook returns -- unlike the pre-port shell, which
            // zero-copy-wrapped the fallback at its own reported size. A too-small fallback
            // buffer would read out of bounds, so refuse rather than emit a corrupt frame;
            // reconnecting_imx274_player.py's own fallback buffer is already sized to
            // frame_size exactly, so this is a defensive guard, not an expected path.
            if (fallback_size < this->params().frame_size) {
                HSB_LOG_ERROR(
                    "HsbControllerOp: fallback buffer ({} bytes) smaller than frame_size "
                    "({} bytes); refusing to emit",
                    fallback_size, this->params().frame_size);
            } else {
                // metadata left empty -> CaptureMetadataT reads all-zero for a fallback
                // frame. Application-owned, stable for the whole run (SensorFactory::
                // fallback_frame's own contract) -- no owner-lifetime handling needed here,
                // unlike the live path above.
                fallback_shown_ = true;
                return { fallback_ptr, metadata };
            }
        }
    }
    return {}; // no frame right now; the next frame_ready() notification re-triggers compute()
}

template class BaseReceiverOp<HsbControllerOp, HsbControllerParams>;

} // namespace hololink::module::operators
