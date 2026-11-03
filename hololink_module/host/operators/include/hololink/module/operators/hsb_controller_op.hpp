/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_OPERATORS_HSB_CONTROLLER_OP_HPP
#define HOLOLINK_MODULE_OPERATORS_HSB_CONTROLLER_OP_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "hololink/module/operators/base_receiver_op.hpp"
#include "hololink/module/operators/hsb_controller.hpp"
#include "hololink/module/operators/network_receiver.hpp"
#include "hololink/module/operators/sensor_factory.hpp"
#include <holoscan/core/parameter_schema.hpp>

namespace hololink::module::operators {

/**
 * @brief Flat parameter aggregate for HsbControllerOp.
 *
 * Union of RoceReceiverParams'/LinuxReceiverParams' fields (this operator is
 * transport-agnostic, selected at construction via network_receiver_factory) plus
 * watchdog_timeout_s, specific to HsbController's reconnect policy.
 */
struct HsbControllerParams {
    CUcontext frame_context { nullptr };
    size_t frame_size { 0 };
    size_t page_size { 0 };
    uint32_t pages { 2 };
    uint32_t queue_size { 1 };
    size_t metadata_offset { 0 };
    /// Seconds without a frame before the device is declared lost.
    double watchdog_timeout_s { 0.5 };
    /// Complete after this many frames (compute() calls), as HSDK4's CountCondition on the
    /// receiver did; 0 or less: no limit.
    int64_t frame_limit { 0 };
    /// Once this operator and every other one that sets frame_limit_ends_run have reached their
    /// frame_limit, end the whole run (ExecutionContext::request_completion()), dropping the frames
    /// still in the pipeline; see hololink::operators::FrameLimitRunGroup for when a run needs it.
    bool frame_limit_ends_run { false };
};

// A partial schema: HSDK5 only needs the field BaseReceiverOp::setup() sizes the tensor pool's
// slots from (holoscan::parameter_byte_span()); the other fields stay opaque to it and are still
// bound by the operator catalog. The schema makes frame_size a recompile-required parameter.
HOLOSCAN_PARAM_SCHEMA(
    HsbControllerParams,
    holoscan::param_schema("com.nvidia.hololink.module.operators.HsbControllerParams", 1).partial(),
    holoscan::field<&HsbControllerParams::frame_size>("frame_size")
        .description("Size of one received frame, in bytes; the tensor pool's slot size."));

/* Reconnection controller operator, ported to HSDK5's Operator<Params> shape via the
 * same BaseReceiverOp<Derived,ParamsT> CRTP base RoceReceiverOp/LinuxReceiverOp are built
 * on -- setup()/contract()/start()/stop()/compute() (including the "tensor"/"metadata"
 * output ports) are entirely inherited; this class implements only the 4 CRTP hooks.
 *
 * Orchestration is otherwise unchanged from the pre-port shell: HsbController (owning the
 * SensorFactory/NetworkReceiver lifecycle, the reconnect watchdog, and the found/lost
 * callbacks) is used exactly as before, via the same sensor_factory/network_receiver_factory
 * objects the application supplies at construction.
 *
 * Two behavioral adaptations, forced by the typed-port/runtime-pool model (not present in
 * the pre-port shell, which zero-copy-wrapped the receiver's own buffer directly):
 *  - frames_ready() reports "a live frame is already queued" (HsbController::frames_ready()),
 *    or a disconnect whose fallback image hasn't been emitted yet -- never merely "a fallback
 *    is available": BaseReceiverOp::compute() re-arms itself whenever frames_ready() is true
 *    after a successful emit, so treating "disconnected" as ready would busy-loop re-emitting
 *    the same fallback frame forever.
 *  - get_next_frame()'s live-frame path tracks the previous frame's buffer owner and waits for
 *    that frame's copy (an event recorded right after BaseReceiverOp::compute() enqueues it)
 *    before releasing it. HsbController is the one receiver in this codebase whose buffer is
 *    torn down and freshly reallocated *while the operator keeps running* (a disconnect's
 *    lost() runs on the reactor thread, independent of compute() timing) -- without this, a
 *    disconnect racing a still-in-flight cudaMemcpyAsync could read memory already freed and
 *    reused by a fresh connection. */
class HsbControllerOp : public BaseReceiverOp<HsbControllerOp, HsbControllerParams> {
public:
    HsbControllerOp(hololink::module::EnumerationMetadata metadata,
        std::shared_ptr<SensorFactory> sensor_factory,
        std::shared_ptr<NetworkReceiverFactory> network_receiver_factory)
        : BaseReceiverOp<HsbControllerOp, HsbControllerParams>(std::move(metadata))
        , sensor_factory_(std::move(sensor_factory))
        , network_receiver_factory_(std::move(network_receiver_factory))
    {
    }

    void start_receiver() override;
    void stop_receiver() override;
    std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>> get_next_frame(
        double timeout_ms, CUstream cuda_stream) override;
    bool frames_ready() override;
    void frame_copy_enqueued(CUstream cuda_stream) override;

private:
    // Waits until the last live frame's copy has finished reading its buffer.
    void wait_for_last_copy();

    std::shared_ptr<SensorFactory> sensor_factory_;
    std::shared_ptr<NetworkReceiverFactory> network_receiver_factory_;
    std::unique_ptr<HsbController> controller_;

    // Tracks the live-frame buffer owner across calls so a disconnect can't free memory a
    // still-in-flight copy from the *previous* frame might still be reading -- see the
    // class-level comment's second adaptation.
    std::shared_ptr<void> last_owner_;
    // Recorded on the activation's stream right after a live frame's copy is enqueued (the
    // activation's stream itself may be gone by the next activation).
    cudaEvent_t last_copy_done_ { nullptr };
    bool live_frame_pending_copy_ = false;
    // Whether the fallback image went out since the last live frame (or since start): until it
    // has, frames_ready() reports a disconnect, so that a lost() wake BaseReceiverOp coalesced
    // with the notification of a compute() still running gets the fallback shown by that
    // compute()'s re-arm.
    std::atomic<bool> fallback_shown_ { false };
};

} // namespace hololink::module::operators

#endif // HOLOLINK_MODULE_OPERATORS_HSB_CONTROLLER_OP_HPP
