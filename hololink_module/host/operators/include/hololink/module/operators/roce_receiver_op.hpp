/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_OPERATORS_ROCE_RECEIVER_OP_HPP
#define HOLOLINK_MODULE_OPERATORS_ROCE_RECEIVER_OP_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <cuda.h>

#include <hololink/core/hololink.hpp> // ExtendedCounter

#include "hololink/module/operators/base_receiver_op.hpp"

#include "hololink/module/frame_metadata.hpp"
#include "hololink/module/roce_data_channel.hpp"
#include "hololink/module/roce_receiver.hpp"
#include <holoscan/core/parameter_schema.hpp>

// Forward declaration so consumers of this header don't need the full
// ReceiverMemoryDescriptor definition (which pulls in <cuda.h> +
// internal cuda_helper machinery). The full type is reached inside
// the operator's .cpp.
namespace hololink::module {
class ReceiverMemoryDescriptor;
}

namespace hololink::module::operators {

/**
 * @brief Flat parameter aggregate for RoceReceiverOp.
 *
 * Set all fields before passing to .set_params(...) (or use designated initializers).
 * Mirrors hololink::operators::RoceReceiverParams (src/hololink/operators/roce_receiver/
 * roce_receiver_op.hpp) field-for-field so the two read almost identically at the call
 * site; metadata_offset here plays the same role native's does.
 */
struct RoceReceiverParams {
    CUcontext frame_context { nullptr };
    size_t frame_size { 0 };
    size_t page_size { 0 };
    uint32_t pages { 2 };
    uint32_t queue_size { 1 };
    size_t metadata_offset { 0 }; ///< 0 -> auto-compute from frame_size.
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
    RoceReceiverParams,
    holoscan::param_schema("com.nvidia.hololink.module.operators.RoceReceiverParams", 1).partial(),
    holoscan::field<&RoceReceiverParams::frame_size>("frame_size")
        .description("Size of one received frame, in bytes; the tensor pool's slot size."));

/* RoCE receiver operator built against the module's V1 surface
 * (RoceDataChannelInterfaceV1 + RoceReceiverInterfaceV1 +
 * FrameMetadataInterfaceV1). The ibverbs receiver thread, host frame
 * buffer, QP/MR setup, and CQ polling live behind the V1 receiver
 * (whose default impl wraps hololink::operators::RoceReceiver). The
 * per-board / per-FPGA-revision override seam is the service-locator
 * lookup on RoceReceiverInterfaceV1 — module/hsb_lite_2510/ publishes
 * a subclass under the same (serial, data_channel) instance_id the
 * channel uses, so RoceReceiverV1::get_service picks it up
 * automatically.
 *
 * The application supplies only the enumeration metadata (constructor
 * argument); start_receiver() resolves the supplement module via
 * Adapter::get_module(metadata_) and fetches the per-board
 * RoceDataChannelInterface (keyed by "serial=<n>;data_channel=<n>")
 * and the per-module FrameMetadataInterface singleton from the
 * locator.
 *
 * Lifecycle (see BaseReceiverOp for the shared start()/stop()/compute()
 * shape):
 *   start_receiver() — resolve module + channel + frame-metadata +
 *               receiver from the metadata, allocate the frame buffer,
 *               start() the receiver with this operator's parameters,
 *               then channel->attach_receiver(receiver) which reads
 *               qp_number / rkey / external_frame_memory off the
 *               receiver. Spawn the blocking-monitor thread.
 *   get_next_frame() — receiver->get_next_frame, decode the EOF
 *               metadata block via the resolved frame_metadata
 *               service, and return the frame's device pointer plus
 *               its decoded fields.
 *   stop_receiver() — channel->detach_receiver, receiver->close, join
 *               the monitor thread. */
class RoceReceiverOp : public BaseReceiverOp<RoceReceiverOp, RoceReceiverParams> {
public:
    using BaseReceiverOp<RoceReceiverOp, RoceReceiverParams>::BaseReceiverOp;

    void start_receiver() override;
    void stop_receiver() override;
    std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>> get_next_frame(
        double timeout_ms, CUstream cuda_stream) override;
    bool frames_ready() override;
    bool receive_path_failed() override;

private:
    // V1 service handles resolved in start_receiver() and held until stop_receiver().
    // The forward-declared ReceiverMemoryDescriptor type uses a shared_ptr holder so the
    // implicit destructor doesn't need its full definition at every operator-
    // construction site.
    std::shared_ptr<hololink::module::RoceDataChannelInterfaceV1> channel_;
    std::shared_ptr<hololink::module::FrameMetadataInterfaceV1> frame_metadata_;
    std::shared_ptr<hololink::module::ReceiverMemoryDescriptor> frame_buffer_;
    std::shared_ptr<hololink::module::RoceReceiverInterfaceV1> receiver_;
    // The same receiver as V2, when the module publishes it (see receive_path_failed()).
    std::shared_ptr<hololink::module::RoceReceiverInterfaceV2> receiver_v2_;
    std::shared_ptr<std::thread> monitor_thread_;

    // Host-side staging buffer for the EOF metadata block. Sized in start_receiver()
    // from frame_metadata_->block_size() and reused in every get_next_frame() call.
    std::vector<uint8_t> host_metadata_;
    bool configured_ = false;

    // Extends the EOF block's 16-bit frame_number to the 32-bit count the native RoceReceiver
    // and the module LinuxReceiverOp report, so frame_number (and the frame_id stamped from
    // it) doesn't wrap every 65536 frames. Restarted by start_receiver().
    hololink::ExtendedCounter<uint32_t, uint16_t> frame_number_;
};

} // namespace hololink::module::operators

#endif // HOLOLINK_MODULE_OPERATORS_ROCE_RECEIVER_OP_HPP
