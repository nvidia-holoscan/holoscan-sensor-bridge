/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_OPERATORS_LINUX_RECEIVER_OP_HPP
#define HOLOLINK_MODULE_OPERATORS_LINUX_RECEIVER_OP_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <cuda.h>

#include "hololink/module/operators/base_receiver_op.hpp"

#include "hololink/module/frame_metadata.hpp"
#include "hololink/module/linux_data_channel.hpp"
#include "hololink/module/linux_receiver.hpp"
#include <holoscan/core/parameter_schema.hpp>

// Forward declaration so consumers of this header don't need the full
// ReceiverMemoryDescriptor definition (which pulls in <cuda.h> +
// internal cuda_helper machinery). The full type is reached inside the
// operator's .cpp.
namespace hololink::module {
class ReceiverMemoryDescriptor;
}

namespace hololink::module::operators {

/**
 * @brief Flat parameter aggregate for LinuxReceiverOp.
 *
 * Set all fields before passing to .set_params(...) (or use designated initializers).
 * receiver_affinity mirrors the module shell's prior HOLOLINK_AFFINITY-env-var default:
 * an empty vector means "use the environment variable, or a fixed default" (resolved in
 * start_receiver()), matching today's behavior exactly.
 */
struct LinuxReceiverParams {
    CUcontext frame_context { nullptr };
    size_t frame_size { 0 };
    size_t page_size { 0 };
    uint32_t pages { 2 };
    uint32_t queue_size { 1 };
    std::vector<int> receiver_affinity {}; ///< Empty -> HOLOLINK_AFFINITY env var, or a fixed default.
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
    LinuxReceiverParams,
    holoscan::param_schema("com.nvidia.hololink.module.operators.LinuxReceiverParams", 1).partial(),
    holoscan::field<&LinuxReceiverParams::frame_size>("frame_size")
        .description("Size of one received frame, in bytes; the tensor pool's slot size."));

/* Software (Linux) receiver operator built against the module's V1
 * surface (LinuxDataChannelInterfaceV1 + LinuxReceiverInterfaceV1). The
 * user-space receiver thread, host frame buffer, and RoCEv2 UDP packet
 * reassembly live behind the V1 receiver (whose default impl wraps
 * hololink::operators::LinuxReceiver). Unlike RoceReceiverOp it links no
 * ibverbs and runs on hosts with no infiniband device.
 *
 * The application supplies only the enumeration metadata (constructor
 * argument); start_receiver() resolves the supplement module via
 * Adapter::get_module(metadata_) and fetches the per-board
 * LinuxDataChannelInterface + LinuxReceiverInterface (keyed by
 * "serial=<n>;data_channel=<n>") from the locator.
 *
 * Unlike the hardware RoCE operator, the software receiver decodes the
 * per-frame metadata as it reassembles the stream, so get_next_frame()
 * stamps those fields straight off the receiver's frame-info struct —
 * there is no device-side EOF-block copy and no
 * FrameMetadataInterfaceV1::decode step. (FrameMetadataInterfaceV1 is
 * still resolved, but only for its block_size(), which sizes the
 * per-frame stride in the buffer.)
 *
 * Lifecycle (see BaseReceiverOp for the shared start()/stop()/compute()
 * shape):
 *   start_receiver() — resolve module + channel + receiver from the
 *               metadata, allocate the frame buffer, create + bind a
 *               datagram socket via channel->configure_socket(fd),
 *               start() the receiver with this operator's parameters,
 *               then channel->attach_receiver(receiver) which reads
 *               qp_number / rkey / local_port off the receiver. Spawn
 *               the worker thread (running blocking_monitor(), with the
 *               configured CPU affinity).
 *   get_next_frame() — receiver->get_next_frame on the operator's
 *               compute stream, returning the frame's device pointer
 *               plus its already-decoded fields.
 *   stop_receiver() — channel->detach_receiver, receiver->close, join
 *               the worker thread, close the socket. */
class LinuxReceiverOp : public BaseReceiverOp<LinuxReceiverOp, LinuxReceiverParams> {
public:
    using BaseReceiverOp<LinuxReceiverOp, LinuxReceiverParams>::BaseReceiverOp;

    void start_receiver() override;
    void stop_receiver() override;
    std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>> get_next_frame(
        double timeout_ms, CUstream cuda_stream) override;
    bool frames_ready() override;

private:
    // V1 service handles resolved in start_receiver() and held until stop_receiver().
    std::shared_ptr<hololink::module::LinuxDataChannelInterfaceV1> channel_;
    std::shared_ptr<hololink::module::LinuxReceiverInterfaceV1> receiver_;
    // Resolved for block_size() only — the software receiver decodes
    // metadata itself, so decode() is never called.
    std::shared_ptr<hololink::module::FrameMetadataInterfaceV1> frame_metadata_;
    std::shared_ptr<hololink::module::ReceiverMemoryDescriptor> frame_buffer_;
    std::shared_ptr<std::thread> monitor_thread_;

    // The datagram socket the receiver runs on. Owned by the operator
    // (created in start_receiver(), closed in stop_receiver()) so it
    // outlives the receiver thread.
    int data_socket_ = -1;
    bool configured_ = false;
};

} // namespace hololink::module::operators

#endif // HOLOLINK_MODULE_OPERATORS_LINUX_RECEIVER_OP_HPP
