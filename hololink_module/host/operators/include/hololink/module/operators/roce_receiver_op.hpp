/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_OPERATORS_ROCE_RECEIVER_OP_HPP
#define HOLOLINK_MODULE_OPERATORS_ROCE_RECEIVER_OP_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>
#include <holoscan/holoscan.hpp>

#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/frame_metadata.hpp"
#include "hololink/module/roce_data_channel.hpp"
#include "hololink/module/roce_receiver.hpp"

// Forward declaration so consumers of this header don't need the full
// ReceiverMemoryDescriptor definition (which pulls in <cuda.h> +
// internal cuda_helper machinery). The full type is reached inside
// the operator's .cpp.
namespace hololink::module {
class ReceiverMemoryDescriptor;
}

namespace hololink::module::operators {

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
 * The application supplies only `enumeration_metadata`; start()
 * resolves the supplement module via Adapter::get_module(metadata) and
 * fetches the per-board RoceDataChannelInterface (keyed by
 * "serial=<n>;data_channel=<n>") and the per-module
 * FrameMetadataInterface singleton from the locator. The data channel
 * and frame-metadata service are not passed in as Args.
 *
 * Lifecycle:
 *   start()   — resolve module + channel + frame-metadata + receiver
 *               from the metadata, allocate the frame buffer, start()
 *               the receiver with the operator's runtime parameters,
 *               then channel->attach_receiver(receiver) which reads
 *               qp_number / rkey / external_frame_memory off the
 *               receiver. Spawn the blocking-monitor thread.
 *   compute() — receiver->get_next_frame, decode the EOF metadata
 *               block via the resolved frame_metadata service, copy
 *               the V1 FrameMetadata into the operator's metadata
 *               map, and emit a gxf::Entity wrapping the frame buffer
 *               as a uint8 tensor over "output".
 *   stop()    — channel->detach_receiver, receiver->close, join the
 *               monitor thread. */
class RoceReceiverOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(RoceReceiverOp);

    ~RoceReceiverOp() override;

    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(holoscan::InputContext&,
        holoscan::OutputContext& op_output,
        holoscan::ExecutionContext&) override;

private:
    // Application-supplied parameters. The data channel and
    // frame-metadata service are resolved internally from this
    // metadata; peer_ip is read from the metadata's `peer_ip`
    // field, and ibv_name / ibv_port are derived from it via
    // hololink::module::ibv_device_for_peer.
    holoscan::Parameter<hololink::module::EnumerationMetadata> metadata_;

    holoscan::Parameter<CUcontext> frame_context_;
    holoscan::Parameter<size_t> frame_size_;
    holoscan::Parameter<size_t> page_size_;
    holoscan::Parameter<uint32_t> pages_;
    holoscan::Parameter<uint32_t> queue_size_;
    holoscan::Parameter<size_t> metadata_offset_;

    // Sensor-side bring-up hooks. device_start fires after the QP is
    // authenticated and the monitor thread is running, so the sensor
    // doesn't stream into a not-yet-armed receive queue. device_stop
    // fires before the channel is unconfigured so the sensor stops
    // emitting before its destination tears down. Both default to no-op.
    holoscan::Parameter<std::function<void()>> device_start_;
    holoscan::Parameter<std::function<void()>> device_stop_;

    // Optional name for the emitted frame tensor. Stereo flows give each
    // leg a distinct name (e.g. "left" / "right") so a downstream
    // consumer can tell the two legs apart. Defaults to "" (unnamed).
    holoscan::Parameter<std::string> out_tensor_name_;

    // Optional per-operator metadata-key renaming. Stereo flows give
    // each leg a distinct prefix (e.g. "left_" / "right_") so the two
    // receivers' frame metadata don't collide when both feed a single
    // downstream consumer. Defaults to identity (keys unchanged). The
    // renamed key names are cached once in start() so compute() does no
    // per-frame string work.
    holoscan::Parameter<std::function<std::string(const std::string&)>>
        rename_metadata_;
    std::string flags_key_;
    std::string psn_key_;
    std::string crc_key_;
    std::string frame_number_key_;
    std::string timestamp_s_key_;
    std::string timestamp_ns_key_;
    std::string bytes_written_key_;
    std::string metadata_s_key_;
    std::string metadata_ns_key_;
    std::string received_s_key_;
    std::string received_ns_key_;
    std::string imm_data_key_;
    std::string page_number_key_;

    // V1 service handles resolved in start() and held until stop().
    // The forward-declared types use shared_ptr holders so the
    // implicit destructor doesn't need the full ReceiverMemoryDescriptor
    // definition at every operator-construction site (the
    // HOLOSCAN_OPERATOR_FORWARD_ARGS ctor template instantiates the
    // destructor wherever the operator is constructed, including the
    // pybind extension TU).
    std::shared_ptr<hololink::module::RoceDataChannelInterfaceV1> channel_;
    std::shared_ptr<hololink::module::FrameMetadataInterfaceV1> frame_metadata_;
    std::shared_ptr<hololink::module::ReceiverMemoryDescriptor> frame_buffer_;
    std::shared_ptr<hololink::module::RoceReceiverInterfaceV1> receiver_;
    std::shared_ptr<std::thread> monitor_thread_;

    // Frame-ready scheduling: compute() runs only when the monitor thread
    // signals a ready frame (via the receiver's set_frame_ready callback),
    // so it never blocks waiting for one — keeping the scheduler free.
    // This is required when several receivers feed a single downstream
    // join (e.g. a frame aligner): a blocking receiver would otherwise
    // starve the others under the greedy scheduler. frame_ready() is the
    // callback the monitor invokes to wake the condition; running_ guards
    // it so a frame arriving mid-teardown doesn't re-arm a retired one.
    std::shared_ptr<holoscan::AsynchronousCondition> frame_ready_condition_;
    std::atomic<bool> running_ { false };
    void frame_ready();
    // Host-side staging buffer for the EOF metadata block. Sized in
    // start() from frame_metadata_->block_size() and reused in every
    // compute() call.
    std::vector<uint8_t> host_metadata_;
    bool configured_ = false;
};

} // namespace hololink::module::operators

#endif // HOLOLINK_MODULE_OPERATORS_ROCE_RECEIVER_OP_HPP
