/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_OPERATORS_LINUX_RECEIVER_OP_HPP
#define HOLOLINK_MODULE_OPERATORS_LINUX_RECEIVER_OP_HPP

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
#include "hololink/module/linux_data_channel.hpp"
#include "hololink/module/linux_receiver.hpp"

// Forward declaration so consumers of this header don't need the full
// ReceiverMemoryDescriptor definition (which pulls in <cuda.h> +
// internal cuda_helper machinery). The full type is reached inside the
// operator's .cpp.
namespace hololink::module {
class ReceiverMemoryDescriptor;
}

namespace hololink::module::operators {

/* Software (Linux) receiver operator built against the module's V1
 * surface (LinuxDataChannelInterfaceV1 + LinuxReceiverInterfaceV1). The
 * user-space receiver thread, host frame buffer, and RoCEv2 UDP packet
 * reassembly live behind the V1 receiver (whose default impl wraps
 * hololink::operators::LinuxReceiver). Unlike RoceReceiverOp it links no
 * ibverbs and runs on hosts with no infiniband device.
 *
 * The application supplies only `enumeration_metadata`; start() resolves
 * the supplement module via Adapter::get_module(metadata) and fetches
 * the per-board LinuxDataChannelInterface + LinuxReceiverInterface
 * (keyed by "serial=<n>;data_channel=<n>") from the locator. The data
 * channel is not passed in as an Arg.
 *
 * Unlike the hardware RoCE operator, the software receiver decodes the
 * per-frame metadata as it reassembles the stream, so compute() stamps
 * those fields straight off the receiver's frame-info struct — there is
 * no device-side EOF-block copy and no FrameMetadataInterfaceV1::decode
 * step. (FrameMetadataInterfaceV1 is still resolved, but only for its
 * block_size(), which sizes the per-frame stride in the buffer.)
 *
 * Lifecycle:
 *   start()   — resolve module + channel + receiver from the metadata,
 *               allocate the frame buffer, create + bind a datagram
 *               socket via channel->configure_socket(fd), start() the
 *               receiver with the operator's runtime parameters, then
 *               channel->attach_receiver(receiver) which reads
 *               qp_number / rkey / local_port off the receiver. Spawn
 *               the worker thread (running blocking_monitor(), with the
 *               configured CPU affinity).
 *   compute() — receiver->get_next_frame on a Holoscan-allocated stream,
 *               stamp the decoded metadata into the operator's metadata
 *               map, and emit a gxf::Entity wrapping the frame buffer as
 *               a uint8 tensor over "output".
 *   stop()    — channel->detach_receiver, receiver->close, join the
 *               worker thread, close the socket. */
class LinuxReceiverOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(LinuxReceiverOp);

    ~LinuxReceiverOp() override;

    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(holoscan::InputContext&,
        holoscan::OutputContext& op_output,
        holoscan::ExecutionContext&) override;

private:
    // Application-supplied parameters. The data channel is resolved
    // internally from this metadata.
    holoscan::Parameter<hololink::module::EnumerationMetadata> metadata_;

    holoscan::Parameter<CUcontext> frame_context_;
    holoscan::Parameter<size_t> frame_size_;
    holoscan::Parameter<size_t> page_size_;
    holoscan::Parameter<uint32_t> pages_;
    holoscan::Parameter<uint32_t> queue_size_;
    // CPU affinity for the receiver worker thread. Defaults from the
    // HOLOLINK_AFFINITY environment variable (or a fixed default) in
    // start().
    holoscan::Parameter<std::vector<int>> receiver_affinity_;

    // Sensor-side bring-up hooks (same contract as RoceReceiverOp).
    holoscan::Parameter<std::function<void()>> device_start_;
    holoscan::Parameter<std::function<void()>> device_stop_;

    // Optional name for the emitted frame tensor (same contract as
    // RoceReceiverOp). Defaults to "" (unnamed).
    holoscan::Parameter<std::string> out_tensor_name_;

    // Optional per-operator metadata-key renaming (same contract as
    // RoceReceiverOp). Defaults to identity; renamed names are cached in
    // start() so compute() does no per-frame string work.
    holoscan::Parameter<std::function<std::string(const std::string&)>>
        rename_metadata_;
    std::string frame_packets_received_key_;
    std::string frame_bytes_received_key_;
    std::string frame_number_key_;
    std::string received_frame_number_key_;
    std::string received_s_key_;
    std::string received_ns_key_;
    std::string timestamp_s_key_;
    std::string timestamp_ns_key_;
    std::string metadata_s_key_;
    std::string metadata_ns_key_;
    std::string packets_dropped_key_;
    std::string crc_key_;
    std::string psn_key_;
    std::string imm_data_key_;
    std::string page_number_key_;
    std::string bytes_written_key_;

    // V1 service handles resolved in start() and held until stop().
    std::shared_ptr<hololink::module::LinuxDataChannelInterfaceV1> channel_;
    std::shared_ptr<hololink::module::LinuxReceiverInterfaceV1> receiver_;
    // Resolved for block_size() only — the software receiver decodes
    // metadata itself, so decode() is never called.
    std::shared_ptr<hololink::module::FrameMetadataInterfaceV1> frame_metadata_;
    std::shared_ptr<hololink::module::ReceiverMemoryDescriptor> frame_buffer_;
    std::shared_ptr<std::thread> monitor_thread_;

    // Frame-ready scheduling (same contract as RoceReceiverOp): compute()
    // runs only when the monitor thread signals a ready frame, so it never
    // blocks — required when several receivers feed one downstream join.
    std::shared_ptr<holoscan::AsynchronousCondition> frame_ready_condition_;
    std::atomic<bool> running_ { false };
    void frame_ready();

    // The datagram socket the receiver runs on. Owned by the operator
    // (created in start(), closed in stop()) so it outlives the
    // receiver thread.
    int data_socket_ = -1;
    bool configured_ = false;
};

} // namespace hololink::module::operators

#endif // HOLOLINK_MODULE_OPERATORS_LINUX_RECEIVER_OP_HPP
