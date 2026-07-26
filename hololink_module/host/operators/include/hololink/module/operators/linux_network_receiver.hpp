/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_OPERATORS_LINUX_NETWORK_RECEIVER_HPP
#define HOLOLINK_MODULE_OPERATORS_LINUX_NETWORK_RECEIVER_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>

#include "hololink/module/data_channel.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/frame_metadata.hpp"
#include "hololink/module/linux_data_channel.hpp"
#include "hololink/module/linux_receiver.hpp"
#include "hololink/module/operators/network_receiver.hpp"

namespace hololink::module {
class ReceiverMemoryDescriptor;
}

namespace hololink::module::operators {

/* Software (Linux) NetworkReceiver: the user-space RoCEv2 reassembly path of
 * LinuxReceiverOp (resolve LinuxDataChannel + LinuxReceiver, create + bind a
 * datagram socket, allocate the frame buffer, start the receiver, attach,
 * monitor thread) behind the construct/run/destruct surface. Links no ibverbs
 * and runs on hosts with no infiniband device, so it is built unconditionally.
 * construct() builds + attaches; run() spawns the monitor; destruct() tears
 * down and leaves the object reusable via a fresh construct(). Build one with
 * make_linux_network_receiver_factory().
 *
 * Unlike RoceNetworkReceiver, the software receiver copies each reassembled
 * frame host->device inside get_next_frame, on the pipeline stream the operator
 * passes in — so the copy overlaps downstream work ordered on that same stream
 * (the operator sets it on the emitted tensor). The per-frame metadata is
 * decoded during reassembly, so stamp_metadata reads it straight off the
 * receiver's frame-info struct with no device read or decode step. */
class LinuxNetworkReceiver : public NetworkReceiver {
public:
    LinuxNetworkReceiver() = default;
    ~LinuxNetworkReceiver() override;

    hololink_module_status_t construct(
        const hololink::module::EnumerationMetadata& metadata,
        const Config& config) override;
    hololink_module_status_t run() override;
    void destruct() override;

    bool get_next_frame(unsigned timeout_ms, CUstream cuda_stream) override;
    bool frames_ready() override;
    CUdeviceptr frame_memory() override;
    std::shared_ptr<void> frame_buffer_owner() override;
    void stamp_metadata(holoscan::MetadataDictionary& metadata) override;
    std::shared_ptr<hololink::module::DataChannelInterfaceV1>
    data_channel() override;

private:
    Config config_ {};

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

    std::shared_ptr<hololink::module::LinuxDataChannelInterfaceV1> channel_;
    std::shared_ptr<hololink::module::DataChannelInterfaceV1> data_channel_;
    // Resolved for block_size() only — the software receiver decodes metadata
    // itself, so FrameMetadataInterfaceV1::decode is never called.
    std::shared_ptr<hololink::module::FrameMetadataInterfaceV1> frame_metadata_;
    std::shared_ptr<hololink::module::ReceiverMemoryDescriptor> frame_buffer_;
    std::shared_ptr<hololink::module::LinuxReceiverInterfaceV1> receiver_;
    std::shared_ptr<std::thread> monitor_thread_;
    hololink::module::LinuxReceiverFrameInfoV1 frame_info_ {};

    // The datagram socket the receiver runs on. Owned here (created in
    // construct(), closed in destruct()) so it outlives the receiver thread.
    int data_socket_ = -1;

    std::atomic<bool> running_ { false };
    bool configured_ = false;
};

/* Factory the application hands to HsbControllerOp's network_receiver_factory
 * to select the software (Linux) transport. */
std::shared_ptr<NetworkReceiverFactory> make_linux_network_receiver_factory();

} // namespace hololink::module::operators

#endif // HOLOLINK_MODULE_OPERATORS_LINUX_NETWORK_RECEIVER_HPP
