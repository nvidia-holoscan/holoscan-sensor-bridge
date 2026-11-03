/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_OPERATORS_ROCE_NETWORK_RECEIVER_HPP
#define HOLOLINK_MODULE_OPERATORS_ROCE_NETWORK_RECEIVER_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>

#include <hololink/core/hololink.hpp> // ExtendedCounter
#include <hololink/core/metadata.hpp>

#include "hololink/module/data_channel.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/frame_metadata.hpp"
#include "hololink/module/operators/network_receiver.hpp"
#include "hololink/module/roce_data_channel.hpp"
#include "hololink/module/roce_receiver.hpp"

namespace hololink::module {
class ReceiverMemoryDescriptor;
}

namespace hololink::module::operators {

/* RoCE NetworkReceiver: the ibverbs data-plane machinery of RoceReceiverOp
 * (resolve RoceDataChannel + FrameMetadata + RoceReceiver, allocate the
 * frame buffer, start the receiver, attach, monitor thread) behind the
 * construct/run/destruct surface. construct() builds + attaches; run()
 * spawns the monitor; destruct() tears down and leaves the object reusable
 * via a fresh construct(). Build one with make_roce_network_receiver_factory(). */
class RoceNetworkReceiver : public NetworkReceiver {
public:
    RoceNetworkReceiver() = default;
    ~RoceNetworkReceiver() override;

    hololink_module_status_t construct(
        const hololink::module::EnumerationMetadata& metadata,
        const Config& config) override;
    hololink_module_status_t run() override;
    void destruct() override;

    bool get_next_frame(unsigned timeout_ms, CUstream cuda_stream) override;
    bool frames_ready() override;
    bool receive_path_failed() override;
    CUdeviceptr frame_memory() override;
    std::shared_ptr<void> frame_buffer_owner() override;
    void stamp_metadata(hololink::Metadata& metadata) override;
    std::shared_ptr<hololink::module::DataChannelInterfaceV1>
    data_channel() override;

private:
    Config config_ {};

    std::shared_ptr<hololink::module::RoceDataChannelInterfaceV1> channel_;
    std::shared_ptr<hololink::module::DataChannelInterfaceV1> data_channel_;
    std::shared_ptr<hololink::module::FrameMetadataInterfaceV1> frame_metadata_;
    std::shared_ptr<hololink::module::ReceiverMemoryDescriptor> frame_buffer_;
    std::shared_ptr<hololink::module::RoceReceiverInterfaceV1> receiver_;
    // The same receiver when its module publishes RoceReceiverInterfaceV2, which also reports a
    // failed receive path; null otherwise.
    std::shared_ptr<hololink::module::RoceReceiverInterfaceV2> receiver_v2_;
    std::shared_ptr<std::thread> monitor_thread_;
    std::vector<uint8_t> host_metadata_;
    hololink::module::RoceReceiverFrameInfoV1 frame_info_ {};
    // Extends the EOF block's 16-bit frame_number to 32 bits, as RoceReceiverOp does.
    // Restarted by construct(), so each connection counts from its first frame.
    hololink::ExtendedCounter<uint32_t, uint16_t> frame_number_;

    std::atomic<bool> running_ { false };
    bool configured_ = false;
};

/* Factory the application hands to HsbControllerOp's network_receiver_factory
 * to select the RoCE transport. */
std::shared_ptr<NetworkReceiverFactory> make_roce_network_receiver_factory();

} // namespace hololink::module::operators

#endif // HOLOLINK_MODULE_OPERATORS_ROCE_NETWORK_RECEIVER_HPP
