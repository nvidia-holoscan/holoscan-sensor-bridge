/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hololink/module/operators/roce_network_receiver.hpp"

#include <unistd.h> // getpagesize

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include <fmt/format.h>

#include "hololink/module/adapter.hpp"
#include "hololink/module/cuda_unique.hpp" // HOLOLINK_MODULE_CUDA_CHECK
#include "hololink/module/ibv_device.hpp" // ibv_device_for_peer
#include "hololink/module/logging.hpp"
#include "hololink/module/module.hpp"
#include "hololink/module/page_size.hpp" // PAGE_SIZE, round_up
#include "hololink/module/receiver_memory_descriptor.hpp"
#include "hololink/module/status.h"

namespace hololink::module::operators {

RoceNetworkReceiver::~RoceNetworkReceiver()
{
    destruct();
}

hololink_module_status_t RoceNetworkReceiver::construct(
    const hololink::module::EnumerationMetadata& metadata, const Config& config)
{
    config_ = config;

    auto rename = config_.rename_metadata;
    if (!rename) {
        rename = [](const std::string& name) { return name; };
    }
    flags_key_ = rename("flags");
    psn_key_ = rename("psn");
    crc_key_ = rename("crc");
    frame_number_key_ = rename("frame_number");
    timestamp_s_key_ = rename("timestamp_s");
    timestamp_ns_key_ = rename("timestamp_ns");
    bytes_written_key_ = rename("bytes_written");
    metadata_s_key_ = rename("metadata_s");
    metadata_ns_key_ = rename("metadata_ns");
    received_s_key_ = rename("received_s");
    received_ns_key_ = rename("received_ns");
    imm_data_key_ = rename("imm_data");
    page_number_key_ = rename("page_number");

    if (config_.queue_size == 0) {
        throw std::runtime_error("Queue size cannot be 0");
    }
    if (config_.queue_size > config_.pages) {
        throw std::runtime_error(fmt::format(
            "Queue size {} cannot be greater than the number of pages {}",
            config_.queue_size, config_.pages));
    }

    auto& adapter = hololink::module::Adapter::get_adapter();
    auto module = adapter.get_module(metadata);
    channel_ = hololink::module::RoceDataChannelInterfaceV1::get_service(metadata);
    data_channel_
        = hololink::module::DataChannelInterfaceV1::get_service(metadata);
    frame_metadata_
        = hololink::module::FrameMetadataInterfaceV1::get_service(module);
    host_metadata_.assign(frame_metadata_->block_size(), 0);

    const size_t metadata_address = hololink::module::round_up(
        config_.frame_size, hololink::module::PAGE_SIZE);
    size_t metadata_offset = config_.metadata_offset;
    if (metadata_offset == 0) {
        metadata_offset = metadata_address;
    }
    if (metadata_offset > metadata_address) {
        throw std::runtime_error(fmt::format(
            "metadata_offset={:#x} is beyond the receiver buffer limit={:#x}.",
            metadata_offset, metadata_address));
    }
    const size_t metadata_size = hololink::module::round_up(
        frame_metadata_->block_size(), hololink::module::PAGE_SIZE);
    const size_t received_frame_size = metadata_address + metadata_size;
    const size_t buffer_size = hololink::module::round_up(
        received_frame_size * config_.pages, getpagesize());
    frame_buffer_ = std::make_shared<hololink::module::ReceiverMemoryDescriptor>(
        config_.frame_context, buffer_size);

    size_t page_size = config_.page_size;
    if (!page_size) {
        page_size = received_frame_size;
    }

    receiver_ = hololink::module::RoceReceiverInterfaceV1::get_service(metadata);

    const std::string peer_ip = metadata.get<std::string>("peer_ip");
    const auto [ibv_name, ibv_port] = hololink::module::ibv_device_for_peer(peer_ip);

    const hololink_module_status_t start_status = receiver_->start(
        ibv_name,
        ibv_port,
        static_cast<uint64_t>(frame_buffer_->get()),
        buffer_size,
        config_.frame_size,
        page_size,
        config_.pages,
        metadata_offset,
        peer_ip,
        config_.queue_size);
    if (start_status != HOLOLINK_MODULE_OK) {
        return start_status;
    }

    const hololink_module_status_t attach_status
        = channel_->attach_receiver(receiver_);
    if (attach_status != HOLOLINK_MODULE_OK) {
        return attach_status;
    }
    configured_ = true;
    running_ = true;

    auto frame_ready = config_.frame_ready;
    receiver_->set_frame_ready([this, frame_ready]() {
        if (running_ && frame_ready) {
            frame_ready();
        }
    });
    return HOLOLINK_MODULE_OK;
}

hololink_module_status_t RoceNetworkReceiver::run()
{
    if (!receiver_) {
        return HOLOLINK_MODULE_INVALID_PARAMETER;
    }
    monitor_thread_ = std::make_shared<std::thread>(
        [self = receiver_, frame_context = config_.frame_context]() {
            HOLOLINK_MODULE_CUDA_CHECK(cuCtxSetCurrent(frame_context));
            self->blocking_monitor();
        });
    return HOLOLINK_MODULE_OK;
}

void RoceNetworkReceiver::destruct()
{
    running_ = false;
    // Teardown must complete even when the device is gone. On loss the control
    // plane is often unreachable, so detach_receiver() (which writes the FPGA
    // to unconfigure the channel) may time out and throw. Swallow it so the
    // rest of the teardown proceeds — this runs on the reactor thread and from
    // the destructor, where an escaping exception would abort the process.
    if (configured_ && channel_) {
        try {
            channel_->detach_receiver();
        } catch (const std::exception& e) {
            HSB_LOG_WARN("RoceNetworkReceiver::destruct: detach_receiver failed "
                         "({}); device likely gone.",
                e.what());
        }
    }
    configured_ = false;
    if (receiver_) {
        receiver_->close();
    }
    if (monitor_thread_ && monitor_thread_->joinable()) {
        monitor_thread_->join();
    }
    monitor_thread_.reset();
    receiver_.reset();
    channel_.reset();
    data_channel_.reset();
    frame_metadata_.reset();
    frame_buffer_.reset();
}

bool RoceNetworkReceiver::get_next_frame(
    unsigned timeout_ms, CUstream /*cuda_stream*/)
{
    if (!receiver_) {
        return false;
    }
    // The RoCE receiver DMAs frames straight to device memory, so there is no
    // host->device copy to place on the pipeline stream — it is ignored.
    return receiver_->get_next_frame(timeout_ms, frame_info_);
}

bool RoceNetworkReceiver::frames_ready()
{
    return receiver_ && receiver_->frames_ready();
}

CUdeviceptr RoceNetworkReceiver::frame_memory()
{
    return static_cast<CUdeviceptr>(frame_info_.frame_memory);
}

std::shared_ptr<void> RoceNetworkReceiver::frame_buffer_owner()
{
    return frame_buffer_;
}

void RoceNetworkReceiver::stamp_metadata(holoscan::MetadataDictionary& metadata)
{
    const CUresult copy_result = cuMemcpyDtoH(
        host_metadata_.data(),
        static_cast<CUdeviceptr>(frame_info_.metadata_memory),
        host_metadata_.size());
    if (copy_result != CUDA_SUCCESS) {
        throw std::runtime_error(
            "While reading frame metadata: cuMemcpyDtoH failed");
    }

    hololink::module::FrameMetadataInterfaceV1::FrameMetadata v1_metadata {};
    const hololink_module_status_t decode_status = frame_metadata_->decode(
        host_metadata_.data(), host_metadata_.size(), v1_metadata);
    if (decode_status != HOLOLINK_MODULE_OK) {
        throw std::runtime_error(std::string(
                                     "While decoding frame metadata: status ")
            + std::to_string(decode_status));
    }

    metadata.set(flags_key_, static_cast<int64_t>(v1_metadata.flags));
    metadata.set(psn_key_, static_cast<int64_t>(v1_metadata.psn));
    metadata.set(crc_key_, static_cast<int64_t>(v1_metadata.crc));
    metadata.set(frame_number_key_, static_cast<int64_t>(v1_metadata.frame_number));
    metadata.set(timestamp_s_key_, static_cast<int64_t>(v1_metadata.timestamp_s));
    metadata.set(timestamp_ns_key_, static_cast<int64_t>(v1_metadata.timestamp_ns));
    metadata.set(bytes_written_key_, static_cast<int64_t>(v1_metadata.bytes_written));
    metadata.set(metadata_s_key_, static_cast<int64_t>(v1_metadata.metadata_s));
    metadata.set(metadata_ns_key_, static_cast<int64_t>(v1_metadata.metadata_ns));
    metadata.set(received_s_key_, static_cast<int64_t>(frame_info_.received_s));
    metadata.set(received_ns_key_, static_cast<int64_t>(frame_info_.received_ns));
    metadata.set(imm_data_key_, static_cast<int64_t>(frame_info_.imm_data));
    metadata.set(page_number_key_, static_cast<int64_t>(frame_info_.imm_data & 0xFFF));
}

std::shared_ptr<hololink::module::DataChannelInterfaceV1>
RoceNetworkReceiver::data_channel()
{
    return data_channel_;
}

class RoceNetworkReceiverFactory : public NetworkReceiverFactory {
public:
    std::shared_ptr<NetworkReceiver> create() override
    {
        return std::make_shared<RoceNetworkReceiver>();
    }
};

std::shared_ptr<NetworkReceiverFactory> make_roce_network_receiver_factory()
{
    return std::make_shared<RoceNetworkReceiverFactory>();
}

} // namespace hololink::module::operators
