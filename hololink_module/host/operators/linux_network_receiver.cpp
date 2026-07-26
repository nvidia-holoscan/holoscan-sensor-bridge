/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hololink/module/operators/linux_network_receiver.hpp"

#include <sched.h>
#include <sys/socket.h>
#include <unistd.h> // close, getpagesize

#include <cerrno> // errno
#include <cstdint>
#include <cstdlib> // getenv, atoi
#include <cstring> // strlen
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "hololink/module/adapter.hpp"
#include "hololink/module/cuda_unique.hpp" // HOLOLINK_MODULE_CUDA_CHECK
#include "hololink/module/logging.hpp"
#include "hololink/module/module.hpp"
#include "hololink/module/page_size.hpp" // PAGE_SIZE, round_up
#include "hololink/module/receiver_memory_descriptor.hpp"
#include "hololink/module/status.h"

namespace hololink::module::operators {

static constexpr int DEFAULT_RECEIVER_AFFINITY = 2;

LinuxNetworkReceiver::~LinuxNetworkReceiver()
{
    destruct();
}

hololink_module_status_t LinuxNetworkReceiver::construct(
    const hololink::module::EnumerationMetadata& metadata, const Config& config)
{
    config_ = config;

    auto rename = config_.rename_metadata;
    if (!rename) {
        rename = [](const std::string& name) { return name; };
    }
    frame_packets_received_key_ = rename("frame_packets_received");
    frame_bytes_received_key_ = rename("frame_bytes_received");
    frame_number_key_ = rename("frame_number");
    received_frame_number_key_ = rename("received_frame_number");
    received_s_key_ = rename("received_s");
    received_ns_key_ = rename("received_ns");
    timestamp_s_key_ = rename("timestamp_s");
    timestamp_ns_key_ = rename("timestamp_ns");
    metadata_s_key_ = rename("metadata_s");
    metadata_ns_key_ = rename("metadata_ns");
    packets_dropped_key_ = rename("packets_dropped");
    crc_key_ = rename("crc");
    psn_key_ = rename("psn");
    imm_data_key_ = rename("imm_data");
    page_number_key_ = rename("page_number");
    bytes_written_key_ = rename("bytes_written");

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
    channel_
        = hololink::module::LinuxDataChannelInterfaceV1::get_service(metadata);
    data_channel_
        = hololink::module::DataChannelInterfaceV1::get_service(metadata);
    frame_metadata_
        = hololink::module::FrameMetadataInterfaceV1::get_service(module);

    const size_t metadata_address = hololink::module::round_up(
        config_.frame_size, hololink::module::PAGE_SIZE);
    const size_t metadata_size = hololink::module::round_up(
        frame_metadata_->block_size(), hololink::module::PAGE_SIZE);
    const size_t received_frame_size = metadata_address + metadata_size;
    const size_t buffer_size = hololink::module::round_up(
        received_frame_size * config_.pages, getpagesize());
    frame_buffer_ = std::make_shared<hololink::module::ReceiverMemoryDescriptor>(
        config_.frame_context, buffer_size);

    size_t page_size = config_.page_size;
    if (!page_size) {
        // Per-frame stride the FPGA cycles through; buffer_size here would land
        // every page > 0 frame outside the buffer, so received_frame_size is used.
        page_size = received_frame_size;
    }

    // Create the datagram socket and bind it to the data plane via the channel
    // before the receiver runs on it (the receiver's local_port, read in
    // attach_receiver, comes from this bound socket).
    data_socket_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (data_socket_ < 0) {
        throw std::runtime_error(
            "While constructing LinuxNetworkReceiver: failed to create data socket");
    }
    const hololink_module_status_t socket_status
        = channel_->configure_socket(data_socket_);
    if (socket_status != HOLOLINK_MODULE_OK) {
        return socket_status;
    }

    receiver_
        = hololink::module::LinuxReceiverInterfaceV1::get_service(metadata);

    const hololink_module_status_t start_status = receiver_->start(
        data_socket_,
        static_cast<uint64_t>(frame_buffer_->get()),
        buffer_size,
        config_.frame_size,
        page_size,
        config_.pages,
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

hololink_module_status_t LinuxNetworkReceiver::run()
{
    if (!receiver_) {
        return HOLOLINK_MODULE_INVALID_PARAMETER;
    }
    // CPU affinity for the receiver worker thread, defaulted from
    // HOLOLINK_AFFINITY (or a fixed default) when the environment sets one.
    std::vector<int> affinity;
    const char* affinity_env = std::getenv("HOLOLINK_AFFINITY");
    if (affinity_env && std::strlen(affinity_env) > 0) {
        affinity = std::vector<int> { std::atoi(affinity_env) };
    } else {
        affinity = std::vector<int> { DEFAULT_RECEIVER_AFFINITY };
    }
    monitor_thread_ = std::make_shared<std::thread>(
        [self = receiver_, frame_context = config_.frame_context, affinity]() {
            HOLOLINK_MODULE_CUDA_CHECK(cuCtxSetCurrent(frame_context));
            if (!affinity.empty()) {
                cpu_set_t cpu_set;
                CPU_ZERO(&cpu_set);
                for (int cpu : affinity) {
                    // CPU_SET is undefined for out-of-range indices (it indexes
                    // a fixed-size bitmap), so drop anything outside [0,
                    // CPU_SETSIZE) rather than corrupt the set.
                    if (cpu < 0 || cpu >= CPU_SETSIZE) {
                        HSB_LOG_WARN(
                            "Ignoring out-of-range receiver CPU affinity {}", cpu);
                        continue;
                    }
                    CPU_SET(cpu, &cpu_set);
                }
                if (sched_setaffinity(0, sizeof(cpu_set), &cpu_set) != 0) {
                    HSB_LOG_WARN(
                        "Failed to set receiver CPU affinity, errno={}", errno);
                }
            }
            self->blocking_monitor();
        });
    return HOLOLINK_MODULE_OK;
}

void LinuxNetworkReceiver::destruct()
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
            HSB_LOG_WARN("LinuxNetworkReceiver::destruct: detach_receiver failed "
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
    if (data_socket_ >= 0) {
        ::close(data_socket_);
        data_socket_ = -1;
    }
}

bool LinuxNetworkReceiver::get_next_frame(unsigned timeout_ms, CUstream cuda_stream)
{
    if (!receiver_) {
        return false;
    }
    // The software receiver issues its host->device frame copy on cuda_stream,
    // so it overlaps downstream work ordered on the same stream (the operator
    // sets it on the emitted tensor). No synchronize here.
    return receiver_->get_next_frame(
        timeout_ms, frame_info_, reinterpret_cast<void*>(cuda_stream));
}

bool LinuxNetworkReceiver::frames_ready()
{
    return receiver_ && receiver_->frames_ready();
}

CUdeviceptr LinuxNetworkReceiver::frame_memory()
{
    return static_cast<CUdeviceptr>(frame_info_.frame_memory);
}

std::shared_ptr<void> LinuxNetworkReceiver::frame_buffer_owner()
{
    return frame_buffer_;
}

void LinuxNetworkReceiver::stamp_metadata(holoscan::MetadataDictionary& metadata)
{
    // The software receiver decoded these during reassembly, so they come
    // straight off the frame-info struct — no device read or decode step.
    metadata.set(frame_packets_received_key_,
        static_cast<int64_t>(frame_info_.frame_packets_received));
    metadata.set(frame_bytes_received_key_,
        static_cast<int64_t>(frame_info_.frame_bytes_received));
    metadata.set(frame_number_key_, static_cast<int64_t>(frame_info_.frame_number));
    metadata.set(received_frame_number_key_,
        static_cast<int64_t>(frame_info_.received_frame_number));
    metadata.set(received_s_key_, static_cast<int64_t>(frame_info_.received_s));
    metadata.set(received_ns_key_, static_cast<int64_t>(frame_info_.received_ns));
    metadata.set(timestamp_s_key_, static_cast<int64_t>(frame_info_.timestamp_s));
    metadata.set(timestamp_ns_key_, static_cast<int64_t>(frame_info_.timestamp_ns));
    metadata.set(metadata_s_key_, static_cast<int64_t>(frame_info_.metadata_s));
    metadata.set(metadata_ns_key_, static_cast<int64_t>(frame_info_.metadata_ns));
    metadata.set(packets_dropped_key_,
        static_cast<int64_t>(frame_info_.packets_dropped));
    metadata.set(crc_key_, static_cast<int64_t>(frame_info_.crc));
    metadata.set(psn_key_, static_cast<int64_t>(frame_info_.psn));
    metadata.set(imm_data_key_, static_cast<int64_t>(frame_info_.imm_data));
    metadata.set(page_number_key_, static_cast<int64_t>(frame_info_.imm_data & 0xFFF));
    metadata.set(bytes_written_key_, static_cast<int64_t>(frame_info_.bytes_written));
}

std::shared_ptr<hololink::module::DataChannelInterfaceV1>
LinuxNetworkReceiver::data_channel()
{
    return data_channel_;
}

class LinuxNetworkReceiverFactory : public NetworkReceiverFactory {
public:
    std::shared_ptr<NetworkReceiver> create() override
    {
        return std::make_shared<LinuxNetworkReceiver>();
    }
};

std::shared_ptr<NetworkReceiverFactory> make_linux_network_receiver_factory()
{
    return std::make_shared<LinuxNetworkReceiverFactory>();
}

} // namespace hololink::module::operators
