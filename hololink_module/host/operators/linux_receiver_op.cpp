/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hololink/module/operators/linux_receiver_op.hpp"
#include "hololink/module/operators/base_receiver_op.inl"

#include <netinet/in.h>
#include <sched.h>
#include <sys/socket.h>
#include <unistd.h> // close, getpagesize

#include <cerrno> // errno
#include <cstdint>
#include <cstdlib> // getenv
#include <cstring> // strlen
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda.h>

#include <fmt/format.h>

#include "hololink/module/adapter.hpp" // Adapter::get_adapter / get_module
#include "hololink/module/logging.hpp" // HSB_LOG_INFO
#include "hololink/module/module.hpp"
#include "hololink/module/networking.hpp" // size_receive_buffer
#include "hololink/module/page_size.hpp" // PAGE_SIZE, round_up
#include "hololink/module/receiver_memory_descriptor.hpp"
#include "hololink/module/status.h"

namespace hololink::module::operators {

static constexpr int DEFAULT_RECEIVER_AFFINITY = 2;

void LinuxReceiverOp::start_receiver()
{
    const auto& p = this->params();

    if (p.queue_size == 0) {
        throw std::runtime_error("Queue size cannot be 0");
    }
    if (p.queue_size > p.pages) {
        throw std::runtime_error(fmt::format("Queue size {} cannot be greater than the number of pages {}", p.queue_size, p.pages));
    }

    // Default the receiver affinity from HOLOLINK_AFFINITY (or a fixed
    // default) when the application didn't set one.
    std::vector<int> affinity = p.receiver_affinity;
    if (affinity.empty()) {
        const char* affinity_env = std::getenv("HOLOLINK_AFFINITY");
        if (affinity_env && std::strlen(affinity_env) > 0) {
            affinity = std::vector<int> { std::atoi(affinity_env) };
        } else {
            affinity = std::vector<int> { DEFAULT_RECEIVER_AFFINITY };
        }
    }
    // CPU_SET() is only defined for [0, CPU_SETSIZE): fail the start on anything else, before the
    // monitor thread uses it.
    for (int cpu : affinity) {
        if (cpu < 0 || cpu >= CPU_SETSIZE) {
            throw std::runtime_error(
                fmt::format("CPU affinity {} is out of range [0, {})", cpu, CPU_SETSIZE));
        }
    }

    // Resolve the supplement module that handled this metadata's
    // enumeration, then fetch the V1 services it published:
    //   - LinuxDataChannelInterfaceV1 (per (serial, data_channel))
    //   - LinuxReceiverInterfaceV1 (per (serial, data_channel))
    //   - FrameMetadataInterfaceV1 (per-module singleton; block_size only)
    auto& adapter = hololink::module::Adapter::get_adapter();
    auto module = adapter.get_module(metadata_);
    channel_ = hololink::module::LinuxDataChannelInterfaceV1::get_service(metadata_);
    frame_metadata_ = hololink::module::FrameMetadataInterfaceV1::get_service(module);

    size_t metadata_address = hololink::module::round_up(
        p.frame_size, hololink::module::PAGE_SIZE);
    // received_frame_size wants to be page aligned; the metadata block is
    // sized by the module's FrameMetadataInterfaceV1, rounded up to
    // PAGE_SIZE so the per-frame stride stays page-aligned.
    static_assert(
        (hololink::module::PAGE_SIZE & (hololink::module::PAGE_SIZE - 1)) == 0);
    const size_t metadata_size = hololink::module::round_up(
        frame_metadata_->block_size(), hololink::module::PAGE_SIZE);
    size_t received_frame_size = metadata_address + metadata_size;
    size_t buffer_size = hololink::module::round_up(
        received_frame_size * p.pages, getpagesize());
    frame_buffer_ = std::make_shared<hololink::module::ReceiverMemoryDescriptor>(
        p.frame_context, buffer_size);
    HSB_LOG_INFO("frame_size={:#x} frame={:#x} buffer_size={:#x}", p.frame_size, frame_buffer_->get(), buffer_size);

    size_t page_size = p.page_size;
    if (!page_size) {
        // Per-frame stride the FPGA cycles through. Setting this to
        // buffer_size would land every page > 0 frame outside the
        // buffer, so received_frame_size is used.
        page_size = received_frame_size;
    }

    // Create the datagram socket and bind it to the data plane via the
    // channel before the receiver runs on it. The receiver's local_port
    // (read in attach_receiver) comes from this bound socket.
    data_socket_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (data_socket_ < 0) {
        throw std::runtime_error("While starting LinuxReceiverOp: failed to create data socket");
    }
    const hololink_module_status_t socket_status = channel_->configure_socket(data_socket_);
    if (socket_status != HOLOLINK_MODULE_OK) {
        throw std::runtime_error(std::string(
                                     "While starting LinuxReceiverOp: channel->configure_socket returned status ")
            + std::to_string(socket_status));
    }
    // One frame of kernel buffering, so a stall in the single-threaded
    // reassembly loop can't overflow the socket and cost us the frame-end
    // packets that complete frames. See size_receive_buffer.
    hololink::module::size_receive_buffer(data_socket_, p.frame_size);

    receiver_ = hololink::module::LinuxReceiverInterfaceV1::get_service(metadata_);

    const hololink_module_status_t start_status = receiver_->start(
        data_socket_,
        static_cast<uint64_t>(frame_buffer_->get()),
        buffer_size,
        p.frame_size,
        page_size,
        p.pages,
        p.queue_size);
    if (start_status != HOLOLINK_MODULE_OK) {
        throw std::runtime_error(std::string(
                                     "While starting LinuxReceiverOp: LinuxReceiverV1::start returned status ")
            + std::to_string(start_status));
    }

    const hololink_module_status_t status = channel_->attach_receiver(receiver_);
    if (status != HOLOLINK_MODULE_OK) {
        throw std::runtime_error(std::string(
                                     "While starting LinuxReceiverOp: channel->attach_receiver returned status ")
            + std::to_string(status));
    }
    configured_ = true;

    // Register the frame-ready callback before the monitor thread starts
    // so no frame slips through before the condition is armed.
    receiver_->set_frame_ready([this]() { this->frame_ready(); });

    monitor_thread_ = std::make_shared<std::thread>(
        [self = receiver_, frame_context = p.frame_context, affinity]() {
            HOLOLINK_MODULE_CUDA_CHECK(cuCtxSetCurrent(frame_context));
            if (!affinity.empty()) {
                cpu_set_t cpu_set;
                CPU_ZERO(&cpu_set);
                for (int cpu : affinity) {
                    CPU_SET(cpu, &cpu_set);
                }
                if (sched_setaffinity(0, sizeof(cpu_set), &cpu_set) != 0) {
                    HSB_LOG_WARN("Failed to set receiver CPU affinity, errno={}", errno);
                }
            }
            self->blocking_monitor();
        });
}

void LinuxReceiverOp::stop_receiver()
{
    // Cleanup runs from this destructor so it executes even if other
    // calls throw below.
    class Cleanup {
    public:
        explicit Cleanup(LinuxReceiverOp& self)
            : self_(self)
        {
        }
        ~Cleanup()
        {
            self_.configured_ = false;
            if (self_.receiver_) {
                self_.receiver_->close();
            }
            if (self_.monitor_thread_ && self_.monitor_thread_->joinable()) {
                self_.monitor_thread_->join();
            }
            self_.monitor_thread_.reset();
            self_.receiver_.reset();
            self_.channel_.reset();
            self_.frame_metadata_.reset();
            self_.frame_buffer_.reset();
            if (self_.data_socket_ >= 0) {
                ::close(self_.data_socket_);
                self_.data_socket_ = -1;
            }
        }

    private:
        LinuxReceiverOp& self_;
    };
    Cleanup cleanup(*this);

    if (configured_ && channel_) {
        const hololink_module_status_t status = channel_->detach_receiver();
        if (status != HOLOLINK_MODULE_OK) {
            throw std::runtime_error(std::string(
                                         "While stopping LinuxReceiverOp: channel->detach_receiver returned status ")
                + std::to_string(status));
        }
    }
}

bool LinuxReceiverOp::frames_ready()
{
    return receiver_->frames_ready();
}

std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>>
LinuxReceiverOp::get_next_frame(double timeout_ms, CUstream cuda_stream)
{
    hololink::module::LinuxReceiverFrameInfoV1 frame_info {};
    if (!receiver_->get_next_frame(static_cast<unsigned>(timeout_ms), frame_info,
            reinterpret_cast<void*>(cuda_stream))) {
        return {};
    }

    // Unlike RoceReceiverOp, the software receiver already decoded these
    // fields while reassembling the stream -- no device-side EOF-block
    // copy or FrameMetadataInterfaceV1::decode step here.
    auto metadata = std::make_shared<hololink::Metadata>();
    (*metadata)["frame_packets_received"] = int64_t(frame_info.frame_packets_received);
    (*metadata)["frame_bytes_received"] = int64_t(frame_info.frame_bytes_received);
    (*metadata)["frame_number"] = int64_t(frame_info.frame_number);
    (*metadata)["received_frame_number"] = int64_t(frame_info.received_frame_number);
    (*metadata)["received_s"] = int64_t(frame_info.received_s);
    (*metadata)["received_ns"] = int64_t(frame_info.received_ns);
    (*metadata)["timestamp_s"] = int64_t(frame_info.timestamp_s);
    (*metadata)["timestamp_ns"] = int64_t(frame_info.timestamp_ns);
    (*metadata)["metadata_s"] = int64_t(frame_info.metadata_s);
    (*metadata)["metadata_ns"] = int64_t(frame_info.metadata_ns);
    (*metadata)["packets_dropped"] = int64_t(frame_info.packets_dropped);
    (*metadata)["crc"] = int64_t(frame_info.crc);
    (*metadata)["psn"] = int64_t(frame_info.psn);
    (*metadata)["imm_data"] = int64_t(frame_info.imm_data);
    (*metadata)["page_number"] = int64_t(frame_info.imm_data & 0xFFF);
    (*metadata)["bytes_written"] = int64_t(frame_info.bytes_written);

    return { static_cast<CUdeviceptr>(frame_info.frame_memory), metadata };
}

template class BaseReceiverOp<LinuxReceiverOp, LinuxReceiverParams>;

} // namespace hololink::module::operators
