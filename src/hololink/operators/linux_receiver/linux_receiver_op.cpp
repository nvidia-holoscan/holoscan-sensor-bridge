/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "linux_receiver_op.hpp"
#include "../base_receiver_op.inl"

#include <cstdlib>
#include <exception>
#include <sched.h>
#include <sys/socket.h>
#include <unistd.h>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/networking.hpp>

#include "linux_receiver.hpp"

namespace hololink::operators {

namespace {
    constexpr int DEFAULT_RECEIVER_AFFINITY = 2;
    constexpr int BUFFER_ALIGNMENT = 0x10000; // 64 KiB boundary
} // unnamed namespace

void LinuxReceiverOp::set_receiver_affinity(const std::vector<int>& affinity)
{
    affinity_override_ = affinity;
}

void LinuxReceiverOp::start_receiver()
{
    const auto& p = this->params();

    // Resolve effective CPU affinity (override > params > env-var > default).
    {
        std::vector<int> base = affinity_override_ ? *affinity_override_ : p.receiver_affinity;
        if (base.empty()) {
            const char* env = std::getenv("HOLOLINK_AFFINITY");
            if (env && std::strlen(env) > 0) {
                base = { std::atoi(env) };
            } else {
                base = { DEFAULT_RECEIVER_AFFINITY };
            }
        }
        // CPU_SET() is only defined for [0, CPU_SETSIZE). Reject anything else here, where the
        // start fails cleanly, not in the receiver thread, where an exception ends the process.
        for (int cpu : base) {
            if (cpu < 0 || cpu >= CPU_SETSIZE) {
                throw std::runtime_error(
                    fmt::format("CPU affinity {} is out of range [0, {})", cpu, CPU_SETSIZE));
            }
        }
        effective_affinity_ = std::move(base);
    }

    if (p.queue_size == 0) {
        throw std::runtime_error("Queue size cannot be 0");
    }
    if (p.queue_size > p.pages) {
        throw std::runtime_error(
            fmt::format("Queue size {} cannot be greater than the number of pages {}",
                p.queue_size, p.pages));
    }

    check_buffer_size(p.frame_size);

    // received_frame_size wants to be page aligned; prove that METADATA_SIZE doesn't upset that.
    // Prove that PAGE_SIZE is a power of two
    static_assert((hololink::core::PAGE_SIZE & (hololink::core::PAGE_SIZE - 1)) == 0, // power of two
        "PAGE_SIZE must be a power of two");
    // Prove that METADATA_SIZE is an even multiple of PAGE_SIZE
    static_assert((hololink::METADATA_SIZE & (hololink::core::PAGE_SIZE - 1)) == 0, // even multiple
        "METADATA_SIZE must be an even multiple of PAGE_SIZE");

    size_t metadata_address = hololink::core::round_up(p.frame_size, hololink::core::PAGE_SIZE);
    size_t received_frame_size = metadata_address + hololink::METADATA_SIZE;
    size_t buffer_size = hololink::core::round_up(
        received_frame_size * p.pages, static_cast<size_t>(getpagesize()));

    frame_memory_ = std::make_shared<hololink::ReceiverMemoryDescriptor>(p.frame_context, buffer_size);

    receiver_.reset(new LinuxReceiver(
        frame_memory_->get(),
        buffer_size,
        // each page needs to include all space in order to capture the FrameMetadata
        received_frame_size,
        p.pages,
        data_socket_.get(),
        received_address_offset(),
        p.queue_size));

    receiver_->set_frame_ready([this](const LinuxReceiver&) {
        this->frame_ready();
    });

    receiver_thread_.reset(new std::thread(&LinuxReceiverOp::run, this));
    // pthread_setname_np silently truncates names longer than 15 characters.
    const int rc = pthread_setname_np(receiver_thread_->native_handle(), "linux_receiver");
    if (rc != 0) {
        HSB_LOG_WARN("Failed to set receiver thread name, error={}", rc);
    }

    try {
        hololink_channel_->authenticate(receiver_->get_qp_number(), receiver_->get_rkey());

        auto [local_ip, local_port] = local_ip_and_port();
        HSB_LOG_INFO("local_ip={} local_port={}", local_ip, local_port);

        // HSB is configured to start with address 0.
        uint64_t distal_memory_address_start = 0;
        hololink_channel_->configure_roce(
            distal_memory_address_start,
            p.frame_size,
            received_frame_size,
            p.pages,
            local_port);
    } catch (const std::exception& e) {
        HSB_LOG_ERROR("start_receiver failed after receiver thread started: {}", e.what());
        receiver_->close();
        receiver_thread_->join();
        receiver_thread_.reset();
        receiver_.reset();
        throw;
    }
}

void LinuxReceiverOp::run()
{
    CudaCheck(cuCtxSetCurrent(this->params().frame_context));

    if (!effective_affinity_.empty()) {
        cpu_set_t cpu_set;
        CPU_ZERO(&cpu_set);
        // start_receiver() checked the range.
        for (int cpu : effective_affinity_) {
            CPU_SET(cpu, &cpu_set);
        }
        if (sched_setaffinity(0, sizeof(cpu_set), &cpu_set) != 0) {
            HSB_LOG_WARN("Failed to set CPU affinity, errno={}", errno);
        }
    }

    receiver_->run();
}

void LinuxReceiverOp::stop_receiver()
{
    // Close and join even if the board doesn't answer (destroying a joinable thread terminates
    // the process), then report that.
    std::exception_ptr unconfigure_error;
    try {
        hololink_channel_->unconfigure();
    } catch (...) {
        unconfigure_error = std::current_exception();
    }
    // Guard against being called after a start_receiver() that failed and cleaned up already.
    if (receiver_) {
        receiver_->close();
    }
    data_socket_.reset();
    if (receiver_thread_) {
        receiver_thread_->join();
        receiver_thread_.reset();
    }
    receiver_.reset();
    frame_memory_.reset();
    if (unconfigure_error) {
        std::rethrow_exception(unconfigure_error);
    }
}

std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>>
LinuxReceiverOp::get_next_frame(double timeout_ms, CUstream cuda_stream)
{
    LinuxReceiverMetadata linux_receiver_metadata;
    if (!receiver_->get_next_frame(static_cast<unsigned>(timeout_ms), linux_receiver_metadata, cuda_stream)) {
        return {};
    }

    auto metadata = std::make_shared<Metadata>();
    (*metadata)["frame_packets_received"] = int64_t(linux_receiver_metadata.frame_packets_received);
    (*metadata)["frame_bytes_received"] = int64_t(linux_receiver_metadata.frame_bytes_received);
    (*metadata)["frame_number"] = int64_t(linux_receiver_metadata.frame_number);
    (*metadata)["received_frame_number"] = int64_t(linux_receiver_metadata.received_frame_number);
    (*metadata)["received_s"] = int64_t(linux_receiver_metadata.received_s);
    (*metadata)["received_ns"] = int64_t(linux_receiver_metadata.received_ns);
    (*metadata)["timestamp_s"] = int64_t(linux_receiver_metadata.frame_metadata.timestamp_s);
    (*metadata)["timestamp_ns"] = int64_t(linux_receiver_metadata.frame_metadata.timestamp_ns);
    (*metadata)["metadata_s"] = int64_t(linux_receiver_metadata.frame_metadata.metadata_s);
    (*metadata)["metadata_ns"] = int64_t(linux_receiver_metadata.frame_metadata.metadata_ns);
    (*metadata)["packets_dropped"] = int64_t(linux_receiver_metadata.packets_dropped);
    (*metadata)["crc"] = int64_t(linux_receiver_metadata.frame_metadata.crc);
    (*metadata)["psn"] = int64_t(linux_receiver_metadata.frame_metadata.psn);
    const auto imm_data = linux_receiver_metadata.imm_data;
    (*metadata)["imm_data"] = int64_t(imm_data);
    (*metadata)["page_number"] = int64_t(imm_data & 0xFFF);
    (*metadata)["bytes_written"] = int64_t(linux_receiver_metadata.frame_metadata.bytes_written);

    return { frame_memory_->get(), metadata };
}

bool LinuxReceiverOp::frames_ready()
{
    return receiver_->frames_ready();
}

uint64_t LinuxReceiverOp::received_address_offset()
{
    // This address is added to the address received from HSB;
    // HSB is configured to start with address 0.
    return frame_memory_->get();
}

void LinuxReceiverOp::check_buffer_size(size_t data_memory_size)
{
    int receiver_buffer_size = 0;
    socklen_t opt_len = sizeof(receiver_buffer_size);

    if (getsockopt(data_socket_.get(), SOL_SOCKET, SO_RCVBUF, &receiver_buffer_size, &opt_len) < 0) {
        throw std::runtime_error(fmt::format("getsockopt failed with errno={}: \"{}\"", errno, strerror(errno)));
    }

    if (static_cast<size_t>(receiver_buffer_size) < data_memory_size) {
        // Round it up to a 64k boundary
        const size_t boundary = static_cast<size_t>(BUFFER_ALIGNMENT) - 1;
        const size_t request_size = (data_memory_size + boundary) & ~boundary;

        if (setsockopt(data_socket_.get(), SOL_SOCKET, SO_RCVBUF, &request_size, sizeof(request_size)) < 0) {
            HSB_LOG_WARN("Failed to set socket buffer size, errno={}: \"{}\"", errno, strerror(errno));
        } else {
            if (getsockopt(data_socket_.get(), SOL_SOCKET, SO_RCVBUF, &receiver_buffer_size, &opt_len) < 0) {
                HSB_LOG_WARN("Failed to get socket buffer size after setting, errno={}: \"{}\"", errno, strerror(errno));
            }
        }

        HSB_LOG_DEBUG("receiver buffer size={}", receiver_buffer_size);
        if (static_cast<size_t>(receiver_buffer_size) < data_memory_size) {
            HSB_LOG_WARN("Kernel receiver buffer size is too small; performance will be unreliable.");
            HSB_LOG_WARN("Resolve this with \"echo {} | sudo tee /proc/sys/net/core/rmem_max\"", request_size);
        }
    }
}

template class BaseReceiverOp<LinuxReceiverOp, LinuxReceiverParams>;

} // namespace hololink::operators
