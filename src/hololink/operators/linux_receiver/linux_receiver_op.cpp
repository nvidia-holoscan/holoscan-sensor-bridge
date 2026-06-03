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

#include <cstdlib>
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
    constexpr int BUFFER_ALIGNMENT = 0x10000; // 64k boundary
}

void LinuxReceiverOp::initialize()
{
    // Set default identity function if rename_metadata is not set
    if (!rename_metadata_) {
        rename_metadata_ = [](const std::string& name) { return name; };
    }

    // Cache the metadata key names using the rename callback
    const auto& rename_fn = rename_metadata_;
    frame_packets_received_metadata_ = rename_fn("frame_packets_received");
    frame_number_metadata_ = rename_fn("frame_number");
    received_frame_number_metadata_ = rename_fn("received_frame_number");
    frame_bytes_received_metadata_ = rename_fn("frame_bytes_received");
    received_s_metadata_ = rename_fn("received_s");
    received_ns_metadata_ = rename_fn("received_ns");
    timestamp_s_metadata_ = rename_fn("timestamp_s");
    timestamp_ns_metadata_ = rename_fn("timestamp_ns");
    metadata_s_metadata_ = rename_fn("metadata_s");
    metadata_ns_metadata_ = rename_fn("metadata_ns");
    packets_dropped_metadata_ = rename_fn("packets_dropped");
    crc_metadata_ = rename_fn("crc");
    imm_data_metadata_ = rename_fn("imm_data");
    psn_metadata_ = rename_fn("psn");
    page_number_metadata_ = rename_fn("page_number");
    bytes_written_metadata_ = rename_fn("bytes_written");

    // Set default receiver affinity if not set
    if (receiver_affinity_.has_value() && receiver_affinity_.get().empty()) {
        // Check environment variable
        const char* affinity_env = std::getenv("HOLOLINK_AFFINITY");
        if (affinity_env && std::strlen(affinity_env) > 0) {
            receiver_affinity_ = { std::atoi(affinity_env) };
        } else {
            receiver_affinity_ = { DEFAULT_RECEIVER_AFFINITY };
        }
    }

    // Call base class initialize
    BaseReceiverOp::initialize();
}

void LinuxReceiverOp::setup(holoscan::OperatorSpec& spec)
{
    // call base class
    BaseReceiverOp::setup(spec);

    // Add our own parameters
    spec.param(receiver_affinity_, "receiver_affinity", "ReceiverAffinity",
        "CPU affinity set for receiver thread", std::vector<int> {});
    spec.param(pages_, "pages", "Pages", "Number of pages to use for the receiver memory", 1u);
    spec.param(queue_size_, "queue_size", "QueueSize",
        "The number of buffers that can be queued up for the receiver, has to be less or equal to the number of pages",
        1u);

    // Note: rename_metadata is handled programmatically via set_rename_metadata() method
    // to avoid YAML-CPP serialization issues with std::function
}

void LinuxReceiverOp::set_rename_metadata(std::function<std::string(const std::string&)> rename_fn)
{
    rename_metadata_ = rename_fn;
}

void LinuxReceiverOp::set_receiver_affinity(const std::vector<int>& affinity)
{
    receiver_affinity_ = affinity;
}

void LinuxReceiverOp::start_receiver()
{
    if (queue_size_.get() == 0) {
        throw std::runtime_error("Queue size cannot be 0");
    }
    if (queue_size_.get() > pages_.get()) {
        throw std::runtime_error(fmt::format("Queue size {{}} cannot be greater than the number of pages {{}}", queue_size_.get(), pages_.get()));
    }

    check_buffer_size(frame_size_.get());

    size_t metadata_address = hololink::core::round_up(frame_size_.get(), hololink::core::PAGE_SIZE);
    // received_frame_size wants to be page aligned; prove that METADATA_SIZE doesn't upset that.
    // Prove that PAGE_SIZE is a power of two
    static_assert((hololink::core::PAGE_SIZE & (hololink::core::PAGE_SIZE - 1)) == 0);
    // Prove that METADATA_SIZE is an even multiple of PAGE_SIZE
    static_assert((hololink::METADATA_SIZE & (hololink::core::PAGE_SIZE - 1)) == 0);
    size_t received_frame_size = metadata_address + hololink::METADATA_SIZE;
    size_t buffer_size = hololink::core::round_up(received_frame_size * pages_.get(), getpagesize());
    frame_memory_.reset(new hololink::ReceiverMemoryDescriptor(frame_context_, buffer_size));

    receiver_.reset(new LinuxReceiver(
        frame_memory_->get(),
        buffer_size,
        received_frame_size,
        pages_.get(),
        data_socket_.get(),
        received_address_offset(),
        queue_size_.get()));

    receiver_->set_frame_ready([this](const LinuxReceiver&) {
        this->frame_ready();
    });

    receiver_thread_.reset(new std::thread(&LinuxReceiverOp::run, this));
    const int error = pthread_setname_np(receiver_thread_->native_handle(), name().c_str());
    if (error != 0) {
        HSB_LOG_WARN("Failed to set thread name, error={}", error);
    }

    hololink_channel_->authenticate(receiver_->get_qp_number(), receiver_->get_rkey());

    auto [local_ip, local_port] = local_ip_and_port();
    HSB_LOG_INFO("local_ip={} local_port={}", local_ip, local_port);

    uint64_t distal_memory_address_start = 0; // See received_address_offset()
    hololink_channel_->configure_roce(distal_memory_address_start, frame_size_, received_frame_size, pages_.get(), local_port);
}

void LinuxReceiverOp::run()
{
    CudaCheck(cuCtxSetCurrent(frame_context_));

    // Set CPU affinity if specified
    if (receiver_affinity_.has_value() && !receiver_affinity_.get().empty()) {
        cpu_set_t cpu_set;
        CPU_ZERO(&cpu_set);
        for (int cpu : receiver_affinity_.get()) {
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
    hololink_channel_->unconfigure();
    data_socket_.reset();
    receiver_->close();
    receiver_thread_->join();
    receiver_thread_.reset();
    frame_memory_.reset();
}

std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>> LinuxReceiverOp::get_next_frame(double timeout_ms, CUstream cuda_stream)
{
    LinuxReceiverMetadata linux_receiver_metadata;
    if (!receiver_->get_next_frame(static_cast<unsigned>(timeout_ms), linux_receiver_metadata, cuda_stream)) {
        return {};
    }

    auto metadata = std::make_shared<Metadata>();
    (*metadata)[frame_packets_received_metadata_] = int64_t(linux_receiver_metadata.frame_packets_received);
    (*metadata)[frame_bytes_received_metadata_] = int64_t(linux_receiver_metadata.frame_bytes_received);
    (*metadata)[frame_number_metadata_] = int64_t(linux_receiver_metadata.frame_number);
    (*metadata)[received_frame_number_metadata_] = int64_t(linux_receiver_metadata.received_frame_number);
    (*metadata)[received_s_metadata_] = int64_t(linux_receiver_metadata.received_s);
    (*metadata)[received_ns_metadata_] = int64_t(linux_receiver_metadata.received_ns);
    (*metadata)[timestamp_s_metadata_] = int64_t(linux_receiver_metadata.frame_metadata.timestamp_s);
    (*metadata)[timestamp_ns_metadata_] = int64_t(linux_receiver_metadata.frame_metadata.timestamp_ns);
    (*metadata)[metadata_s_metadata_] = int64_t(linux_receiver_metadata.frame_metadata.metadata_s);
    (*metadata)[metadata_ns_metadata_] = int64_t(linux_receiver_metadata.frame_metadata.metadata_ns);
    (*metadata)[packets_dropped_metadata_] = int64_t(linux_receiver_metadata.packets_dropped);
    (*metadata)[crc_metadata_] = int64_t(linux_receiver_metadata.frame_metadata.crc);
    (*metadata)[psn_metadata_] = int64_t(linux_receiver_metadata.frame_metadata.psn);
    auto imm_data = linux_receiver_metadata.imm_data;
    (*metadata)[imm_data_metadata_] = int64_t(imm_data);
    (*metadata)[page_number_metadata_] = int64_t(imm_data & 0xFFF);
    (*metadata)[bytes_written_metadata_] = int64_t(linux_receiver_metadata.frame_metadata.bytes_written);

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
        size_t boundary = BUFFER_ALIGNMENT - 1;
        size_t request_size = (data_memory_size + boundary) & ~boundary;

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

} // namespace hololink::operators
