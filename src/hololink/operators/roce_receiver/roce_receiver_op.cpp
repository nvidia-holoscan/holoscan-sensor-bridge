/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "roce_receiver_op.hpp"
#include "../base_receiver_op.inl"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>

#include <exception>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/networking.hpp>

#include "roce_receiver.hpp"

namespace hololink::operators {

void RoceReceiverOp::start_receiver()
{
    const auto& p = this->params();

    if (p.queue_size == 0) {
        throw std::runtime_error("Queue size cannot be 0");
    }
    if (p.queue_size > p.pages) {
        throw std::runtime_error(fmt::format("Queue size {} cannot be greater than the number of pages {}", p.queue_size, p.pages));
    }

    // Compute effective metadata offset (auto-detect when param is zero).
    size_t metadata_address = hololink::core::round_up(p.frame_size, hololink::core::PAGE_SIZE);
    size_t metadata_offset = (p.metadata_offset == 0) ? metadata_address : p.metadata_offset;
    if (metadata_offset < metadata_address) {
        throw std::runtime_error(fmt::format(
            "metadata_offset {:#x} overlaps frame data (frame ends at {:#x})",
            metadata_offset, metadata_address));
    }

    // received_frame_size wants to be page aligned; prove that METADATA_SIZE doesn't upset that.
    // Prove that PAGE_SIZE is a power of two
    static_assert((hololink::core::PAGE_SIZE & (hololink::core::PAGE_SIZE - 1)) == 0, // power of two
        "PAGE_SIZE must be a power of two");
    // Prove that METADATA_SIZE is an even multiple of PAGE_SIZE
    static_assert((hololink::METADATA_SIZE & (hololink::core::PAGE_SIZE - 1)) == 0, // even multiple
        "METADATA_SIZE must be an even multiple of PAGE_SIZE");

    // Size each page from the effective metadata_offset (not metadata_address) so
    // a caller-supplied offset that lies beyond the page-aligned frame end is
    // still fully covered by the allocated buffer.
    size_t received_frame_size = metadata_offset + hololink::METADATA_SIZE;
    size_t buffer_size = hololink::core::round_up(
        received_frame_size * p.pages, static_cast<size_t>(getpagesize()));

    // Without an ibv_name, the device that reaches the board; picked before anything is
    // allocated, so that failing to find one leaves nothing to undo.
    const auto [ibv_name, ibv_port] = p.ibv_name.empty()
        ? hololink_channel_->infiniband_device()
        : std::make_pair(p.ibv_name, p.ibv_port);

    frame_memory_ = std::make_shared<hololink::ReceiverMemoryDescriptor>(
        p.frame_context, buffer_size);
    HSB_LOG_INFO("frame_size={:#x} frame={:#x} buffer_size={:#x}", p.frame_size, frame_memory_->get(), buffer_size);

    const std::string& peer_ip = hololink_channel_->peer_ip();
    HSB_LOG_INFO("ibv_name={} ibv_port={} peer_ip={}", ibv_name, ibv_port, peer_ip);

    receiver_.reset(new RoceReceiver(
        ibv_name.c_str(),
        ibv_port,
        frame_memory_->get(),
        buffer_size,
        p.frame_size,
        received_frame_size,
        p.pages,
        metadata_offset,
        peer_ip.c_str(),
        p.queue_size,
        frame_memory_->get_host_ptr()));
    receiver_->set_frame_ready([this](const RoceReceiver&) {
        this->frame_ready();
    });

    if (!receiver_->start()) {
        // start() has released the verbs resources it created.
        receiver_.reset();
        frame_memory_.reset();
        throw std::runtime_error("Failed to start RoceReceiver");
    }

    // From here on only the receiver thread frees the verbs resources start() created (when
    // blocking_monitor() returns), so every failure closes and joins it, as stop_receiver() does.
    receiver_thread_.reset(new std::thread(&RoceReceiverOp::run, this));
    // pthread_setname_np silently truncates names longer than 15 characters.
    const int rc = pthread_setname_np(receiver_thread_->native_handle(), "roce_receiver");
    if (rc != 0) {
        HSB_LOG_WARN("Failed to set receiver thread name, error={}", rc);
    }

    try {
        hololink_channel_->authenticate(receiver_->get_qp_number(), receiver_->get_rkey());

        auto [local_ip, local_port] = local_ip_and_port();
        HSB_LOG_INFO("local_ip={} local_port={}", local_ip, local_port);

        hololink_channel_->configure_roce(
            receiver_->external_frame_memory(),
            p.frame_size,
            received_frame_size,
            p.pages,
            local_port);
    } catch (const std::exception& e) {
        // authenticate(), local_ip_and_port() or configure_roce() failed.  Log
        // the reason, join the receiver thread (which frees the verbs
        // resources), then re-throw so the caller sees a clean kStart failure
        // rather than a dangling thread.
        HSB_LOG_ERROR("start_receiver failed after receiver thread started: {}", e.what());
        receiver_->close();
        receiver_thread_->join();
        receiver_thread_.reset();
        receiver_.reset();
        frame_memory_.reset();
        throw;
    }
}

void RoceReceiverOp::run()
{
    CudaCheck(cuCtxSetCurrent(this->params().frame_context));
    receiver_->blocking_monitor();
}

void RoceReceiverOp::stop_receiver()
{
    // Close and join even if the board doesn't answer (destroying a joinable thread terminates
    // the process), then report that.
    std::exception_ptr unconfigure_error;
    try {
        hololink_channel_->unconfigure();
    } catch (...) {
        unconfigure_error = std::current_exception();
    }
    data_socket_.reset();
    // Guard against being called after a failed start_receiver() that already
    // cleaned up receiver_ and receiver_thread_ in the configure_roce catch block.
    if (receiver_) {
        receiver_->close();
    }
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

bool RoceReceiverOp::frames_ready()
{
    return receiver_->frames_ready();
}

bool RoceReceiverOp::receive_path_failed()
{
    return receiver_ && receiver_->receive_path_failed();
}

std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>>
RoceReceiverOp::get_next_frame(double timeout_ms, CUstream /*cuda_stream*/)
{
    RoceReceiverMetadata roce_receiver_metadata;
    if (!receiver_->get_next_frame(timeout_ms, roce_receiver_metadata)) {
        return {};
    }

    auto metadata = std::make_shared<Metadata>();
    (*metadata)["received_frame_number"] = int64_t(roce_receiver_metadata.received_frame_number);
    (*metadata)["received_s"] = int64_t(roce_receiver_metadata.received_s);
    (*metadata)["received_ns"] = int64_t(roce_receiver_metadata.received_ns);
    (*metadata)["imm_data"] = int64_t(roce_receiver_metadata.imm_data);
    (*metadata)["page_number"] = int64_t(roce_receiver_metadata.imm_data & 0xFFF);
    CUdeviceptr frame_memory = roce_receiver_metadata.frame_memory;
    (*metadata)["dropped"] = int64_t(roce_receiver_metadata.dropped);
    (*metadata)["frame_number"] = int64_t(roce_receiver_metadata.frame_number);
    (*metadata)["timestamp_s"] = int64_t(roce_receiver_metadata.frame_metadata.timestamp_s);
    (*metadata)["timestamp_ns"] = int64_t(roce_receiver_metadata.frame_metadata.timestamp_ns);
    (*metadata)["metadata_s"] = int64_t(roce_receiver_metadata.frame_metadata.metadata_s);
    (*metadata)["metadata_ns"] = int64_t(roce_receiver_metadata.frame_metadata.metadata_ns);
    (*metadata)["crc"] = int64_t(roce_receiver_metadata.frame_metadata.crc);
    (*metadata)["bytes_written"] = int64_t(roce_receiver_metadata.frame_metadata.bytes_written);

    return { frame_memory, metadata };
}

std::tuple<std::string, uint32_t> RoceReceiverOp::local_ip_and_port()
{
    sockaddr_in ip {};
    ip.sin_family = AF_UNSPEC;
    socklen_t ip_len = sizeof(ip);
    if (getsockname(data_socket_.get(), reinterpret_cast<sockaddr*>(&ip), &ip_len) < 0) {
        throw std::runtime_error(
            fmt::format("getsockname failed with errno={}: \"{}\"", errno, strerror(errno)));
    }

    const std::string local_ip = inet_ntoa(ip.sin_addr);
    // RoCE traffic always arrives on the standard RoCEv2 UDP port 4791.
    const in_port_t local_port = 4791;
    return { local_ip, local_port };
}

template class BaseReceiverOp<RoceReceiverOp, RoceReceiverParams>;

} // namespace hololink::operators
