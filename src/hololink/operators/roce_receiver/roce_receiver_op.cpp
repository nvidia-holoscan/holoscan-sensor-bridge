/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <chrono>

#include <netinet/in.h>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/logging_internal.hpp>

#include "roce_receiver.hpp"

namespace hololink::operators {

void RoceReceiverOp::setup(holoscan::OperatorSpec& spec)
{
    // call base class
    BaseReceiverOp::setup(spec);

    // and add our own parameters
    spec.param(ibv_name_, "ibv_name", "IBVName", "IBV device to use", std::string("roceP5p3s0f0"));
    spec.param(ibv_port_, "ibv_port", "IBVPort", "Port number of IBV device", 1u);
}

void RoceReceiverOp::start_receiver()
{
    size_t metadata_address = hololink::core::round_up(frame_size_.get(), hololink::core::PAGE_SIZE);
    // page_size wants to be page aligned; prove that METADATA_SIZE doesn't upset that.
    // Prove that PAGE_SIZE is a power of two
    static_assert((hololink::core::PAGE_SIZE & (hololink::core::PAGE_SIZE - 1)) == 0);
    // Prove that METADATA_SIZE is an even multiple of PAGE_SIZE
    static_assert((hololink::METADATA_SIZE & (hololink::core::PAGE_SIZE - 1)) == 0);
    size_t page_size = metadata_address + hololink::METADATA_SIZE;
    size_t buffer_size = page_size * PAGES;
    frame_memory_.reset(new ReceiverMemoryDescriptor(frame_context_, buffer_size));
    HSB_LOG_INFO("frame_size={:#x} frame={:#x} buffer_size={:#x}", frame_size_.get(), frame_memory_->get(), buffer_size);

    const std::string& peer_ip = hololink_channel_->peer_ip();
    HSB_LOG_INFO(
        "ibv_name_={} ibv_port_={} peer_ip={}", ibv_name_.get(), ibv_port_.get(), peer_ip);
    receiver_.reset(new RoceReceiver(
        ibv_name_.get().c_str(),
        ibv_port_.get(),
        frame_memory_->get(),
        buffer_size,
        frame_size_.get(),
        page_size,
        PAGES,
        metadata_address,
        peer_ip.c_str()));
    receiver_->set_frame_ready([this](const RoceReceiver&) {
        this->frame_ready();
    });
    if (!receiver_->start()) {
        throw std::runtime_error("Failed to start RoceReceiver");
    }
    hololink_channel_->authenticate(receiver_->get_qp_number(), receiver_->get_rkey());

    receiver_thread_.reset(new std::thread(&hololink::operators::RoceReceiverOp::run, this));
    const int error = pthread_setname_np(receiver_thread_->native_handle(), name().c_str());
    if (error != 0) {
        throw std::runtime_error("Failed to set thread name");
    }

    auto [local_ip, local_port] = local_ip_and_port();
    HSB_LOG_INFO("local_ip={} local_port={}", local_ip, local_port);

    hololink_channel_->configure_roce(receiver_->external_frame_memory(), frame_size_, page_size, PAGES, local_port);
}

void RoceReceiverOp::run()
{
    CudaCheck(cuCtxSetCurrent(frame_context_));
    receiver_->blocking_monitor();
}

void RoceReceiverOp::stop_receiver()
{
    hololink_channel_->unconfigure();
    data_socket_.reset();
    receiver_->close();
    receiver_thread_->join();
    receiver_thread_.reset();
    frame_memory_.reset();
}

std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>> RoceReceiverOp::get_next_frame(double timeout_ms)
{
    RoceReceiverMetadata roce_receiver_metadata;
    if (!receiver_->get_next_frame(timeout_ms, roce_receiver_metadata)) {
        return {};
    }

    auto metadata = std::make_shared<Metadata>();
    (*metadata)["frame_number"] = int64_t(roce_receiver_metadata.frame_number);
    (*metadata)["rx_write_requests"] = int64_t(roce_receiver_metadata.rx_write_requests);
    (*metadata)["received_s"] = int64_t(roce_receiver_metadata.received_s);
    (*metadata)["received_ns"] = int64_t(roce_receiver_metadata.received_ns);
    (*metadata)["imm_data"] = int64_t(roce_receiver_metadata.imm_data);
    CUdeviceptr frame_memory = roce_receiver_metadata.frame_memory;
    (*metadata)["frame_memory"] = int64_t(frame_memory);
    (*metadata)["dropped"] = int64_t(roce_receiver_metadata.dropped);
    (*metadata)["timestamp_s"] = int64_t(roce_receiver_metadata.frame_metadata.timestamp_s);
    (*metadata)["timestamp_ns"] = int64_t(roce_receiver_metadata.frame_metadata.timestamp_ns);
    (*metadata)["metadata_s"] = int64_t(roce_receiver_metadata.frame_metadata.metadata_s);
    (*metadata)["metadata_ns"] = int64_t(roce_receiver_metadata.frame_metadata.metadata_ns);
    (*metadata)["crc"] = int64_t(roce_receiver_metadata.frame_metadata.crc);

    return { frame_memory, metadata };
}

std::tuple<std::string, uint32_t> RoceReceiverOp::local_ip_and_port()
{
    sockaddr_in ip {};
    ip.sin_family = AF_UNSPEC;
    socklen_t ip_len = sizeof(ip);
    if (getsockname(data_socket_.get(), (sockaddr*)&ip, &ip_len) < 0) {
        throw std::runtime_error(
            fmt::format("getsockname failed with errno={}: \"{}\"", errno, strerror(errno)));
    }

    const std::string local_ip = inet_ntoa(ip.sin_addr);
    // This is what you'd normally use
    // const in_port_t local_port = ip.sin_port;
    // But we're going to tell the other side that we're listening
    // to the ROCE receiver port at 4791.
    const in_port_t local_port = 4791;
    return { local_ip, local_port };
}

} // namespace hololink::operators
