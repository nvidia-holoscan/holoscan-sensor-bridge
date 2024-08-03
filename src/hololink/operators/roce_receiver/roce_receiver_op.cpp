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

#include <hololink/data_channel.hpp>
#include <hololink/native/cuda_helper.hpp>

#include "roce_receiver.hpp"

namespace hololink::operators {

static constexpr int64_t MS_PER_SEC = 1000;
static constexpr int64_t US_PER_SEC = 1000 * MS_PER_SEC;
static constexpr int64_t NS_PER_SEC = 1000 * US_PER_SEC;

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
    const std::string& peer_ip = hololink_channel_->peer_ip();
    HOLOSCAN_LOG_INFO(
        "ibv_name_={} ibv_port_={} peer_ip={}", ibv_name_.get(), ibv_port_.get(), peer_ip);
    receiver_.reset(new RoceReceiver(ibv_name_.get().c_str(), ibv_port_.get(), frame_memory_,
        frame_size_.get(), peer_ip.c_str()));
    if (!receiver_->start()) {
        throw std::runtime_error("Failed to start RoceReceiver");
    }
    hololink_channel_->authenticate(receiver_->get_qp_number(), receiver_->get_rkey());

    // we don't actually receive anything here because CX7 hides it.
    sockaddr_in address {};
    if (bind(data_socket_.get(), (sockaddr*)&address, sizeof(address)) < 0) {
        throw std::runtime_error(
            fmt::format("bind failed with errno={}: \"{}\"", errno, strerror(errno)));
    }

    receiver_thread_.reset(new std::thread(&hololink::operators::RoceReceiverOp::run, this));
    const int error = pthread_setname_np(receiver_thread_->native_handle(), "receiver_thread");
    if (error != 0) {
        throw std::runtime_error("Failed to set thread name");
    }
}

void RoceReceiverOp::run()
{
    CudaCheck(cuCtxSetCurrent(frame_context_));
    receiver_->blocking_monitor();
}

void RoceReceiverOp::stop_()
{
    data_socket_.reset();
    receiver_->close();
    receiver_thread_->join();
    receiver_thread_.reset();
}

std::shared_ptr<hololink::Metadata> RoceReceiverOp::get_next_frame(double timeout_ms)
{
    RoceReceiverMetadata roce_receiver_metadata;
    if (!receiver_->get_next_frame(timeout_ms, roce_receiver_metadata)) {
        return {};
    }

    const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch())
                               .count();

    // Extend the timestamp we got from the data,
    // (which is ns plus 2 bits of seconds).  Note that
    // we don't look at the 2 bits of seconds here.
    const int64_t ns = roce_receiver_metadata.imm_data % NS_PER_SEC;
    int64_t timestamp_ns = (now_ns - (now_ns % NS_PER_SEC)) + ns;
    if (timestamp_ns > now_ns) {
        timestamp_ns -= NS_PER_SEC;
    }

    auto metadata = std::make_shared<Metadata>();
    (*metadata)["frame_number"] = int64_t(roce_receiver_metadata.frame_number);
    (*metadata)["rx_write_requests"] = int64_t(roce_receiver_metadata.rx_write_requests);
    (*metadata)["received_ns"] = roce_receiver_metadata.received_ns;
    (*metadata)["timestamp_ns"] = timestamp_ns;
    (*metadata)["imm_data"] = int64_t(roce_receiver_metadata.imm_data);

    return metadata;
}

std::tuple<std::string, uint32_t> RoceReceiverOp::local_ip_and_port()
{
    auto [local_ip, local_port] = BaseReceiverOp::local_ip_and_port();
    return { local_ip, 4791 };
}

} // namespace hololink::operators
