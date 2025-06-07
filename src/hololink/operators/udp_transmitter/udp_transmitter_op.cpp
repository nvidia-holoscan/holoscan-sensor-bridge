/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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
#include "udp_transmitter_op.hpp"

#include <hololink/common/cuda_error.hpp>
#include <hololink/core/logging_internal.hpp>

namespace hololink::operators {
namespace {
    static const char* input_name = "input";
    // The field size sets a theoretical limit of 65,535 bytes
    // (8-byte header + 65,527 bytes of data) for a UDP datagram.
    // However, the actual limit for the data length, which is
    // imposed by the underlying IPv4 protocol, is 65,507 bytes
    // (65,535 bytes − 8-byte UDP header − 20-byte IP header).
    static const uint16_t max_udp_data_size = 65507;
} // unnamed namespace

void UdpTransmitterOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input<Tensor>(input_name);

    spec.param(destination_ip_,
        "ip",
        "ip",
        "Destination IP",
        std::string(""));

    spec.param(destination_port_,
        "port",
        "port",
        "Destination Port",
        static_cast<uint16_t>(5000));

    spec.param(max_buffer_size_,
        "max_buffer_size",
        "max buffer size",
        "Maximum buffer size",
        static_cast<uint16_t>(max_udp_data_size));

    spec.param(queue_size_,
        "queue_size",
        "queue size",
        "The number of buffers that can wait to be transmitted",
        static_cast<uint16_t>(1));

    // In lossy mode, if the incoming data rate is higher than the outgoing data rate,
    // the operator will drop the incoming data if the queue is full.
    spec.param(lossy_,
        "lossy",
        "lossy",
        "Lossy mode",
        true);
}

void UdpTransmitterOp::start()
{
    thread_ = std::thread([this] {
        // Create UDP socket
        socket_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (socket_ < 0) {
            perror("Socket creation failed");
            exit(EXIT_FAILURE);
        }

        // Configure server address
        memset(&destination_address_, 0, sizeof(destination_address_));
        destination_address_.sin_family = AF_INET;
        destination_address_.sin_port = htons(destination_port_.get());
        destination_address_.sin_addr.s_addr = inet_addr(destination_ip_.get().c_str());

        running_ = true;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return !running_ || (!tensor_queue_.empty()); });
            if (!running_)
                break;
            auto tensor = std::move(tensor_queue_.front());
            tensor_queue_.pop();
            lock.unlock();
            cv_.notify_all();
            host_buffer_.resize(std::min(static_cast<int64_t>(max_buffer_size_), tensor->nbytes()));
            memset(host_buffer_.data(), 0, host_buffer_.size());
            cudaMemcpy(&host_buffer_.front(), tensor->data(), host_buffer_.size(), cudaMemcpyDeviceToHost);

            // Send data via UDP
            auto bytes = ::sendto(
                socket_,
                &host_buffer_.front(),
                host_buffer_.size(),
                0,
                reinterpret_cast<sockaddr*>(&destination_address_),
                sizeof(destination_address_));

            if (bytes < 0)
                throw std::runtime_error(fmt::format("UDP send failed: {} (errno: {})", strerror(errno), errno));

            HSB_LOG_DEBUG("Socket: {}, Size: {}, Bytes sent: {}, Error: {}", socket_, host_buffer_.size(), bytes, strerror(errno));
        }

        close(socket_);
    });
}

void UdpTransmitterOp::stop()
{
    running_ = false;
    cv_.notify_all();
    thread_.join();
}

void UdpTransmitterOp::compute(holoscan::InputContext& op_input, [[maybe_unused]] holoscan::OutputContext& op_output,
    [[maybe_unused]] holoscan::ExecutionContext& context)
{
    auto maybe_tensor = op_input.receive<Tensor>(input_name);
    if (!maybe_tensor) {
        HSB_LOG_ERROR("Failed to receive message from port 'in'");
        return;
    }

    auto tensor = maybe_tensor.value();
    queue_buffer(std::move(tensor));
}

void UdpTransmitterOp::queue_buffer(Tensor buffer)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (lossy_.get() && tensor_queue_.size() >= queue_size_.get()) {
        HSB_LOG_DEBUG("Dropping buffer due to lossy mode and full queue");
        return;
    }
    cv_.wait(lock, [this] { return tensor_queue_.size() < queue_size_.get(); });
    tensor_queue_.push(std::move(buffer));
    lock.unlock();
    cv_.notify_all();
}

} // namespace hololink::operators
