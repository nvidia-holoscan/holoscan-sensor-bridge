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
#include "udp_transmitter_op.hpp"

#include <cuda_runtime.h>
#include <hololink/core/logging_internal.hpp>

namespace hololink::operators {

namespace {
    // The field size sets a theoretical limit of 65,535 bytes
    // (8-byte header + 65,527 bytes of data) for a UDP datagram.
    // However, the actual limit for the data length, which is
    // imposed by the underlying IPv4 protocol, is 65,507 bytes
    // (65,535 bytes − 8-byte UDP header − 20-byte IP header).
    static const uint16_t max_udp_data_size = 65507;
} // unnamed namespace

void UdpTransmitterOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input(input, "input")
        .expects_tensor(holoscan::TensorInputSpec {
            .representation = holoscan::TensorRepresentation {
                .memory_kind = holoscan::MemoryKind::kCudaDevice,
            },
        });
}

holoscan::Contract UdpTransmitterOp::contract() const
{
    holoscan::Contract result;
    result.trigger(holoscan::OnEach { input });
    return result;
}

void UdpTransmitterOp::start()
{
    const auto& p = params();
    const auto ip = p.ip;
    const auto port = p.port;
    const auto max_buffer_size = p.max_buffer_size;

    if (max_buffer_size == 0) {
        throw std::runtime_error("UdpTransmitterOp: max_buffer_size must be > 0");
    }
    if (max_buffer_size > max_udp_data_size) {
        throw std::runtime_error(
            fmt::format("UdpTransmitterOp: max_buffer_size {} exceeds the IPv4 UDP payload limit {}",
                max_buffer_size, max_udp_data_size));
    }

    // Set before the thread exists, not from inside it: stop() may run before the new
    // thread reaches its first statement, and a running_ = true executed in the thread
    // after stop()'s running_ = false would strand the loop below forever.
    running_ = true;
    thread_ = std::thread([this, ip, port, max_buffer_size] {
        socket_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (socket_ < 0) {
            perror("Socket creation failed");
            exit(EXIT_FAILURE);
        }

        // Configure server address
        memset(&destination_address_, 0, sizeof(destination_address_));
        destination_address_.sin_family = AF_INET;
        destination_address_.sin_port = htons(port);
        destination_address_.sin_addr.s_addr = inet_addr(ip.c_str());

        int current_cuda_device = -1;
        try {
            while (true) {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return !running_ || !tensor_queue_.empty(); });
                if (!running_)
                    break;
                auto sample = std::move(tensor_queue_.front());
                tensor_queue_.pop();
                lock.unlock();
                cv_.notify_all();

                const auto& tensor = sample.data;
                const int64_t total_bytes = tensor.nbytes();
                if (total_bytes == 0)
                    continue;

                // Switch CUDA context to the tensor's source device when needed.
                const DLDevice dev = tensor.device();
                if (dev.device_type == kDLCUDA && dev.device_id != current_cuda_device) {
                    cudaSetDevice(dev.device_id);
                    current_cuda_device = dev.device_id;
                }

                // Wait for the producer's GPU work to finish, then copy to host.
                if (auto r = tensor.synchronize_for_host(); !r)
                    throw std::runtime_error(
                        fmt::format("synchronize_for_host failed: {}",
                            holoscan::error_code_name(r.error().code)));
                host_buffer_.resize(total_bytes);
                if (cudaMemcpy(host_buffer_.data(), tensor.data(), total_bytes, cudaMemcpyDeviceToHost)
                    != cudaSuccess)
                    throw std::runtime_error("cudaMemcpy DeviceToHost failed");

                // A tensor may be larger than max_buffer_size, so split it across
                // multiple UDP datagrams instead of silently truncating it.
                int64_t offset = 0;
                while (offset < total_bytes) {
                    const int64_t chunk_size = std::min(
                        static_cast<int64_t>(max_buffer_size), total_bytes - offset);

                    auto bytes = ::sendto(
                        socket_,
                        host_buffer_.data() + offset,
                        chunk_size,
                        0,
                        reinterpret_cast<sockaddr*>(&destination_address_),
                        sizeof(destination_address_));

                    if (bytes < 0)
                        throw std::runtime_error(
                            fmt::format("UDP send failed: {} (errno: {})", strerror(errno), errno));

                    HSB_LOG_DEBUG("Socket: {}, Size: {}, Bytes sent: {}",
                        socket_, chunk_size, bytes);

                    offset += bytes;
                }
            }
        } catch (const std::exception& e) {
            HSB_LOG_ERROR("UDP transmitter worker thread error: {}", e.what());
            {
                std::lock_guard<std::mutex> lock(mutex_);
                running_ = false;
            }
            cv_.notify_all();
        }

        if (socket_ >= 0) {
            close(socket_);
            socket_ = -1;
        }
    });
}

void UdpTransmitterOp::stop()
{
    // The send loop re-checks running_ from cv_.wait's predicate, which it evaluates
    // holding mutex_.  Clearing the flag without that mutex leaves a window where the
    // predicate has already read running_ as true but the thread has not yet parked on
    // cv_: the notify_all() below lands on an empty wait queue, is lost, and the thread
    // sleeps until the next notify -- which never comes once the graph is stopping, so
    // the join() below hangs forever.  Take mutex_ so the store cannot fall in that
    // window.  This is the same deadlock that hung RoceTransmitterOp::stop().
    {
        std::lock_guard<std::mutex> lock(mutex_);
        running_ = false;
    }
    cv_.notify_all();
    thread_.join();
}

holoscan::expected<void, holoscan::Error> UdpTransmitterOp::compute(
    holoscan::ExecutionContext& /*context*/)
{
    auto maybe_sample = input.receive();
    if (!maybe_sample) {
        HSB_LOG_ERROR("Failed to receive message from port 'input'");
        return holoscan::make_unexpected(std::move(maybe_sample).error());
    }

    enqueue(std::move(*maybe_sample));
    return {};
}

void UdpTransmitterOp::enqueue(holoscan::Sample<holoscan::Tensor> sample)
{
    const auto& p = params();
    const auto queue_size = p.queue_size;
    std::unique_lock<std::mutex> lock(mutex_);
    if (p.lossy && tensor_queue_.size() >= queue_size) {
        HSB_LOG_DEBUG("Dropping buffer due to lossy mode and full queue");
        return;
    }
    // Also exit the wait if the operator is shutting down, to avoid blocking
    // the compute thread indefinitely when stop() is called mid-enqueue.
    cv_.wait(lock, [this, queue_size] {
        return !running_ || queue_size == 0 || tensor_queue_.size() < queue_size;
    });
    if (!running_)
        return;
    tensor_queue_.push(std::move(sample));
    lock.unlock();
    cv_.notify_all();
}

} // namespace hololink::operators
