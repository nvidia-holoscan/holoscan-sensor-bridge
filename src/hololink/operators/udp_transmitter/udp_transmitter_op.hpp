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

#ifndef OPERATORS_UDP_TRANSMITTER_UDP_TRANSMITTER_OP_HPP
#define OPERATORS_UDP_TRANSMITTER_UDP_TRANSMITTER_OP_HPP

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include <arpa/inet.h>

namespace hololink::operators {

struct UdpTransmitterParams {
    std::string ip { "" };
    uint16_t port { 5000 };
    uint16_t max_buffer_size { 65507 };
    uint16_t queue_size { 1 };
    bool lossy { true };
};

HOLOSCAN_PARAM_SCHEMA(
    UdpTransmitterParams,
    holoscan::param_schema("com.nvidia.hololink.operators.UdpTransmitterParams", 1),
    holoscan::field<&UdpTransmitterParams::ip>("ip")
        .description("Destination IPv4 address."),
    holoscan::field<&UdpTransmitterParams::port>("port")
        .description("Destination UDP port."),
    holoscan::field<&UdpTransmitterParams::max_buffer_size>("max_buffer_size")
        .description("Maximum bytes sent in one UDP datagram."),
    holoscan::field<&UdpTransmitterParams::queue_size>("queue_size")
        .description("Depth of the background-thread transmit queue."),
    holoscan::field<&UdpTransmitterParams::lossy>("lossy")
        .description("Drop new samples instead of blocking when the queue is full."));

// Receives a CUDA tensor and sends it over UDP from a background thread.
// The worker thread establishes its own CUDA context to copy device→host
// (via synchronize_for_host + cudaMemcpy) before transmitting via sendto().
class UdpTransmitterOp : public holoscan::Operator<UdpTransmitterParams> {
public:
    UdpTransmitterOp() = default;

    void setup(holoscan::OperatorSpec& spec) override;
    [[nodiscard]] holoscan::Contract contract() const override;
    void start() override;
    void stop() override;
    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
        holoscan::ExecutionContext& context) override;

    // Public: graph wiring (add_flow) and contract() both reference this by address.
    holoscan::Input<holoscan::Tensor> input;

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> running_ { false };
    std::thread thread_;
    ::sockaddr_in destination_address_ {};
    int socket_ { -1 };
    // Sample<Tensor> keeps the runtime tensor buffer alive until the worker
    // thread is done with it — no device→host copy needed on the compute thread.
    std::queue<holoscan::Sample<holoscan::Tensor>> tensor_queue_;
    std::vector<uint8_t> host_buffer_;

    void enqueue(holoscan::Sample<holoscan::Tensor> sample);
};

} // namespace hololink::operators

#endif /* OPERATORS_UDP_TRANSMITTER_UDP_TRANSMITTER_OP_HPP */
