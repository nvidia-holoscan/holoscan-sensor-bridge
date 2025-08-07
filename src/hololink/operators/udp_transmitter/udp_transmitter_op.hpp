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

#ifndef OPERATORS_UDP_TRANSMITTER_UDP_TRANSMITTER_OP_HPP
#define OPERATORS_UDP_TRANSMITTER_UDP_TRANSMITTER_OP_HPP

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <hololink/core/smart_object_pool.hpp>
#include <holoscan/holoscan.hpp>

#include <arpa/inet.h>

namespace hololink::operators {

class UdpTransmitterOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(UdpTransmitterOp)

    UdpTransmitterOp() = default;

    void setup(holoscan::OperatorSpec& spec) override;

    void start() override;
    void stop() override;

    void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
        holoscan::ExecutionContext& context) override;

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> running_;
    std::thread thread_;
    ::sockaddr_in destination_address_ {};
    int socket_ {};
    using Tensor = std::shared_ptr<holoscan::Tensor>;
    std::queue<Tensor> tensor_queue_;
    std::vector<uint8_t> host_buffer_;
    void queue_buffer(Tensor buffer);

    // Parameters
    holoscan::Parameter<std::string> destination_ip_;
    holoscan::Parameter<uint16_t> destination_port_;
    holoscan::Parameter<uint16_t> max_buffer_size_;
    holoscan::Parameter<uint16_t> queue_size_;
    holoscan::Parameter<bool> lossy_;
};

} // namespace hololink::operators

#endif /* OPERATORS_UDP_TRANSMITTER_UDP_TRANSMITTER_OP_HPP */
