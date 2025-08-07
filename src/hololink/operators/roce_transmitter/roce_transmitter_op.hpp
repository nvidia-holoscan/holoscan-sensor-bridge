/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights
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

#ifndef SRC_HOLOLINK_OPERATORS_ROCE_TRANSMITTER_ROCE_TRANSMITTER_OP_HPP
#define SRC_HOLOLINK_OPERATORS_ROCE_TRANSMITTER_ROCE_TRANSMITTER_OP_HPP

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include <holoscan/holoscan.hpp>

namespace hololink::operators {

/***
 * The RoceTransmitterOp operator sends data over the RoCE NIC.
 *
 * Input:
 *  'in' - The port that receives the data to be sent.
 */
class RoceTransmitterOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(RoceTransmitterOp)

    RoceTransmitterOp() = default;

    void start() override;
    void stop() override;

    void setup(holoscan::OperatorSpec& spec) override;

    void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
        holoscan::ExecutionContext& context) override;

    /***
     * Flushes the pending buffers.
     */
    bool flush(std::chrono::milliseconds timeout = std::chrono::milliseconds(0));

    struct Error : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    struct ConnectionInfo {
        uint32_t qp_num;
    };
    using OnStartCallback = std::function<void(const ConnectionInfo&)>;
    using OnStopCallback = std::function<void(const ConnectionInfo&)>;

protected:
    std::mutex& get_lock(); // Ensures reentrency protection for ibv calls.

private:
    using Tensor = std::shared_ptr<holoscan::Tensor>;
    class Buffer;
    struct Resource;
    std::shared_ptr<Resource> resource_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool flushing_ {};
    std::atomic<bool> running_ {};
    std::thread completion_thread_;
    std::thread transmission_thread_;

    bool connect();
    void queue_buffer(const Tensor& buffer);
    bool send(uint64_t id, std::shared_ptr<Buffer> buffer);
    void completion_thread();
    void transmission_thread();

    // Parameters
    holoscan::Parameter<std::string> ibv_name_;
    holoscan::Parameter<uint32_t> ibv_port_;
    holoscan::Parameter<std::string> hololink_ip_;
    holoscan::Parameter<uint32_t> ibv_qp_;
    holoscan::Parameter<uint64_t> buffer_size_;
    holoscan::Parameter<uint64_t> queue_size_;
    holoscan::Parameter<OnStartCallback> on_start_;
    holoscan::Parameter<OnStopCallback> on_stop_;
};

} // namespace hololink::operators

#endif /* SRC_HOLOLINK_OPERATORS_ROCE_TRANSMITTER_ROCE_TRANSMITTER_OP_HPP */
