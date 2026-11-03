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

#ifndef SRC_HOLOLINK_OPERATORS_ROCE_TRANSMITTER_ROCE_TRANSMITTER_OP_HPP
#define SRC_HOLOLINK_OPERATORS_ROCE_TRANSMITTER_ROCE_TRANSMITTER_OP_HPP

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

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

namespace hololink::operators {

struct RoceTransmitterOp_ConnectionInfo {
    uint32_t qp_num;
};

struct RoceTransmitterParams {
    // empty: the verbs device on the network interface that reaches hololink_ip (the routing
    // table's), with its own port instead of ibv_port
    std::string ibv_name {};
    uint32_t ibv_port { 1 };
    std::string hololink_ip {};
    uint32_t ibv_qp { 2 };
    uint64_t buffer_size { 1024ULL * 1024 * 1024 };
    uint64_t queue_size { 1 };
    int cuda_device_id { 0 }; // CUDA device on which RDMA staging buffers are allocated
    std::function<void(const RoceTransmitterOp_ConnectionInfo&)> on_start {};
    std::function<void(const RoceTransmitterOp_ConnectionInfo&)> on_stop {};
};

// on_start/on_stop are std::function, which holoscan::field<>'s schema machinery can't
// represent (scalar/enum/string/vector/optional only) -- omitted here. No Python caller sets
// them; they keep their no-op default, same as every C++-only caller today.
HOLOSCAN_PARAM_SCHEMA(
    RoceTransmitterParams,
    holoscan::param_schema("com.nvidia.hololink.operators.RoceTransmitterParams", 1),
    holoscan::field<&RoceTransmitterParams::ibv_name>("ibv_name")
        .description("IBV device name used for transmission; empty: the one on the network "
                     "interface that reaches hololink_ip."),
    holoscan::field<&RoceTransmitterParams::ibv_port>("ibv_port")
        .description("Port number of the IBV device."),
    holoscan::field<&RoceTransmitterParams::hololink_ip>("hololink_ip")
        .description("IP address of the Hololink board receiving this stream."),
    holoscan::field<&RoceTransmitterParams::ibv_qp>("ibv_qp")
        .description("QP number for the IBV stream data is transmitted to."),
    holoscan::field<&RoceTransmitterParams::buffer_size>("buffer_size")
        .description("Maximum size in bytes of a single transmitted buffer."),
    holoscan::field<&RoceTransmitterParams::queue_size>("queue_size")
        .description("Number of buffers that can wait to be transmitted."),
    holoscan::field<&RoceTransmitterParams::cuda_device_id>("cuda_device_id")
        .description("CUDA device on which RDMA staging buffers are allocated."));

/***
 * The RoceTransmitterOp operator sends data over the RoCE NIC.
 *
 * Input:
 *  'input' - The port that receives the tensor to be sent.
 *
 * The worker thread establishes its own CUDA context, registers the tensor's
 * GPU memory as an RDMA region, and posts ibverbs Send Work Requests.
 */
class RoceTransmitterOp : public holoscan::Operator<RoceTransmitterParams> {
public:
    RoceTransmitterOp() = default;

    void setup(holoscan::OperatorSpec& spec) override;
    [[nodiscard]] holoscan::Contract contract() const override;
    void start() override;
    void stop() override;
    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
        holoscan::ExecutionContext& context) override;

    /***
     * Flushes the pending buffers.
     */
    bool flush(std::chrono::milliseconds timeout = std::chrono::milliseconds(0));

    // Connection info exposed via on_start / on_stop callbacks.
    using ConnectionInfo = RoceTransmitterOp_ConnectionInfo;

    struct Error : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    // Public: graph wiring (add_flow) and contract() both reference this by address.
    holoscan::Input<holoscan::Tensor> input;

protected:
    std::mutex& get_lock(); // Ensures reentrancy protection for ibv calls.

private:
    class Buffer;
    struct Resource;
    std::shared_ptr<Resource> resource_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> running_ {};
    std::thread completion_thread_;
    std::thread transmission_thread_;

    bool connect();
    void queue_buffer(holoscan::Sample<holoscan::Tensor> sample, const holoscan::CancellationView& cancellation,
        cudaStream_t producer_stream);
    bool send(uint64_t id, std::shared_ptr<Buffer> buffer);
    void completion_thread();
    void transmission_thread();
};

} // namespace hololink::operators

#endif /* SRC_HOLOLINK_OPERATORS_ROCE_TRANSMITTER_ROCE_TRANSMITTER_OP_HPP */
