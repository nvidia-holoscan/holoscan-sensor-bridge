/**
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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
 *
 * See README.md for detailed information.
 */
#ifndef SRC_HOLOLINK_OPERATORS_GPU_ROCE_TRANSCEIVER_OP_HPP
#define SRC_HOLOLINK_OPERATORS_GPU_ROCE_TRANSCEIVER_OP_HPP

#include <memory>
#include <string>
#include <thread>

#include "gpu_roce_transceiver.hpp"
#include "hololink/operators/roce_receiver/roce_receiver_op.hpp"

namespace hololink::operators {

/**
 * @brief DOCA-based RoCE transceiver operator
 *
 * This operator uses DOCA Verbs for GPU-based CQ/QP management instead of
 * the traditional IB-based CPU management approach.
 */
class GpuRoceTransceiverOp : public RoceReceiverOp {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(GpuRoceTransceiverOp, RoceReceiverOp);

    void setup(holoscan::OperatorSpec& spec) override;

    GpuRoceTransceiverOp() = default;
    ~GpuRoceTransceiverOp() = default;

    // Override BaseReceiverOp virtual functions
    void start() override;
    void stop() override;
    void start_receiver() override;
    void stop_receiver() override;

    std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>> get_next_frame(double timeout_ms) override;

    CUdeviceptr frame_memory()
    {
        return (CUdeviceptr)(gpu_roce_transceiver_ ? gpu_roce_transceiver_->get_rx_ring_data_addr() : 0);
    }

private:
    holoscan::Parameter<std::string> ibv_name_;
    holoscan::Parameter<uint32_t> ibv_port_;
    holoscan::Parameter<uint32_t> gpu_id_;
    std::shared_ptr<GpuRoceTransceiver> gpu_roce_transceiver_;

    // Thread for GPU-based monitoring
    std::unique_ptr<std::thread> gpu_monitor_thread_;

    void gpu_monitor_loop();
    // Can we get rid of it?
    // std::unique_ptr<ReceiverMemoryDescriptor> frame_memory_;
    static constexpr unsigned PAGES = 1024;
};

} // namespace hololink::operators

#endif // SRC_HOLOLINK_OPERATORS_GPU_ROCE_TRANSCEIVER_OP_HPP
