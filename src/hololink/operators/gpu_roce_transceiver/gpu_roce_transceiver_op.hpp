/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 *
 * See README.md for detailed information.
 */

#ifndef SRC_HOLOLINK_OPERATORS_GPU_ROCE_TRANSCEIVER_OP_HPP
#define SRC_HOLOLINK_OPERATORS_GPU_ROCE_TRANSCEIVER_OP_HPP

#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "gpu_roce_transceiver.hpp"
#include "hololink/operators/base_receiver_op.hpp"

namespace hololink::operators {

/**
 * @brief GPU RoCE transceiver operator.
 *
 * This operator uses DOCA Verbs for GPU-based CQ/QP management instead of
 * the traditional IB-based CPU management approach.
 *
 * A single operator can serve several sensors of the same Hololink board: tx_ibv_qp
 * lists one remote QP per sensor ("2,4,6,8") and hololink_channels lists the
 * DataChannel of each sensor in the same order. Each pair gets its own local QP and
 * ring buffer. Without hololink_channels, hololink_channel is the only sensor.
 */
class GpuRoceTransceiverOp : public BaseReceiverOp {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(GpuRoceTransceiverOp, BaseReceiverOp);

    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void start_receiver() override;
    void stop_receiver() override;
    std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>> get_next_frame(double timeout_ms, CUstream cuda_stream) override;
    std::tuple<std::string, uint32_t> local_ip_and_port() override;
    bool frames_ready() override;

    CUdeviceptr frame_memory(unsigned sensor_id);

    uint8_t* get_ring_addr(bool is_rx, unsigned sensor_id);
    size_t get_ring_stride_sz(bool is_rx, unsigned sensor_id);
    uint32_t get_ring_stride_num(bool is_rx, unsigned sensor_id);
    uint64_t* get_ring_flag_addr(bool is_rx, unsigned sensor_id);

    GpuRoceTransceiverOp() = default;
    ~GpuRoceTransceiverOp() = default;

private:
    holoscan::Parameter<std::vector<DataChannel*>> hololink_channels_;
    holoscan::Parameter<std::string> ibv_name_;
    holoscan::Parameter<uint32_t> ibv_port_;
    holoscan::Parameter<std::string> tx_ibv_qp_;
    holoscan::Parameter<uint32_t> gpu_id_;
    holoscan::Parameter<uint32_t> forward_;
    std::shared_ptr<GpuRoceTransceiver> gpu_roce_transceiver_;
    std::unique_ptr<std::thread> gpu_monitor_thread_;
    // Channels whose HSB RoCE configuration has been (at least partially) written, to unconfigure on stop
    std::vector<DataChannel*> configured_channels_;

    void gpu_monitor_loop();
    // The DataChannel of each QP in tx_ibv_qp: hololink_channels, or hololink_channel alone
    std::vector<DataChannel*> data_channels();
    static constexpr unsigned PAGES = 1024;
};

} // namespace hololink::operators

#endif // SRC_HOLOLINK_OPERATORS_GPU_ROCE_TRANSCEIVER_OP_HPP
