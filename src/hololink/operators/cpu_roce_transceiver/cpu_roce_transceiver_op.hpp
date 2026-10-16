/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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
 */

#ifndef SRC_HOLOLINK_OPERATORS_CPU_ROCE_TRANSCEIVER_OP_HPP
#define SRC_HOLOLINK_OPERATORS_CPU_ROCE_TRANSCEIVER_OP_HPP

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>

#include <hololink/operators/base_receiver_op.hpp>
#include <hololink/transport/roce/cpu_roce_transceiver.hpp>

namespace hololink::operators {

/**
 * CPU-polled RoCE transceiver operator with host or DMA-BUF payload rings.
 *
 * Like GpuRoceTransceiverOp, forward mode autonomously sends each received
 * payload back to the peer. Non-forward mode runs independent RX and TX
 * workers and exposes their host rings and flags to the application.
 * Neither mode publishes frames to the Holoscan output.
 * frame_metadata reserves the fixed HSB trailer in each RX slot; the
 * transport validates its layout. The application must configure the HSB
 * metadata-packet setting to match. Deserialization stays with the consumer.
 *
 * Placement is transport-owned. Forward uses poll_cpu; Duplex uses rx_cpu
 * and tx_cpu together or neither. -1 means unset: the polling context
 * inherits the process CPU set. The operator does not read HOLOLINK_AFFINITY.
 * A CQ monitor failure is logged and leaves the datapath inactive; like the
 * other HSB receiver operators, it is not propagated through compute().
 */
class CpuRoceTransceiverOp : public BaseReceiverOp {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(CpuRoceTransceiverOp, BaseReceiverOp);

    CpuRoceTransceiverOp() = default;
    ~CpuRoceTransceiverOp() override;

    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;
    void stop() override;

    void start_receiver() override;
    void stop_receiver() override;
    std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>> get_next_frame(
        double timeout_ms, CUstream cuda_stream) override;
    std::tuple<std::string, uint32_t> local_ip_and_port() override;
    bool frames_ready() override;

    /**
     * Raw ring views are valid only after a successful start_receiver() and
     * before stop_receiver(). A GPU-memory RX address is a CUDA device pointer
     * and must not be dereferenced by the CPU. Concurrent flag access must use
     * CpuRoceTransceiver::load_flag_acquire() and store_flag_release(); plain
     * C++ loads and stores are data races.
     */
    uint8_t* get_ring_addr(bool is_rx) const;
    size_t get_ring_stride_sz(bool is_rx) const;
    uint32_t get_ring_stride_num(bool is_rx) const;
    uint64_t* get_ring_flag_addr(bool is_rx) const;

private:
    using Transceiver = hololink::transport::roce::CpuRoceTransceiver;

    void monitor_loop() noexcept;

    holoscan::Parameter<std::string> ibv_name_;
    holoscan::Parameter<uint32_t> ibv_port_;
    holoscan::Parameter<uint32_t> tx_ibv_qp_;
    holoscan::Parameter<uint32_t> slots_;
    holoscan::Parameter<uint32_t> forward_;
    holoscan::Parameter<bool> gpu_memory_;
    holoscan::Parameter<bool> frame_metadata_;
    holoscan::Parameter<int> poll_cpu_;
    holoscan::Parameter<int> rx_cpu_;
    holoscan::Parameter<int> tx_cpu_;

    std::shared_ptr<hololink::ReceiverMemoryDescriptor> rx_gpu_memory_;
    int rx_dmabuf_fd_ = -1;
    std::unique_ptr<Transceiver> transceiver_;
    std::unique_ptr<std::thread> monitor_thread_;
    std::mutex monitor_mutex_;
    std::condition_variable monitor_cv_;
    bool hololink_configured_ = false;
    bool stopped_ = false;
};

} // namespace hololink::operators

#endif // SRC_HOLOLINK_OPERATORS_CPU_ROCE_TRANSCEIVER_OP_HPP
