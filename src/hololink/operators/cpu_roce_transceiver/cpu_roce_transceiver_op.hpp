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
 */

#ifndef SRC_HOLOLINK_OPERATORS_CPU_ROCE_TRANSCEIVER_OP_HPP
#define SRC_HOLOLINK_OPERATORS_CPU_ROCE_TRANSCEIVER_OP_HPP

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>

#include <cuda.h>

#include <hololink/core/data_channel.hpp>
#include <hololink/operators/base_receiver_op.hpp>
#include <hololink/transport/roce/cpu_roce_transceiver.hpp>
#include <holoscan/core/parameter_schema.hpp>

namespace hololink::operators {

/**
 * @brief Flat parameter aggregate for CpuRoceTransceiverOp.
 *
 * Set all fields before passing to CpuRoceTransceiverOp (or use designated initializers).
 */
struct CpuRoceTransceiverParams {
    // ---- Base receiver fields (shared with BaseReceiverOp) ----
    /// CUDA context of the gpu_memory payload ring; unused with host memory.
    CUcontext frame_context { nullptr };
    size_t frame_size { 0 };
    bool trim { false }; ///< Not used: the transceiver owns the data path.
    /// Has no effect for this operator: BaseReceiverOp::contract() completes a receiver after
    /// this many compute() calls, and this one runs compute() only to fail the run when its
    /// receive path fails, so it never reaches a limit. Leave it 0.
    int64_t frame_limit { 0 };
    /// Has no effect for this operator, as frame_limit.
    bool frame_limit_ends_run { false };

    // ---- CPU RoCE transceiver specific ----
    /// IBV device to use. Empty: the one on the network interface that reaches the board
    /// (DataChannel::infiniband_device()), with its own port instead of ibv_port.
    std::string ibv_name {};
    uint32_t ibv_port { 1 }; ///< Port number of the IBV device.
    uint32_t tx_ibv_qp { 2 }; ///< Remote QP number for transmit operations.
    /// Payload ring slots (max 4096). Duplex uses this count for RX and TX; TX must be a power of
    /// two. Each slot is HSB-page-rounded frame_size, plus one HSB page for frame metadata or when
    /// frame_size exceeds one HSB page; RX is registered twice against RLIMIT_MEMLOCK for host
    /// memory.
    uint32_t slots { 1024 };
    uint32_t forward { 0 }; ///< 1: Forward mode; 0: Duplex.
    /// Use CUDA device memory exported as DMA-BUF for the Forward RX ring. Fail instead of falling
    /// back to host memory. Forward-only.
    bool gpu_memory { false };
    /// Reserve and receive the fixed HSB metadata trailer after each RX payload. The transport
    /// validates its location but does not deserialize or forward the trailer. The application
    /// must configure the HSB metadata-packet setting to match.
    bool frame_metadata { false };
    /// Forward Config::poll_cpu. -1: unset (inherit the monitor thread's mask). Duplex rejects it.
    int poll_cpu { -1 };
    /// Duplex Config::rx_cpu. Set with tx_cpu, or leave both -1.
    int rx_cpu { -1 };
    /// Duplex Config::tx_cpu. Set with rx_cpu, or leave both -1.
    int tx_cpu { -1 };
};

// A partial schema: HSDK5 only needs the field BaseReceiverOp::setup() sizes the tensor pool's
// slots from (holoscan::parameter_byte_span()); the other fields stay opaque to it and are still
// bound by the operator catalog. The schema makes frame_size a recompile-required parameter.
HOLOSCAN_PARAM_SCHEMA(
    CpuRoceTransceiverParams,
    holoscan::param_schema("com.nvidia.hololink.operators.CpuRoceTransceiverParams", 1).partial(),
    holoscan::field<&CpuRoceTransceiverParams::frame_size>("frame_size")
        .description("Size of one received frame, in bytes; the tensor pool's slot size."));

/**
 * @brief CPU-polled RoCE transceiver operator with host or DMA-BUF payload rings.
 *
 * Like GpuRoceTransceiverOp, forward mode autonomously sends each received
 * payload back to the peer. Non-forward (Duplex) mode runs independent RX and
 * TX workers and exposes their host rings and flags to the application through
 * get_ring_addr() and the other ring accessors. Neither mode emits on the
 * tensor or metadata output: get_next_frame() always returns {}.
 * frame_metadata reserves the fixed HSB trailer in each RX slot; the
 * transport validates its layout. The application must configure the HSB
 * metadata-packet setting to match. Deserialization stays with the consumer.
 * The board must run HSB IP 0x2602 or newer, for the 12-bit immediate-data
 * slot layout.
 *
 * Placement is transport-owned. Forward uses poll_cpu; Duplex uses rx_cpu
 * and tx_cpu together or neither. -1 means unset: the polling context
 * inherits the process CPU set. The operator does not read HOLOLINK_AFFINITY.
 *
 * A CQ monitor failure ends the run, as a fatal receive-path event does in
 * RoceReceiverOp: the monitor wakes compute(), which fails the run because
 * receive_path_failed() is set. A failure that comes while the operator
 * starts fails its start instead.
 */
class CpuRoceTransceiverOp
    : public BaseReceiverOp<CpuRoceTransceiverOp, CpuRoceTransceiverParams> {
public:
    using BaseReceiverOp<CpuRoceTransceiverOp, CpuRoceTransceiverParams>::BaseReceiverOp;
    ~CpuRoceTransceiverOp() override;

    // BaseReceiverOp virtual hooks
    void start_receiver() override;
    void stop_receiver() override;
    std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>> get_next_frame(
        double timeout_ms, CUstream cuda_stream) override;
    std::tuple<std::string, uint32_t> local_ip_and_port() override;
    bool frames_ready() override;
    bool receive_path_failed() override;

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
    friend struct CpuRoceTransceiverOpTestAccess;
    using Transceiver = hololink::transport::roce::CpuRoceTransceiver;

    void monitor_loop() noexcept;
    // The CQ monitor failed with \p what. Outside teardown, marks the receive path failed, wakes
    // get_next_frame() and posts a notification, so that compute() fails the run.
    void on_monitor_failure(const char* what) noexcept;

    std::shared_ptr<hololink::ReceiverMemoryDescriptor> rx_gpu_memory_;
    int rx_dmabuf_fd_ = -1;
    std::unique_ptr<Transceiver> transceiver_;
    std::unique_ptr<std::thread> monitor_thread_;
    std::mutex monitor_mutex_;
    std::condition_variable monitor_cv_;
    bool hololink_configured_ = false;
    // Guarded by monitor_mutex_.
    bool stopped_ = false;
    // Guarded by monitor_mutex_. Set, with the monitor's error, when the CQ monitor fails outside
    // teardown.
    bool receive_path_failed_ = false;
    std::string receive_path_failure_;
};

} // namespace hololink::operators

#endif // SRC_HOLOLINK_OPERATORS_CPU_ROCE_TRANSCEIVER_OP_HPP
