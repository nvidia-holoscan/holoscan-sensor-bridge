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

#include <cuda.h>

// Forward-declare to keep this header free of DOCA/ibverbs dependencies.
// The full definition is included only in gpu_roce_transceiver_op.cpp.
namespace hololink::operators {
class GpuRoceTransceiver;
}

#include "hololink/operators/base_receiver_op.hpp"
#include <hololink/core/data_channel.hpp>
#include <holoscan/core/parameter_schema.hpp>

namespace hololink::operators {

/**
 * @brief Flat parameter aggregate for GpuRoceTransceiverOp.
 *
 * Contains CUDA-buffer and GPU-RoCE-specific configuration.
 * Set all fields before passing to GpuRoceTransceiverOp (or use designated initializers).
 */
struct GpuRoceTransceiverParams {
    // ---- Base receiver fields (shared with BaseReceiverOp) ----
    CUcontext frame_context { nullptr };
    size_t frame_size { 0 };
    bool trim { false }; ///< Not used: GPU kernel owns the data path.
    // Note: CUDA device placement is specified at graph compile time via
    // CompileOptions::bind_tensor_output_device(), not as a per-params field.

    // ---- GPU RoCE transceiver specific ----
    /// IBV device to use. Empty: the one on the network interface that reaches the board
    /// (DataChannel::infiniband_device()), with its own port instead of ibv_port.
    std::string ibv_name {};
    uint32_t ibv_port { 1 };
    /// Remote QP numbers for Tx, comma-separated, one per DataChannel (e.g. "2,4,6,8").
    std::string tx_ibv_qp { "2" };
    /// DataChannels, one per QP of tx_ibv_qp in the same order, all on the same board. Empty: the
    /// constructor's hololink_channel is the only one.
    std::vector<std::shared_ptr<DataChannel>> hololink_channels {};
    uint32_t gpu_id { 0 }; ///< CUDA device ID.
    uint32_t forward { 0 }; ///< Enable forward mode (1 = on).
    /// Has no effect for this operator: BaseReceiverOp::contract() completes a receiver after
    /// this many compute() calls, and this one never runs compute() (its GPU kernel owns the
    /// data path), so it never reaches a limit. Leave it 0.
    int64_t frame_limit { 0 };
    /// Has no effect for this operator, as frame_limit.
    bool frame_limit_ends_run { false };
};

// A partial schema: HSDK5 only needs the field BaseReceiverOp::setup() sizes the tensor pool's
// slots from (holoscan::parameter_byte_span()); the other fields stay opaque to it and are still
// bound by the operator catalog. The schema makes frame_size a recompile-required parameter.
HOLOSCAN_PARAM_SCHEMA(
    GpuRoceTransceiverParams,
    holoscan::param_schema("com.nvidia.hololink.operators.GpuRoceTransceiverParams", 1).partial(),
    holoscan::field<&GpuRoceTransceiverParams::frame_size>("frame_size")
        .description("Size of one received frame, in bytes; the tensor pool's slot size."));

/**
 * @brief GPU RoCE transceiver operator.
 *
 * Uses DOCA Verbs for GPU-based CQ/QP management instead of the traditional
 * CPU-side ibverbs approach.  The GPU persistent kernel owns the data path;
 * get_next_frame() always returns {} so the base class never emits a tensor or
 * metadata sample.  Downstream operators access frame data directly via
 * get_ring_addr(), get_ring_stride_sz(), get_ring_stride_num(), and
 * get_ring_flag_addr().
 *
 * A single operator can serve several sensors of the same Hololink board:
 * GpuRoceTransceiverParams::tx_ibv_qp lists one remote QP per sensor ("2,4,6,8") and
 * GpuRoceTransceiverParams::hololink_channels lists the DataChannel of each sensor in the
 * same order. Each pair gets its own local QP and ring buffer. Without hololink_channels,
 * the constructor's hololink_channel is the only sensor.
 */
class GpuRoceTransceiverOp
    : public BaseReceiverOp<GpuRoceTransceiverOp, GpuRoceTransceiverParams> {
public:
    using BaseReceiverOp<GpuRoceTransceiverOp, GpuRoceTransceiverParams>::BaseReceiverOp;

    // BaseReceiverOp virtual hooks
    void start_receiver() override;
    void stop_receiver() override;
    std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>> get_next_frame(double timeout_ms, CUstream cuda_stream) override;
    std::tuple<std::string, uint32_t> local_ip_and_port() override;
    bool frames_ready() override;

    // Direct data-access API for downstream operators.
    CUdeviceptr frame_memory(unsigned sensor_id);

    uint8_t* get_ring_addr(bool is_rx, unsigned sensor_id);
    size_t get_ring_stride_sz(bool is_rx, unsigned sensor_id);
    uint32_t get_ring_stride_num(bool is_rx, unsigned sensor_id);
    uint64_t* get_ring_flag_addr(bool is_rx, unsigned sensor_id);

private:
    std::shared_ptr<GpuRoceTransceiver> gpu_roce_transceiver_;
    std::unique_ptr<std::thread> gpu_monitor_thread_;
    // Channels whose HSB RoCE configuration has been (at least partially) written, to unconfigure
    // on stop.
    std::vector<std::shared_ptr<DataChannel>> configured_channels_;

    void gpu_monitor_loop();
    // The DataChannel of each QP in tx_ibv_qp: hololink_channels, or hololink_channel alone.
    std::vector<std::shared_ptr<DataChannel>> data_channels();
    static constexpr unsigned PAGES = 1024;
};

} // namespace hololink::operators

#endif // SRC_HOLOLINK_OPERATORS_GPU_ROCE_TRANSCEIVER_OP_HPP
