/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SRC_HOLOLINK_OPERATORS_LINUX_COE_RECEIVER_LINUX_COE_RECEIVER_OP
#define SRC_HOLOLINK_OPERATORS_LINUX_COE_RECEIVER_LINUX_COE_RECEIVER_OP

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>

#include "hololink/operators/base_receiver_op.hpp"
#include <hololink/core/data_channel.hpp>
#include <holoscan/core/parameter_schema.hpp>

namespace hololink::operators {

class LinuxCoeReceiver;

/**
 * @brief Flat parameter aggregate for LinuxCoeReceiverOp.
 *
 * Contains CUDA-buffer and COE (IEEE 1722/AVTP)-specific configuration.
 * Set all fields before passing to LinuxCoeReceiverOp (or use designated initializers).
 */
struct LinuxCoeReceiverParams {
    // ---- Base receiver fields (shared with BaseReceiverOp) ----
    CUcontext frame_context { nullptr };
    size_t frame_size { 0 };
    bool trim { true };
    // Note: CUDA device placement is specified at graph compile time via
    // CompileOptions::bind_tensor_output_device(), not as a per-params field.

    // ---- COE-specific fields ----
    /// CPU affinity set for the receiver thread.
    /// Empty → auto-detect from HOLOLINK_AFFINITY env-var, fallback to core 2.
    std::vector<int> receiver_affinity {};
    /// Network interface the raw AF_PACKET/AVTP socket binds to (e.g. "eth0").
    std::string coe_interface {};
    /// Sensor frame width in pixels — passed through to DataChannel::configure_coe(), which
    /// needs it to program the FPGA's COE packetizer.
    uint32_t pixel_width { 0 };
    /// Each camera sharing one COE network connection must use a unique channel number
    /// (0..63); matches DataChannel::configure_coe()'s own channel parameter width.
    uint8_t coe_channel { 0 };
    /// Complete after this many frames (compute() calls), as HSDK4's CountCondition on the
    /// receiver did; 0 or less: no limit.
    int64_t frame_limit { 0 };
    /// Once this operator and every other one that sets frame_limit_ends_run have reached their
    /// frame_limit, end the whole run (ExecutionContext::request_completion()), dropping the frames
    /// still in the pipeline; see hololink::operators::FrameLimitRunGroup for when a run needs it.
    bool frame_limit_ends_run { false };
    /// Fail the run after this many milliseconds without a frame, counted from the device start
    /// and then from each frame; 0 disables it. The run ends kFailed with a diagnostic naming
    /// the stall, as with FusaCoeCaptureParams::stall_timeout_ms. This replaces HSDK4's
    /// capture_stalled()/capture_stall_duration_ms(), which the callers that set a stall
    /// timeout checked after the run.
    uint32_t stall_timeout_ms { 0 };
};

// A partial schema: HSDK5 only needs the field BaseReceiverOp::setup() sizes the tensor pool's
// slots from (holoscan::parameter_byte_span()); the other fields stay opaque to it and are still
// bound by the operator catalog. The schema makes frame_size a recompile-required parameter.
HOLOSCAN_PARAM_SCHEMA(
    LinuxCoeReceiverParams,
    holoscan::param_schema("com.nvidia.hololink.operators.LinuxCoeReceiverParams", 1).partial(),
    holoscan::field<&LinuxCoeReceiverParams::frame_size>("frame_size")
        .description("Size of one received frame, in bytes; the tensor pool's slot size."));

/**
 * @brief Operator that receives frames via IEEE 1722/AVTP Class-Oriented-Encapsulation (COE)
 * packets, read directly off a raw AF_PACKET socket (not a UDP/RoCE transport like
 * LinuxReceiverOp/RoceReceiverOp).
 *
 * Derives from BaseReceiverOp<LinuxCoeReceiverOp, LinuxCoeReceiverParams>. Runtime
 * dependencies are supplied through the constructor and configuration through
 * LinuxCoeReceiverParams.
 *
 * Output:
 *  'output' — holoscan::Tensor wrapping the received CUDA device buffer.
 */
class LinuxCoeReceiverOp : public BaseReceiverOp<LinuxCoeReceiverOp, LinuxCoeReceiverParams> {
public:
    using BaseReceiverOp<LinuxCoeReceiverOp, LinuxCoeReceiverParams>::BaseReceiverOp;
    ~LinuxCoeReceiverOp() override;

    // Start and stop the stall monitor (LinuxCoeReceiverParams::stall_timeout_ms) around
    // BaseReceiverOp's own start() and stop().
    void start() override;
    void stop() override;
    // BaseReceiverOp::compute(), which fails the run once the stall monitor has declared a stall,
    // also in the activation that was waiting for a frame when it did.
    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
        holoscan::ExecutionContext& context) override;

    // BaseReceiverOp virtual hooks
    void create_data_socket() override;
    // BaseReceiverOp::timeout()'s log, except after a stall, which compute() reports instead.
    void timeout(holoscan::ExecutionContext& context) override;
    void start_receiver() override;
    void stop_receiver() override;
    std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>> get_next_frame(double timeout_ms, CUstream cuda_stream) override;
    bool frames_ready() override;

    /**
     * @brief Override the CPU affinity for the receiver thread.
     *
     * Takes effect on the next start().  When set, this overrides the
     * receiver_affinity field in LinuxCoeReceiverParams.
     *
     * @param affinity  List of CPU indices to pin the receiver thread to.
     */
    void set_receiver_affinity(const std::vector<int>& affinity);

private:
    // Optional CPU affinity override; set via set_receiver_affinity().
    std::optional<std::vector<int>> affinity_override_;

    // Effective receiver affinity resolved in start_receiver(), used in run().
    std::vector<int> effective_affinity_;

    std::shared_ptr<LinuxCoeReceiver> receiver_;
    std::unique_ptr<std::thread> receiver_thread_;
    std::shared_ptr<hololink::ReceiverMemoryDescriptor> frame_memory_;

    void run();

    // Stall detection: the receiver thread stamps last_frame_time_ on every frame, and
    // stall_monitor_thread_ declares a stall once a whole stall_timeout_ms passes without one.
    // It then wakes compute(), the only place that can end the run.
    std::thread stall_monitor_thread_;
    std::mutex stall_mutex_;
    std::condition_variable stall_cv_;
    // Read from start(): params() is only for the runtime's callbacks, not for these threads.
    std::chrono::milliseconds stall_timeout_ {}; // guarded by stall_mutex_
    bool stall_monitor_running_ = false; // guarded by stall_mutex_
    std::chrono::steady_clock::time_point last_frame_time_ {}; // guarded by stall_mutex_
    uint64_t frames_received_ = 0; // since the monitor started; guarded by stall_mutex_
    std::atomic<bool> stalled_ { false };
    std::atomic<int64_t> stall_duration_ms_ { 0 };
    // compute() calls since start(), as the frame limit's ComputeCalls counts them.
    uint64_t compute_calls_ = 0;

    void start_stall_monitor();
    void stop_stall_monitor();
    void frame_arrived();
    void watch_for_stall();
    [[nodiscard]] holoscan::Error stall_error() const;
};

} // namespace hololink::operators

#endif /* SRC_HOLOLINK_OPERATORS_LINUX_COE_RECEIVER_LINUX_COE_RECEIVER_OP */
