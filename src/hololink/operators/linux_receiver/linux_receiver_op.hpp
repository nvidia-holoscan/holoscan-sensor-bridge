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

#ifndef SRC_HOLOLINK_OPERATORS_LINUX_RECEIVER_LINUX_RECEIVER_OP
#define SRC_HOLOLINK_OPERATORS_LINUX_RECEIVER_LINUX_RECEIVER_OP

#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>

#include "hololink/operators/base_receiver_op.hpp"
#include <hololink/core/data_channel.hpp>
#include <holoscan/core/parameter_schema.hpp>

namespace hololink::operators {

class LinuxReceiver;

/**
 * @brief Flat parameter aggregate for LinuxReceiverOp.
 *
 * Contains CUDA-buffer and Linux-socket-specific configuration.
 * Set all fields before passing to LinuxReceiverOp (or use designated initializers).
 */
struct LinuxReceiverParams {
    // ---- Base receiver fields (shared with BaseReceiverOp) ----
    CUcontext frame_context { nullptr };
    size_t frame_size { 0 };
    bool trim { true };
    // Note: CUDA device placement is specified at graph compile time via
    // CompileOptions::bind_tensor_output_device(), not as a per-params field.

    // ---- Linux-specific fields ----
    /// CPU affinity set for the receiver thread.
    /// Empty → auto-detect from HOLOLINK_AFFINITY env-var, fallback to core 2.
    std::vector<int> receiver_affinity {};
    uint32_t pages { 1 }; ///< HSDK4 default.
    uint32_t queue_size { 1 }; ///< HSDK4 default. Must be ≤ pages.
    /// Complete after this many compute() calls (HSDK4: a CountCondition on the receiver): one
    /// per frame, plus, with watch_ingress, one per ingress timeout; 0 or less: no limit.
    int64_t frame_limit { 0 };
    /// Once this operator and every other one that sets frame_limit_ends_run have reached their
    /// frame_limit, end the whole run (ExecutionContext::request_completion()), dropping the frames
    /// still in the pipeline; see hololink::operators::FrameLimitRunGroup for when a run needs it.
    bool frame_limit_ends_run { false };
    /// Also run compute() after each second without a frame, as HSDK4 receivers did with
    /// use_frame_ready_condition=false (the sub-frame players): it logs "Ingress frame timeout;
    /// ignoring." once per stall, and each such run counts toward frame_limit.
    bool watch_ingress { false };
};

// A partial schema: HSDK5 only needs the field BaseReceiverOp::setup() sizes the tensor pool's
// slots from (holoscan::parameter_byte_span()); the other fields stay opaque to it and are still
// bound by the operator catalog. The schema makes frame_size a recompile-required parameter.
HOLOSCAN_PARAM_SCHEMA(
    LinuxReceiverParams,
    holoscan::param_schema("com.nvidia.hololink.operators.LinuxReceiverParams", 1).partial(),
    holoscan::field<&LinuxReceiverParams::frame_size>("frame_size")
        .description("Size of one received frame, in bytes; the tensor pool's slot size."));

/**
 * @brief Operator that receives frames via a Linux UDP socket.
 *
 * Derives from BaseReceiverOp<LinuxReceiverParams>. Runtime dependencies are
 * supplied through the constructor and configuration through LinuxReceiverParams.
 *
 * Outputs (see BaseReceiverOp):
 *  'tensor'   — holoscan::Tensor, a copy of the received frame in a pool slot (1-D uint8,
 *               device memory).
 *  'metadata' — hololink::CaptureMetadataT, stamped with the tensor's capture_time and
 *               frame_id.
 */
class LinuxReceiverOp : public BaseReceiverOp<LinuxReceiverOp, LinuxReceiverParams> {
public:
    using BaseReceiverOp<LinuxReceiverOp, LinuxReceiverParams>::BaseReceiverOp;

    // BaseReceiverOp virtual hooks
    void start_receiver() override;
    void stop_receiver() override;
    std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>> get_next_frame(double timeout_ms, CUstream cuda_stream) override;
    bool frames_ready() override;
    bool watch_ingress() const override { return params().watch_ingress; }

    /**
     * @brief Override the CPU affinity for the receiver thread.
     *
     * Takes effect on the next start().  When set, this overrides the
     * receiver_affinity field in LinuxReceiverParams.
     *
     * @param affinity  List of CPU indices to pin the receiver thread to.
     */
    void set_receiver_affinity(const std::vector<int>& affinity);

private:
    // Optional CPU affinity override; set via set_receiver_affinity().
    std::optional<std::vector<int>> affinity_override_;

    // Effective receiver affinity resolved in start_receiver(), used in run().
    std::vector<int> effective_affinity_;

    std::shared_ptr<LinuxReceiver> receiver_;
    std::unique_ptr<std::thread> receiver_thread_;
    // shared_ptr so the allocator can later hand pool slots directly backed by
    // this RDMA-registered memory (zero-copy path, pending HSDK5 custom allocator API).
    std::shared_ptr<hololink::ReceiverMemoryDescriptor> frame_memory_;

    void run();
    void check_buffer_size(size_t data_memory_size);
    uint64_t received_address_offset();
};

} // namespace hololink::operators

#endif /* SRC_HOLOLINK_OPERATORS_LINUX_RECEIVER_LINUX_RECEIVER_OP */
