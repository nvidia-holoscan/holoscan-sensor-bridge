/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef SRC_HOLOLINK_OPERATORS_ROCE_RECEIVER_ROCE_RECEIVER_OP
#define SRC_HOLOLINK_OPERATORS_ROCE_RECEIVER_ROCE_RECEIVER_OP

#include <memory>
#include <string>
#include <thread>

#include <cuda.h>

#include "hololink/operators/base_receiver_op.hpp"
#include <hololink/core/data_channel.hpp>
#include <holoscan/core/parameter_schema.hpp>

namespace hololink::operators {

class RoceReceiver;

/**
 * @brief Flat parameter aggregate for RoceReceiverOp.
 *
 * Contains CUDA-buffer and RoCE-specific configuration.
 * Set all fields before passing to RoceReceiverOp (or use designated initializers).
 */
struct RoceReceiverParams {
    // ---- Base receiver fields (shared with BaseReceiverOp) ----
    CUcontext frame_context { nullptr };
    size_t frame_size { 0 };
    bool trim { true };
    // Note: CUDA device placement is specified at graph compile time via
    // CompileOptions::bind_tensor_output_device(), not as a per-params field.

    // ---- RoCE-specific fields ----
    /// IBV device to receive with. Empty: the one on the network interface that reaches the
    /// board (DataChannel::infiniband_device()), with its own port instead of ibv_port.
    std::string ibv_name {};
    uint32_t ibv_port { 1 };
    uint32_t pages { 2 }; ///< HSDK4 default.
    uint32_t queue_size { 1 }; ///< HSDK4 default. Must be ≤ pages.
    size_t metadata_offset { 0 }; ///< 0 → auto-compute from frame_size.
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
    RoceReceiverParams,
    holoscan::param_schema("com.nvidia.hololink.operators.RoceReceiverParams", 1).partial(),
    holoscan::field<&RoceReceiverParams::frame_size>("frame_size")
        .description("Size of one received frame, in bytes; the tensor pool's slot size."));

/**
 * @brief Operator that receives frames over an InfiniBand/RoCE NIC.
 *
 * Derives from BaseReceiverOp<RoceReceiverParams>. Runtime dependencies are
 * supplied through the constructor and configuration through RoceReceiverParams.
 *
 * Outputs (see BaseReceiverOp):
 *  'tensor'   — holoscan::Tensor, a copy of the received frame in a pool slot (1-D uint8,
 *               device memory).
 *  'metadata' — hololink::CaptureMetadataT, stamped with the tensor's capture_time and
 *               frame_id.
 */
class RoceReceiverOp : public BaseReceiverOp<RoceReceiverOp, RoceReceiverParams> {
public:
    using BaseReceiverOp<RoceReceiverOp, RoceReceiverParams>::BaseReceiverOp;

    // BaseReceiverOp virtual hooks
    void start_receiver() override;
    void stop_receiver() override;
    std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>> get_next_frame(double timeout_ms, CUstream cuda_stream) override;
    std::tuple<std::string, uint32_t> local_ip_and_port() override;
    bool frames_ready() override;
    bool watch_ingress() const override { return params().watch_ingress; }
    bool receive_path_failed() override;

private:
    std::shared_ptr<RoceReceiver> receiver_;
    std::unique_ptr<std::thread> receiver_thread_;
    // shared_ptr so the allocator can later hand pool slots directly backed by
    // this RDMA-registered memory (zero-copy path, pending HSDK5 custom allocator API).
    std::shared_ptr<hololink::ReceiverMemoryDescriptor> frame_memory_;

    void run();
};

} // namespace hololink::operators

#endif /* SRC_HOLOLINK_OPERATORS_ROCE_RECEIVER_ROCE_RECEIVER_OP */
