/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_OPERATORS_COMPUTE_CRC_OP_HPP
#define HOLOLINK_MODULE_OPERATORS_COMPUTE_CRC_OP_HPP

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <queue>
#include <string>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/parameter.hpp>

#include <cuda.h>
#include <nvcomp/crc32.h>

namespace hololink::module::operators {

/* Native module frame-CRC operator.
 *
 * Receives a tensor on "input", computes a CRC32 checksum over the frame
 * bytes on the GPU via nvcomp, and re-emits the tensor unchanged on
 * "output". The computed CRC is retrieved via get_computed_crc() — usually
 * by a paired CheckCrcOp wired downstream. Uses CRC-32/JAMCRC
 * (bit-inverted CRC32), matching the format HSB records.
 *
 * Params:
 *   cuda_device_ordinal — CUDA device for the computation (default 0)
 *   frame_size          — upper bound on the region size, used to tune the
 *                         nvcomp kernel. The actual CRC block size always
 *                         comes from the received tensor. */
class ComputeCrcOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(ComputeCrcOp);

    void start() override;
    void stop() override;
    void setup(holoscan::OperatorSpec& spec) override;
    void compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
        holoscan::ExecutionContext&) override;

    /* Synchronizes on the oldest pending frame's CRC event and returns its
     * CRC32 (JAMCRC format) for the paired CheckCrcOp. The synchronization is
     * deferred to here (off the compute() path) so the GPU CRC overlaps the
     * pipeline work between ComputeCrcOp and this call — the reason the two
     * operators are split. Each pending frame owns its event and readback
     * buffer, so the value stays tied to its frame even when the next frame's
     * compute() runs concurrently under a multi-threaded scheduler. */
    uint32_t get_computed_crc();

protected:
    holoscan::Parameter<int> cuda_device_ordinal_;
    holoscan::Parameter<uint64_t> frame_size_;

    bool is_integrated_ = false;
    bool host_memory_warning_ = false;
    nvcompBatchedCRC32Opts_t crc_opts_;

private:
    // One in-flight CRC computation. Each frame gets its own slot so the next
    // frame's compute() never reuses an event or readback buffer that a still-
    // pending get_computed_crc() depends on. This is what lets the GPU CRC
    // synchronization stay deferred to get_computed_crc() (the reason
    // ComputeCrcOp and CheckCrcOp are separate) without racing across frames.
    struct CrcSlot {
        cudaEvent_t event = 0;
        size_t* message_size_dptr = nullptr;
        void** message_dptr = nullptr;
        uint32_t* result_dptr = nullptr;
        uint32_t computed_crc = 0;
    };

    // Returns a slot ready for a new computation: a recycled one if available,
    // otherwise a freshly allocated one. Caller must already have selected the
    // CUDA device (compute() does).
    CrcSlot* acquire_slot();

    // slot_storage_ owns every slot (std::deque keeps element addresses stable
    // as it grows); free_slots_ holds reusable slots and pending_slots_ holds
    // those awaiting a get_computed_crc(). slots_mutex_ guards all three.
    std::deque<CrcSlot> slot_storage_;
    std::queue<CrcSlot*> free_slots_;
    std::queue<CrcSlot*> pending_slots_;
    std::mutex slots_mutex_;
};

/* Native module CRC-check operator, paired with ComputeCrcOp.
 *
 * Copies its "input" tensor to "output" so it can be inserted anywhere in
 * a pipeline, and fetches the CRC computed by the linked ComputeCrcOp
 * (stalling on a CUDA stream sync if needed), passing it to the virtual
 * check_crc(). The default check_crc() stores the value in pipeline
 * metadata under computed_crc_metadata_name (default "computed_crc").
 * Subclass and override check_crc() for custom validation.
 *
 * Params:
 *   compute_crc_op              — the ComputeCrcOp that computed the CRC
 *   computed_crc_metadata_name  — metadata key for the stored CRC */
class CheckCrcOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(CheckCrcOp);

    virtual ~CheckCrcOp() = default;

    void setup(holoscan::OperatorSpec& spec) override;
    void compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
        holoscan::ExecutionContext&) override;

    virtual void check_crc(uint32_t computed_crc);

protected:
    holoscan::Parameter<std::shared_ptr<ComputeCrcOp>> compute_crc_op_;
    holoscan::Parameter<std::string> computed_crc_metadata_name_;
};

} // namespace hololink::module::operators

#endif // HOLOLINK_MODULE_OPERATORS_COMPUTE_CRC_OP_HPP
