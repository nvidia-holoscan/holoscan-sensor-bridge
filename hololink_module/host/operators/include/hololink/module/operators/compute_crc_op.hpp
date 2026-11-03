/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_OPERATORS_COMPUTE_CRC_OP_HPP
#define HOLOLINK_MODULE_OPERATORS_COMPUTE_CRC_OP_HPP

#include <cstdint>
#include <memory>

#include <cuda.h>
#include <nvcomp/crc32.h>

#include <capture_metadata_traits.hpp>
#include <hololink/operators/capture_metadata_utils.hpp> // DropCounter
#include <hololink/operators/compute_crc/crc_channel.hpp>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/temporal_contract.hpp>

namespace hololink::module::operators {

/**
 * @brief Configuration parameters for ComputeCrcOp.
 */
struct ComputeCrcParams {
    int cuda_device_ordinal = 0;
    /// Upper bound on the region size, used to tune the nvcomp kernel heuristic. The actual
    /// CRC block size always comes from the received tensor.
    uint64_t frame_size = 0;
};

HOLOSCAN_PARAM_SCHEMA(
    ComputeCrcParams,
    holoscan::param_schema("com.nvidia.hololink.module.operators.ComputeCrcParams", 1),
    holoscan::field<&ComputeCrcParams::cuda_device_ordinal>("cuda_device_ordinal")
        .description("CUDA device on which to run the CRC32 kernel."),
    holoscan::field<&ComputeCrcParams::frame_size>("frame_size")
        .description("Upper bound on the region size, for the nvcomp kernel heuristic; the "
                     "actual CRC block size always comes from the received tensor."));

/* Starts a CRC32 (JAMCRC, as the HSB reports it) of each received tensor on the GPU with nvcomp
 * and passes the tensor/metadata pair on unchanged. The CRC goes to the CheckCrcOp that shares
 * its hololink::operators::CrcChannel, which waits for it only when it validates that frame, so
 * the GPU CRC overlaps the pipeline work between the two operators -- the reason they are split.
 * Pass the same channel to both graph.op<>() calls; default-constructed, start() fails.
 * The "tensor" output forwards the received tensor. HSDK5 gives an operator a CUDA stream only
 * when it allocates tensors or receives them from an operator that does, so the tensor must go on
 * to an operator that allocates its own (a converter such as CsiToBayerOp), not straight to
 * CheckCrcOp, which takes only the metadata. */
class ComputeCrcOp : public holoscan::Operator<ComputeCrcParams> {
public:
    ComputeCrcOp() = default;
    explicit ComputeCrcOp(std::shared_ptr<hololink::operators::CrcChannel> crc_channel)
        : crc_channel_(std::move(crc_channel))
    {
    }

    holoscan::Input<holoscan::Tensor> input_tensor { "in_tensor" };
    holoscan::Input<hololink::CaptureMetadataT> input_metadata { "in_metadata" };
    holoscan::Output<holoscan::Tensor> output_tensor { "tensor" };
    holoscan::Output<hololink::CaptureMetadataT> output_metadata { "metadata" };

    void setup(holoscan::OperatorSpec& spec) override;
    [[nodiscard]] holoscan::Contract contract() const override;
    void start() override;
    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
        holoscan::ExecutionContext& context) override;

private:
    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute_impl(
        holoscan::ExecutionContext& context);

    std::shared_ptr<hololink::operators::CrcChannel> crc_channel_;
    int cuda_device_ordinal_ = 0;
    uint64_t frame_size_ = 0;
    bool is_integrated_ = false;
    bool host_memory_warning_ = false;
    nvcompBatchedCRC32Opts_t crc_opts_ {};
    // Frames dropped because downstream was behind, reported at most once per second.
    hololink::operators::DropCounter dropped_frames_;
    // Frames whose payload and metadata were stamped as different frames, reported at most once
    // per second.
    hololink::operators::DropCounter mismatched_frames_;
};

/* Receives a frame's metadata from anywhere downstream of ComputeCrcOp that keeps the
 * frame_number ComputeCrcOp saw: the converter after it, for instance, unless that converter cuts
 * sub-frame bands, which it renumbers (then take the metadata ahead of the converter). Takes that
 * frame's CRC from the CrcChannel its ComputeCrcOp was given (waiting for the GPU to finish it),
 * calls check_crc() with it, and passes the metadata on unchanged on "metadata", which may be
 * left unconnected. Derive from it and override check_crc() to validate. */
class CheckCrcOp : public holoscan::Operator<> {
public:
    explicit CheckCrcOp(std::shared_ptr<hololink::operators::CrcChannel> crc_channel)
        : crc_channel_(std::move(crc_channel))
    {
    }
    virtual ~CheckCrcOp() = default;

    holoscan::Input<hololink::CaptureMetadataT> input_metadata { "in_metadata" };
    holoscan::Output<hololink::CaptureMetadataT> output_metadata { "metadata" };

    void setup(holoscan::OperatorSpec& spec) override;
    [[nodiscard]] holoscan::Contract contract() const override;
    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
        holoscan::ExecutionContext& context) override;

    [[nodiscard]] uint32_t get_last_crc() const { return last_computed_crc_; }
    // Called with each frame's computed CRC; the default stores it for get_last_crc().
    virtual void check_crc(uint32_t computed_crc);
    // As above, with the frame's metadata (its frame_number, and the HSB's own crc). The default
    // calls check_crc(computed_crc).
    virtual void check_crc(uint32_t computed_crc, const hololink::CaptureMetadataT& metadata);

private:
    std::shared_ptr<hololink::operators::CrcChannel> crc_channel_;
    uint32_t last_computed_crc_ = 0;
    // Each reported at most once per second.
    hololink::operators::DropCounter missing_crcs_;
    hololink::operators::DropCounter dropped_frames_;
};

} // namespace hololink::module::operators

#endif // HOLOLINK_MODULE_OPERATORS_COMPUTE_CRC_OP_HPP
