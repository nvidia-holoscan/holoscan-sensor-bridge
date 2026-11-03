/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SRC_OPERATORS_COMPUTE_CRC
#define SRC_OPERATORS_COMPUTE_CRC

#include <functional>
#include <memory>
#include <type_traits>

#include <capture_metadata_traits.hpp>
#include <hololink/operators/capture_metadata_utils.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "crc_channel.hpp"

#include <cuda.h>
#include <nvcomp/crc32.h>

namespace hololink::operators {

namespace detail {
    // The only thing that differs per payload type: how to reach the tensor the CRC is actually
    // computed over. holoscan::Tensor IS the tensor; a descriptor-schema type like
    // holoscan::schema::ImageT wraps its real payload in its own .data tensor field.
    template <typename PayloadT>
    const holoscan::Tensor& crc_payload_tensor(const PayloadT& value);

    template <>
    inline const holoscan::Tensor& crc_payload_tensor<holoscan::Tensor>(const holoscan::Tensor& value)
    {
        return value;
    }

    template <>
    inline const holoscan::Tensor& crc_payload_tensor<holoscan::schema::ImageT>(const holoscan::schema::ImageT& value)
    {
        return *value.data;
    }
} // namespace detail

/**
 * @class ComputeCrcOp
 * @brief Operator that computes CRC32 checksums on GPU tensors using nvCOMP.
 *
 * Receives a tensor/metadata pair on the "in_tensor"/"in_metadata" ports, starts a CRC32 of the
 * tensor on the GPU using nvcomp, and emits both unchanged on the "tensor"/"metadata" ports.
 * The CRC is handed to the CheckCrcOp that shares its CrcChannel, which waits for it only when
 * it validates that frame, so the GPU CRC overlaps the work between the two operators.
 * With emit_crc=true, compute waits for that frame's result and emits it on "crc" before
 * forwarding the tensor and metadata. This supports validation graphs carrying a completed
 * scalar per frame, without a CheckCrcOp/channel callback side branch.
 *
 * The "tensor" output forwards the received tensor. HSDK5 gives an operator a CUDA stream only
 * when it allocates tensors or receives them from an operator that does, so the tensor must go
 * on to an operator that allocates its own (a converter such as CsiToBayerOp), not straight to
 * CheckCrcOp, which takes only the metadata.
 *
 * @tparam PayloadT The tensor-carrying payload type for the "in_tensor"/"tensor" ports --
 * either holoscan::Tensor (a plain Tensor-domain pipeline) or holoscan::schema::ImageT (an
 * image-domain pipeline). This is a pass-through CRC operator with no opinion on which side
 * of the Tensor-to-ImageT boundary it sits, so it is instantiated for whichever payload type
 * the app actually wires in (see the ComputeCrcTensorOp/ComputeCrcImageOp aliases below).
 *
 * Uses the CRC-32/JAMCRC algorithm (bit-inverted CRC32) which matches the format
 * recorded from HSB.
 */
template <typename PayloadT>
class ComputeCrcOp : public holoscan::Operator<> {
public:
    /**
     * @param crc_channel Shared with the CheckCrcOp that validates this operator's CRCs.
     * @param cuda_device_ordinal The GPU to compute on.
     * @param frame_size Upper bound of the tensor size, to tune the nvcomp kernel.
     */
    // emit_crc completes the GPU read before forwarding and emits a frame-paired scalar,
    // matching the synchronous validation pipeline. The default keeps channel handoff.
    ComputeCrcOp(std::shared_ptr<CrcChannel> crc_channel, int cuda_device_ordinal = 0,
        uint64_t frame_size = 0, bool emit_crc = false);

    void setup(holoscan::OperatorSpec& spec) override;
    [[nodiscard]] holoscan::Contract contract() const override;
    void start() override;
    holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

    holoscan::Input<PayloadT> input_tensor { "in_tensor" };
    holoscan::Input<hololink::CaptureMetadataT> input_metadata { "in_metadata" };
    holoscan::Output<PayloadT> output_tensor { "tensor" };
    holoscan::Output<hololink::CaptureMetadataT> output_metadata { "metadata" };
    holoscan::Output<uint32_t> output_crc { "crc" };

protected:
    holoscan::expected<void, holoscan::Error> compute_impl(holoscan::ExecutionContext& context);

    std::shared_ptr<CrcChannel> crc_channel_;
    int cuda_device_ordinal_;
    uint64_t frame_size_;
    bool emit_crc_;
    bool is_integrated_ = false;
    bool host_memory_warning_ = false;
    nvcompBatchedCRC32Opts_t crc_opts_;
    // Frames dropped because downstream was behind, reported at most once per second.
    hololink::operators::DropCounter dropped_frames_;
    // Frames whose payload and metadata were stamped as different frames, reported at most once
    // per second.
    hololink::operators::DropCounter mismatched_frames_;
};

using ComputeCrcTensorOp = ComputeCrcOp<holoscan::Tensor>;
using ComputeCrcImageOp = ComputeCrcOp<holoscan::schema::ImageT>;

/**
 * @class CheckCrcOp
 * @brief Operator that validates the CRC its ComputeCrcOp computed for each frame.
 *
 * Receives a frame's metadata from anywhere downstream of ComputeCrcOp that keeps the
 * frame_number ComputeCrcOp saw: the converter after it, for instance, unless that converter cuts
 * sub-frame bands, which it renumbers (then take the metadata ahead of the converter). Takes that
 * frame's CRC from the shared CrcChannel (waiting for the GPU to finish it), calls check_crc()
 * with it, and emits the metadata unchanged on the "metadata" port, which may be left
 * unconnected. Derive from it and override check_crc() to validate.
 */
class CheckCrcOp : public holoscan::Operator<> {
public:
    /// @param crc_channel The CrcChannel its ComputeCrcOp was given.
    explicit CheckCrcOp(std::shared_ptr<CrcChannel> crc_channel);
    virtual ~CheckCrcOp() = default;

    void setup(holoscan::OperatorSpec& spec) override;
    [[nodiscard]] holoscan::Contract contract() const override;
    holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

    holoscan::Input<hololink::CaptureMetadataT> input_metadata { "in_metadata" };
    holoscan::Output<hololink::CaptureMetadataT> output_metadata { "metadata" };

    /**
     * @brief Returns the most recently validated CRC value.
     */
    uint32_t get_last_crc() const { return last_computed_crc_; }

    /**
     * @brief Virtual method called with each frame's computed CRC (JAMCRC, as the HSB reports it).
     *
     * Override in derived classes to implement custom CRC validation logic. The default
     * implementation stores the value, accessible via get_last_crc().
     */
    virtual void check_crc(uint32_t computed_crc);

    /**
     * @brief As above, with the frame's metadata (its frame_number, and the HSB's own crc).
     *
     * The default implementation calls check_crc(computed_crc).
     */
    virtual void check_crc(uint32_t computed_crc, const hololink::CaptureMetadataT& metadata);

protected:
    std::shared_ptr<CrcChannel> crc_channel_;
    uint32_t last_computed_crc_ = 0;
    // Frames whose CRC never arrived (dropped before ComputeCrcOp), and frames dropped because
    // downstream was behind, each reported at most once per second.
    hololink::operators::DropCounter missing_crcs_;
    hololink::operators::DropCounter dropped_frames_;
};

/**
 * @brief A CrcChannel together with a check to run for each frame: what a ComputeCrcOp and a
 * CheckCrcOp built from the operator catalog share (Python's hololink.CrcChannel). Python cannot
 * subclass CheckCrcOp, so the catalog's CheckCrcOp calls `check` instead of an override.
 */
struct CrcCheckChannel : std::enable_shared_from_this<CrcCheckChannel> {
    /// Called with the frame's GPU CRC, the HSB's CRC (metadata.crc) and metadata.frame_number.
    using Check = std::function<void(uint32_t computed_crc, int64_t received_crc, int64_t frame_number)>;

    explicit CrcCheckChannel(Check check_crc = {}, size_t max_pending = 16)
        : channel(std::make_shared<CrcChannel>(max_pending))
        , check(std::move(check_crc))
    {
    }

    std::shared_ptr<CrcChannel> channel;
    Check check;
};

/// A CheckCrcOp that runs a CrcCheckChannel's check for each frame it checks.
class CallbackCheckCrcOp : public CheckCrcOp {
public:
    explicit CallbackCheckCrcOp(std::shared_ptr<CrcCheckChannel> crc_channel);

    void check_crc(uint32_t computed_crc, const hololink::CaptureMetadataT& metadata) override;

private:
    std::shared_ptr<CrcCheckChannel> crc_check_;
};

} // namespace hololink::operators

#endif /* SRC_OPERATORS_COMPUTE_CRC */
