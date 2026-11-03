/*
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

#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>

#include <capture_metadata_traits.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/operators/capture_metadata_utils.hpp>

#include <cuda.h>

namespace hololink::operators {

// Params for HSDK5 graph.op<>() construction pattern.
struct SubFrameCombinerParams {
    int cuda_device_ordinal = 0;
    // Rows of the full frame the sub-frames are assembled into (the camera height). Required.
    uint32_t height = 0;
    // What to do with a frame missing a band, or holding a band marked degraded (part of its data
    // lost upstream; see CsiToBayerParams::drop_incomplete). By default it is still emitted, the
    // missing rows holding an earlier frame's content and the sample marked
    // holoscan::SampleFlags::kDegraded -- as HSDK4 emitted partial frames, unmarked. When set, it
    // is discarded instead.
    bool drop_incomplete = false;
};

/**
 * @brief Operator class to combine sub-frames into a single frame.
 *
 * Each input sample is one horizontal band ([rows, W, C]) of a frame, placed at the row its
 * CaptureMetadataT::sub_frame_offset names. The bands of a frame are assembled in a device buffer
 * this operator owns: an output-pool loan is only valid within the compute() that allocated it,
 * while a frame spans one compute() per band. Once the frame's last row has arrived, the frame is
 * copied into a runtime-pool tensor and emitted as [height, W, C], together with the first band's
 * metadata (sub_frame_offset and sub_frame_rows reset to 0). A frame missing a band is still
 * emitted, flagged kDegraded, unless SubFrameCombinerParams::drop_incomplete discards it.
 */
class SubFrameCombinerOp : public holoscan::Operator<SubFrameCombinerParams> {
public:
    void setup(holoscan::OperatorSpec& spec) override;
    [[nodiscard]] holoscan::Contract contract() const override;
    void start() override;
    void stop() override;
    holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

    holoscan::Input<holoscan::schema::ImageT> input_tensor { "in_tensor" };
    holoscan::Input<hololink::CaptureMetadataT> input_metadata { "in_metadata" };
    holoscan::Output<holoscan::schema::ImageT> output_tensor { "tensor" };
    holoscan::Output<hololink::CaptureMetadataT> output_metadata { "metadata" };

private:
    // Frames whose payload and metadata were stamped as different frames, reported at most once
    // per second.
    hololink::operators::DropCounter mismatched_frames_;
    holoscan::expected<void, holoscan::Error> emit_frame(holoscan::ExecutionContext& context);
    holoscan::expected<void, holoscan::Error> compute_impl(holoscan::ExecutionContext& context);
    void reset_pending_frame();
    void count_dropped_frame(const char* reason);

    int cuda_device_ordinal_ = 0;
    uint32_t height_ = 0;

    CUcontext cuda_context_ = nullptr;
    CUdevice cuda_device_ = 0;

    // The frame being assembled: height_ rows of row_bytes_, densely packed.
    hololink::common::UniqueCUdeviceptr frame_memory_;
    size_t frame_memory_size_ = 0;

    // Set by the pending frame's first sub-frame; every later sub-frame of the frame must match.
    std::array<int64_t, 3> pending_shape_ {};
    DLDataType pending_dtype_ {};
    size_t row_bytes_ = 0;
    std::optional<hololink::CaptureMetadataT> pending_output_metadata_;
    // The first sub-frame's ImageT descriptor fields (encoding/color_space/etc.) minus its
    // tensor -- every sub-frame of one frame shares the same geometry/encoding; only the height
    // differs from the combined frame's.
    std::optional<holoscan::schema::ImageT> pending_image_descriptor_;
    // The first sub-frame's ORIGINAL (propagated, not re-minted) sample metadata -- its
    // capture_time is reused for the combined frame's own emit, since the combined frame's true
    // capture instant is when its first sub-frame arrived, not whenever accumulation finishes.
    std::optional<holoscan::SampleMetadata> pending_sample_metadata_;

    int64_t expected_frame_number_ = 0;
    uint64_t received_rows_ = 0;
    /// A sub-frame of the pending frame was skipped (non-contiguous offset): its rows hold a
    /// previous frame's content, so the frame is incomplete.
    bool frame_has_gap_ = false;
    /// Union of the pending frame's bands' sample flags (e.g. kDegraded).
    holoscan::SampleFlags pending_flags_ = holoscan::SampleFlags::kNone;
    bool drop_incomplete_ = false;

    // Assembled frames dropped because the downstream consumer was behind (output pool empty or
    // queue full), reported at most once per second.
    uint32_t dropped_frames_ { 0 };
    std::chrono::steady_clock::time_point drop_report_time_ {};
};

} // namespace hololink::operators
