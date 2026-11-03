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

#ifndef SRC_OPERATORS_CSI_TO_BAYER_CSI_TO_BAYER
#define SRC_OPERATORS_CSI_TO_BAYER_CSI_TO_BAYER

#include "hololink/common/cuda_helper.hpp"
#include <chrono>
#include <memory>
#include <optional>

#include <capture_metadata_traits.hpp>
#include <hololink/core/csi_controller.hpp>
#include <hololink/core/csi_formats.hpp>
#include <hololink/operators/capture_metadata_utils.hpp>
#include <hololink/operators/csi_geometry_schema.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include <cuda.h>

#include "sub_frame_bands.hpp"

namespace hololink::common {

class CudaFunctionLauncher;

} // namespace hololink::common

namespace hololink::operators {

// Params for HSDK5 graph.op<>() construction pattern.
struct CsiToBayerParams {
    int cuda_device_ordinal = 0;
    // 0: whole frames. Otherwise the rows of each sub-frame band: even, and a divisor of the
    // frame height; the receiver's frame_size must then be bytes_per_line * sub_frame_rows.
    uint32_t sub_frame_rows = 0;
    // CSI geometry — filled by camera->configure_converter(params.geometry) before set_params()
    hololink::csi::CsiGeometry geometry;
    // CFA phase of the reconstructed Bayer mosaic -- not derivable from geometry, so the camera
    // driver's own camera->get_bayer_format() must be threaded through here explicitly. Stamped
    // onto every emitted schema::ImageT's encoding field (see bayer_image_encoding()); unset
    // stamps CUSTOM, so a downstream demosaic uses its own grid position rather than a guess.
    std::optional<hololink::csi::BayerFormat> bayer_format;
    // Whether the non-sub-frame input tensor still includes the start_byte_ header region.
    // BaseReceiverOp-derived receivers (LinuxReceiverOp/RoceReceiverOp) emit the full raw
    // buffer including it (true, the default -- matches every receiver except the one below);
    // FusaCoeCaptureOp already strips it before emitting its tensor (see fusa_coe_capture.cpp's
    // `src = buffer.cuda_device_ptr + core_.start_byte()`), so an app wiring CsiToBayerOp
    // downstream of FusaCoeCaptureOp must set this to false. This is a fixed fact about which
    // producer feeds a given CsiToBayerOp instance, known at graph-compose time -- not something
    // to infer per frame from the received buffer's size, which a receiver's own trim-to-
    // bytes_written behavior (see BaseReceiverOp::compute()) can make ambiguous on an ordinary
    // dropped-packet frame.
    bool header_included = true;
    // What to do with a frame (in sub-frame mode, a band) part of whose data was lost upstream:
    // a holed chunk, or a full frame the HSB wrote only part of. By default it is still emitted,
    // with the lost rows stale (sub-frame mode) or zeroed (full frames) and the sample marked
    // holoscan::SampleFlags::kDegraded -- as HSDK4 passed such frames on, unmarked. A band whose
    // last chunk was dropped, or was too short to finish it, is never emitted; the band after it
    // goes out marked degraded. When set, a degraded frame or band is withheld instead (and
    // SubFrameCombinerOp then lacks a band of its frame).
    bool drop_incomplete = false;
};

HOLOSCAN_PARAM_SCHEMA(
    CsiToBayerParams,
    holoscan::param_schema("com.nvidia.hololink.operators.CsiToBayerParams", 1),
    holoscan::field<&CsiToBayerParams::cuda_device_ordinal>("cuda_device_ordinal")
        .description("CUDA device on which to run the CSI-to-Bayer conversion kernel."),
    holoscan::field<&CsiToBayerParams::sub_frame_rows>("sub_frame_rows")
        .description("Row count of one sub-frame activation, or 0 for whole-frame mode."),
    holoscan::field<&CsiToBayerParams::geometry>("geometry")
        .template nested<hololink::csi::CsiGeometry>()
        .description("CSI frame geometry, filled by the camera driver's configure_converter()."),
    holoscan::field<&CsiToBayerParams::bayer_format>("bayer_format")
        .enum_values({ holoscan::param_enum_value("bggr", hololink::csi::BayerFormat::BGGR),
            holoscan::param_enum_value("rggb", hololink::csi::BayerFormat::RGGB),
            holoscan::param_enum_value("gbrg", hololink::csi::BayerFormat::GBRG),
            holoscan::param_enum_value("grbg", hololink::csi::BayerFormat::GRBG) })
        .description("CFA phase of the reconstructed Bayer mosaic's top-left 2x2 block."),
    holoscan::field<&CsiToBayerParams::header_included>("header_included")
        .description("Whether the input tensor still includes the start_byte_ header region."),
    holoscan::field<&CsiToBayerParams::drop_incomplete>("drop_incomplete")
        .description("Withhold a band or frame with lost data instead of passing it on marked "
                     "degraded."));

class CsiToBayerOp : public holoscan::Operator<CsiToBayerParams>, public hololink::csi::CsiConverter {
public:
    // Port declarations — stable public members registered in setup()
    holoscan::Input<holoscan::Tensor> input_tensor { "in_tensor" };
    holoscan::Input<hololink::CaptureMetadataT> input_metadata { "in_metadata" };
    holoscan::Output<holoscan::schema::ImageT> output_tensor { "tensor" };
    holoscan::Output<hololink::CaptureMetadataT> output_metadata { "metadata" };

    void start() override;
    void stop() override;
    void setup(holoscan::OperatorSpec& spec) override;
    [[nodiscard]] holoscan::Contract contract() const override;
    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
        holoscan::ExecutionContext& context) override;

    uint32_t receiver_start_byte() override;
    uint32_t received_line_bytes(uint32_t line_bytes) override;
    uint32_t transmitted_line_bytes(hololink::csi::PixelFormat pixel_format, uint32_t pixel_width) override;
    void configure(uint32_t start_byte, uint32_t bytes_per_line, uint32_t pixel_width,
        uint32_t pixel_height, hololink::csi::PixelFormat pixel_format,
        uint32_t trailing_bytes) override;

    /**
     * Get the length of the CSI data in bytes
     */
    size_t get_csi_length() const;

    /**
     * Get the size of the sub frame in bytes.
     * If sub_frame_rows_ is non-zero, this will return the size of the sub frame in bytes.
     * Otherwise, it will return the size of the full frame in bytes.
     */
    size_t get_sub_frame_size() const;

private:
    // Frames dropped because the downstream queue was full, reported at most once per second.
    uint32_t backpressure_drops_ { 0 };
    std::chrono::steady_clock::time_point backpressure_report_time_ {};
    // Sub-frame mode: cuts the received chunks into bands (see sub_frame_bands.hpp).
    std::optional<SubFrameBands> sub_frame_bands_;
    bool drop_incomplete_ { false };
    // Frames/sub-frames part of whose data was lost upstream, reported at most once per second.
    hololink::operators::DropCounter incomplete_frames_;
    // Frames whose payload and metadata were stamped as different frames, reported at most once
    // per second.
    hololink::operators::DropCounter mismatched_frames_;
    void count_incomplete_frame(int64_t sub_frame_offset);
    // Sub-frame mode: received chunks larger than a band, dropped; reported at most once per second.
    hololink::operators::DropCounter oversized_chunks_;
    void count_oversized_chunk(size_t chunk_size);
    void carry_over_sub_frame_tail(const uint8_t* in_base, size_t in_total_size, size_t tail_size,
        CUstream cuda_stream);
    void count_backpressure_drop(int64_t sub_frame_offset);
    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute_impl(
        holoscan::ExecutionContext& context);

    int cuda_device_ordinal_ { 0 };
    uint32_t sub_frame_rows_ { 0 };
    std::optional<hololink::csi::BayerFormat> bayer_format_;
    bool header_included_ { true };

    CUcontext cuda_context_ = nullptr;
    CUdevice cuda_device_ = 0;
    bool is_integrated_ = false;
    bool host_memory_warning_ = false;

    std::shared_ptr<hololink::common::CudaFunctionLauncher> cuda_function_launcher_;

    size_t sub_frame_memory_size_ = 0;
    common::UniqueCUdeviceptr sub_frame_memory_;

    uint32_t pixel_width_ = 0;
    uint32_t pixel_height_ = 0;
    hololink::csi::PixelFormat pixel_format_ = hololink::csi::PixelFormat::RAW_8;
    uint32_t start_byte_ = 0;
    uint32_t bytes_per_line_ = 0;
    size_t csi_length_ = 0;

    size_t frame_size_ = 0; ///< the size of the frame in bytes, might be different from the csi_length_ if sub_frame_rows_ is set

    bool configured_ = false;
};

} // namespace hololink::operators

#endif /* SRC_OPERATORS_CSI_TO_BAYER_CSI_TO_BAYER */
