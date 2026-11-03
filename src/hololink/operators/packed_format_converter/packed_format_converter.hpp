/*
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
 */

#pragma once

#include <memory>
#include <optional>

#include <capture_metadata_traits.hpp>
#include <hololink/core/csi_controller.hpp>
#include <hololink/core/csi_formats.hpp>
#include <hololink/operators/capture_metadata_utils.hpp>
#include <hololink/operators/csi_geometry_schema.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include <cuda.h>

namespace hololink::common {
class CudaFunctionLauncher;
} // namespace hololink::common

namespace hololink::operators {

struct PackedFormatConverterParams {
    int cuda_device_ordinal = 0;
    // CFA phase of the reconstructed Bayer mosaic -- not derivable from the packed-format
    // conversion itself, so the camera driver's own camera->get_bayer_format() must be threaded
    // through here explicitly. Stamped onto every emitted schema::ImageT's encoding field (see
    // bayer_image_encoding()); unset stamps CUSTOM, leaving the CFA phase to the consumer.
    std::optional<hololink::csi::BayerFormat> bayer_format;
    // Geometry of the HSB-packed frames, as the sensor driver reports it through
    // FusaCoeCaptureOp::make_geometry_adapter(geometry, /*packetizer_enabled=*/true). Required.
    // start_byte is applied only when header_included is true.
    // trailing_bytes is not part of the unpacked image.
    hololink::csi::CsiGeometry geometry;
    // Full-frame CRC paths retain the CSI header; ordinary capture exports pixels only.
    bool header_included = false;
};

HOLOSCAN_PARAM_SCHEMA(
    PackedFormatConverterParams,
    holoscan::param_schema("com.nvidia.hololink.operators.PackedFormatConverterParams", 1),
    holoscan::field<&PackedFormatConverterParams::cuda_device_ordinal>("cuda_device_ordinal")
        .description("CUDA device on which to run the unpack kernel."),
    holoscan::field<&PackedFormatConverterParams::bayer_format>("bayer_format")
        .enum_values({ holoscan::param_enum_value("bggr", hololink::csi::BayerFormat::BGGR),
            holoscan::param_enum_value("rggb", hololink::csi::BayerFormat::RGGB),
            holoscan::param_enum_value("gbrg", hololink::csi::BayerFormat::GBRG),
            holoscan::param_enum_value("grbg", hololink::csi::BayerFormat::GRBG) })
        .description("CFA phase of the reconstructed Bayer mosaic's top-left 2x2 block; unset "
                     "stamps CUSTOM."),
    holoscan::field<&PackedFormatConverterParams::geometry>("geometry")
        .template nested<hololink::csi::CsiGeometry>()
        .description("Geometry of the HSB-packed frames; start_byte applies when header_included is true."),
    holoscan::field<&PackedFormatConverterParams::header_included>("header_included")
        .description("Input includes a CSI header of geometry.start_byte bytes (default: pixels only)."));

/**
 * Unpacks HSB-packed RAW_8/RAW_10/RAW_12 frames (the FuSa CoE packetizer's line format) into a
 * 16-bit Bayer schema::ImageT. Configured entirely through PackedFormatConverterParams: HSDK5's
 * graph.op<>() defers construction, so a sensor driver can no longer configure a live instance
 * through the CsiConverter interface as it did in HSDK4.
 */
class PackedFormatConverterOp : public holoscan::Operator<PackedFormatConverterParams> {
public:
    void setup(holoscan::OperatorSpec& spec) override;
    [[nodiscard]] holoscan::Contract contract() const override;
    void start() override;
    void stop() override;
    holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

    holoscan::Input<holoscan::Tensor> input_tensor { "in_tensor" };
    holoscan::Input<hololink::CaptureMetadataT> input_metadata { "in_metadata" };
    holoscan::Output<holoscan::schema::ImageT> output_tensor { "tensor" };
    holoscan::Output<hololink::CaptureMetadataT> output_metadata { "metadata" };

    // Bytes a packed line of pixel_width pixels occupies before the HSB pads it to 64: the
    // CSI-2 line as the format's packetizer program (hololink::csi::get_packetizer_program())
    // reshapes it, a whole number of the program's loops.
    static uint32_t transmitted_line_bytes(hololink::csi::PixelFormat pixel_format, uint32_t pixel_width);

private:
    holoscan::expected<void, holoscan::Error> compute_impl(holoscan::ExecutionContext& context);
    void count_dropped_frame(const char* reason);

    CUcontext cuda_context_ = nullptr;
    CUdevice cuda_device_ = 0;
    bool is_integrated_ = false;
    bool host_memory_warning_ = false;

    std::shared_ptr<hololink::common::CudaFunctionLauncher> cuda_function_launcher_;

    uint32_t bytes_per_line_ = 0;
    uint32_t pixel_width_ = 0;
    uint32_t pixel_height_ = 0;
    hololink::csi::PixelFormat pixel_format_ = {};
    hololink::operators::DropCounter dropped_frames_;
    // Frames whose payload and metadata were stamped as different frames, reported at most once
    // per second.
    hololink::operators::DropCounter mismatched_frames_;
};

} // namespace hololink::operators
