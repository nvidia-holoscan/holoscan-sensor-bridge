/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_OPERATORS_PACKED_FORMAT_CONVERTER_OP_HPP
#define HOLOLINK_MODULE_OPERATORS_PACKED_FORMAT_CONVERTER_OP_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include <cuda.h>

#include <capture_metadata_traits.hpp>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include <hololink/operators/capture_metadata_utils.hpp> // DropCounter

#include "hololink/module/csi_converter.hpp"
#include "hololink/module/operators/csi_geometry_schema.hpp"

namespace hololink::module::operators {

class CudaFunctionLauncher;

/**
 * @brief Configuration parameters for PackedFormatConverterOp.
 */
struct PackedFormatConverterParams {
    int cuda_device_ordinal = 0;
    // Stamped onto every emitted schema::ImageT's encoding field; unset stamps CUSTOM, so a
    // downstream demosaic uses its own grid position rather than a guess.
    std::optional<hololink::module::csi::BayerFormat> bayer_format;
    // Received-buffer geometry. This operator's only producer in either the module or
    // native tree is FusaCoeCaptureOp, which always strips the header before emitting --
    // so start_byte/trailing_bytes are ignored here (forced to 0 in configure()); populate
    // directly from the SAME hololink::module::csi::CsiGeometry
    // FusaCoeCaptureOp::make_geometry_adapter() fills for FusaCoeCaptureParams::geometry, no
    // separate training call needed.
    hololink::module::csi::CsiGeometry geometry;
};

HOLOSCAN_PARAM_SCHEMA(
    PackedFormatConverterParams,
    holoscan::param_schema("com.nvidia.hololink.module.operators.PackedFormatConverterParams", 1),
    holoscan::field<&PackedFormatConverterParams::cuda_device_ordinal>("cuda_device_ordinal")
        .description("CUDA device on which to run the unpack kernel."),
    holoscan::field<&PackedFormatConverterParams::bayer_format>("bayer_format")
        .enum_values({ holoscan::param_enum_value("bggr", hololink::module::csi::BayerFormat::BGGR),
            holoscan::param_enum_value("rggb", hololink::module::csi::BayerFormat::RGGB),
            holoscan::param_enum_value("gbrg", hololink::module::csi::BayerFormat::GBRG),
            holoscan::param_enum_value("grbg", hololink::module::csi::BayerFormat::GRBG) })
        .description("CFA phase of the reconstructed Bayer mosaic's top-left 2x2 block."),
    holoscan::field<&PackedFormatConverterParams::geometry>("geometry")
        .template nested<hololink::module::csi::CsiGeometry>()
        .description("Received-buffer geometry; start_byte/trailing_bytes are ignored "
                     "(this operator's only producer, FusaCoeCaptureOp, always strips "
                     "the header before emitting)."));

/* Native module packed-CSI -> 16-bit converter operator, ported to HSDK5's
 * Operator<Params>/typed-port shape.
 *
 * The pre-port shell was dual-role: a holoscan::Operator AND a
 * hololink::module::csi::CsiConverterV1 the receiver (FusaCoeCaptureOp) trained through
 * configure_converter() to learn the received byte layout. HSDK5's graph.op<>() defers
 * construction, so there is no live instance to train that way. Since this operator's only
 * producer is FusaCoeCaptureOp (whose own make_geometry_adapter() already captures the same
 * geometry), this port drops the CsiConverterV1 role entirely and takes the geometry
 * directly via Params, auto-configuring in start() -- no adapter object needed here.
 *
 * Self-contained: it carries its own copy of the unpack engine (the NVRTC
 * packed8bitTo16bit / packed10bitTo16bit / packed12bitTo16bit kernels), expressed against
 * the module hololink::module::csi::PixelFormat. */
class PackedFormatConverterOp : public holoscan::Operator<PackedFormatConverterParams> {
public:
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

private:
    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute_impl(
        holoscan::ExecutionContext& context);

    void configure(uint32_t bytes_per_line, uint32_t pixel_width, uint32_t pixel_height,
        hololink::module::csi::PixelFormat pixel_format);

    int cuda_device_ordinal_ = 0;
    std::optional<hololink::module::csi::BayerFormat> bayer_format_;

    CUcontext cuda_context_ = nullptr;
    CUdevice cuda_device_ = 0;
    bool is_integrated_ = false;
    bool host_memory_warning_ = false;

    std::shared_ptr<CudaFunctionLauncher> cuda_function_launcher_;

    uint32_t bytes_per_line_ = 0;
    uint32_t pixel_width_ = 0;
    uint32_t pixel_height_ = 0;
    hololink::module::csi::PixelFormat pixel_format_
        = hololink::module::csi::PixelFormat::RAW_8;
    bool configured_ = false;
    // Frames dropped because downstream was behind, reported at most once per second.
    hololink::operators::DropCounter dropped_frames_;
    // Frames whose payload and metadata were stamped as different frames, reported at most once
    // per second.
    hololink::operators::DropCounter mismatched_frames_;
    void count_dropped_frame(const char* reason);
};

} // namespace hololink::module::operators

#endif // HOLOLINK_MODULE_OPERATORS_PACKED_FORMAT_CONVERTER_OP_HPP
