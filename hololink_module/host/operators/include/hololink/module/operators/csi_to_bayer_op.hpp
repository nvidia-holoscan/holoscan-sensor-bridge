/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_OPERATORS_CSI_TO_BAYER_OP_HPP
#define HOLOLINK_MODULE_OPERATORS_CSI_TO_BAYER_OP_HPP

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
#include <hololink/operators/csi_to_bayer/sub_frame_bands.hpp>

#include "hololink/module/csi_converter.hpp"
#include "hololink/module/cuda_unique.hpp" // UniqueCUdeviceptr
#include "hololink/module/operators/csi_geometry_schema.hpp"

namespace hololink::module::operators {

class CudaFunctionLauncher;

/**
 * @brief Configuration parameters for CsiToBayerOp.
 */
struct CsiToBayerParams {
    int cuda_device_ordinal = 0;
    // 0: whole frames. Otherwise the rows of each sub-frame band: even, and a divisor of the
    // frame height; the receiver's frame_size must then be bytes_per_line * sub_frame_rows.
    uint32_t sub_frame_rows = 0;
    // CSI geometry, as reported by the sensor driver; take it from CsiToBayerConverter::geometry().
    hololink::module::csi::CsiGeometry geometry;
    // CFA phase of the reconstructed Bayer mosaic -- not derivable from geometry, so the
    // camera driver's own camera->get_bayer_format() (or equivalent) must be threaded
    // through here explicitly. Stamped onto every emitted schema::ImageT's encoding field.
    // Unset stamps CUSTOM, so a downstream demosaic uses its own grid position rather than a guess.
    std::optional<hololink::module::csi::BayerFormat> bayer_format;
    // Whether the non-sub-frame input tensor still includes the start_byte header region.
    // BaseReceiverOp-derived receivers (LinuxReceiverOp/RoceReceiverOp) emit the full raw
    // buffer including it (true, the default); FusaCoeCaptureOp already strips it before
    // emitting (see fusa_coe_capture_op.cpp), so an app wiring CsiToBayerOp downstream of
    // FusaCoeCaptureOp (the non-packetized branch) must set this to false.
    bool header_included = true;
    // What to do with a frame (in sub-frame mode, a band) part of whose data was lost upstream;
    // see hololink::operators::CsiToBayerParams::drop_incomplete. By default it is emitted
    // marked holoscan::SampleFlags::kDegraded; when set, it is withheld.
    bool drop_incomplete = false;
};

HOLOSCAN_PARAM_SCHEMA(
    CsiToBayerParams,
    holoscan::param_schema("com.nvidia.hololink.module.operators.CsiToBayerParams", 1),
    holoscan::field<&CsiToBayerParams::cuda_device_ordinal>("cuda_device_ordinal")
        .description("CUDA device on which to run the CSI-to-Bayer conversion kernel."),
    holoscan::field<&CsiToBayerParams::sub_frame_rows>("sub_frame_rows")
        .description("Row count of one sub-frame activation, or 0 for whole-frame mode."),
    holoscan::field<&CsiToBayerParams::geometry>("geometry")
        .template nested<hololink::module::csi::CsiGeometry>()
        .description("CSI frame geometry, filled by the camera driver's configure_converter()."),
    holoscan::field<&CsiToBayerParams::bayer_format>("bayer_format")
        .enum_values({ holoscan::param_enum_value("bggr", hololink::module::csi::BayerFormat::BGGR),
            holoscan::param_enum_value("rggb", hololink::module::csi::BayerFormat::RGGB),
            holoscan::param_enum_value("gbrg", hololink::module::csi::BayerFormat::GBRG),
            holoscan::param_enum_value("grbg", hololink::module::csi::BayerFormat::GRBG) })
        .description("CFA phase of the reconstructed Bayer mosaic's top-left 2x2 block."),
    holoscan::field<&CsiToBayerParams::header_included>("header_included")
        .description("Whether the input tensor still includes the start_byte header region."),
    holoscan::field<&CsiToBayerParams::drop_incomplete>("drop_incomplete")
        .description("Withhold a band or frame with lost data instead of passing it on marked "
                     "degraded."));

/* Native module CSI -> Bayer converter operator, ported to HSDK5's Operator<Params>/
 * typed-port shape.
 *
 * The pre-port shell was dual-role: a holoscan::Operator AND a
 * hololink::module::csi::CsiConverterV1 the module sensor drivers (Imx274Cam / Vb1940Cam)
 * trained directly via configure_converter(). HSDK5's graph.op<>() defers construction, so
 * there is no live instance at app-authoring time to train that way (the same problem
 * FusaCoeCaptureOp's CsiConverterV1 role already hit and solved). This port drops the
 * CsiConverterV1 inheritance from the operator itself; CsiToBayerConverter (declared after this
 * class) takes over that role: a sensor driver trains it, it reports get_csi_length() for the
 * receiver's frame_size, and its geometry() is then copied into CsiToBayerParams::geometry
 * before graph.op<CsiToBayerOp>() is called. make_geometry_adapter() wraps it for callers
 * that only need the geometry.
 *
 * Self-contained: it carries its own copy of the CSI -> Bayer engine (the NVRTC
 * frameReconstruction8/10/12 kernels, the sub-frame accumulation, and the geometry helpers),
 * expressed against the module hololink::module::csi::PixelFormat. */
class CsiToBayerOp : public holoscan::Operator<CsiToBayerParams> {
public:
    holoscan::Input<holoscan::Tensor> input_tensor { "in_tensor" };
    holoscan::Input<hololink::CaptureMetadataT> input_metadata { "in_metadata" };
    holoscan::Output<holoscan::schema::ImageT> output_tensor { "tensor" };
    holoscan::Output<hololink::CaptureMetadataT> output_metadata { "metadata" };

    /** No live CsiToBayerOp instance is needed to call this -- call it, and
     * camera->configure_converter(adapter), before graph.op<CsiToBayerOp>(), then copy
     * `out` into CsiToBayerParams::geometry. The adapter is a CsiToBayerConverter that also
     * writes what the sensor reports into `out`; use a CsiToBayerConverter directly to get the
     * receiver's frame_size as well. */
    static std::shared_ptr<hololink::module::csi::CsiConverterV1> make_geometry_adapter(
        hololink::module::csi::CsiGeometry& out);

    void start() override;
    void stop() override;
    void setup(holoscan::OperatorSpec& spec) override;
    [[nodiscard]] holoscan::Contract contract() const override;
    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
        holoscan::ExecutionContext& context) override;

    // Pure geometry math, identical to the pre-port CsiConverterV1 methods -- static now
    // that no vtable/live-instance is required. CsiToBayerConverter delegates to these so the
    // math lives in exactly one place.
    static uint32_t receiver_start_byte();
    static uint32_t received_line_bytes(uint32_t line_bytes);
    static uint32_t transmitted_line_bytes(
        hololink::module::csi::PixelFormat pixel_format, uint32_t pixel_width);

    /// CSI data length in bytes: start_byte + bytes_per_line * pixel_height + trailing_bytes.
    static size_t csi_length(const hololink::module::csi::CsiGeometry& geometry);

private:
    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute_impl(
        holoscan::ExecutionContext& context);

    // Was CsiConverterV1::configure(); now private, called from start() when
    // params().geometry.pixel_width != 0.
    void configure(uint32_t start_byte, uint32_t bytes_per_line, uint32_t pixel_width,
        uint32_t pixel_height, hololink::module::csi::PixelFormat pixel_format,
        uint32_t trailing_bytes);
    // The bytes of the chunk past the band just completed start the next band.
    void carry_over_sub_frame_tail(const uint8_t* in_base, size_t in_total_size, size_t tail_size,
        CUstream cuda_stream);
    void count_backpressure_drop(int64_t sub_frame_offset);
    void count_incomplete_frame(int64_t sub_frame_offset);
    void count_oversized_chunk(size_t chunk_size);

    int cuda_device_ordinal_ = 0;
    uint32_t sub_frame_rows_ = 0;
    std::optional<hololink::module::csi::BayerFormat> bayer_format_;
    bool header_included_ = true;
    bool drop_incomplete_ = false;
    // Sub-frame mode: cuts the received chunks into bands, as the native CsiToBayerOp does.
    std::optional<hololink::operators::SubFrameBands> sub_frame_bands_;
    // Frames whose payload and metadata were stamped as different frames, reported at most once
    // per second.
    hololink::operators::DropCounter mismatched_frames_;
    // Reported at most once per second.
    hololink::operators::DropCounter backpressure_drops_;
    hololink::operators::DropCounter incomplete_frames_;
    // Sub-frame mode: received chunks larger than a band, dropped.
    hololink::operators::DropCounter oversized_chunks_;

    CUcontext cuda_context_ = nullptr;
    CUdevice cuda_device_ = 0;
    bool is_integrated_ = false;
    bool host_memory_warning_ = false;

    std::shared_ptr<CudaFunctionLauncher> cuda_function_launcher_;

    size_t sub_frame_memory_size_ = 0;
    hololink::module::UniqueCUdeviceptr sub_frame_memory_;

    uint32_t pixel_width_ = 0;
    uint32_t pixel_height_ = 0;
    hololink::module::csi::PixelFormat pixel_format_
        = hololink::module::csi::PixelFormat::RAW_8;
    uint32_t start_byte_ = 0;
    uint32_t bytes_per_line_ = 0;
    size_t csi_length_ = 0;

    bool configured_ = false;
};

/* The converter a sensor driver trains, standing in for CsiToBayerOp while the graph is being
 * built (HSDK5 constructs operators from their params, so there is no live CsiToBayerOp to pass
 * to camera->configure_converter(); this is the old `csi_op` role without the operator). It
 * applies the same layout rules CsiToBayerOp uses to read received frames:
 *
 *     auto csi_to_bayer = std::make_shared<CsiToBayerConverter>();
 *     camera->configure_converter(csi_to_bayer);
 *     receiver.frame_size = csi_to_bayer->get_csi_length();
 *     csi_params.geometry = csi_to_bayer->geometry();
 */
class CsiToBayerConverter : public hololink::module::csi::CsiConverterV1 {
public:
    void configure(uint32_t start_byte, uint32_t received_bytes_per_line, uint32_t pixel_width,
        uint32_t pixel_height, hololink::module::csi::PixelFormat pixel_format,
        uint32_t trailing_bytes = 0) override;
    uint32_t receiver_start_byte() override;
    uint32_t received_line_bytes(uint32_t line_bytes) override;
    uint32_t transmitted_line_bytes(
        hololink::module::csi::PixelFormat pixel_format, uint32_t pixel_width) override;

    /** Length of the CSI data in bytes (start_byte + bytes_per_line * height + trailing): the
     * frame_size the receiver operator must be given. Throws if the sensor has not yet
     * configured this converter. */
    size_t get_csi_length() const;

    /** What the sensor reported, for CsiToBayerParams::geometry. Throws if the sensor has not
     * yet configured this converter. */
    const hololink::module::csi::CsiGeometry& geometry() const;

private:
    hololink::module::csi::CsiGeometry geometry_;
    bool configured_ = false;
};

} // namespace hololink::module::operators

#endif // HOLOLINK_MODULE_OPERATORS_CSI_TO_BAYER_OP_HPP
