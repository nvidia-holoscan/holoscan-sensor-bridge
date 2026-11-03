/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_OPERATORS_IMAGE_PROCESSOR_OP_HPP
#define HOLOLINK_MODULE_OPERATORS_IMAGE_PROCESSOR_OP_HPP

#include <cstdint>
#include <memory>

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

#include "hololink/module/cuda_unique.hpp" // UniqueCUdeviceptr

namespace hololink::module::operators {

class CudaFunctionLauncher;

/**
 * @brief Configuration parameters for ImageProcessorOp.
 */
struct ImageProcessorParams {
    int pixel_format = 0;
    int bayer_format = 0;
    int32_t optical_black = 0;
    int cuda_device_ordinal = 0;
    // The input image's size in pixels. When both are set, one output tensor pool slot holds one
    // such image of 16-bit samples; when either is 0 (the default), a slot keeps the 64 MiB
    // ceiling of earlier releases, so that applications that don't know the size still run.
    uint32_t pixel_width = 0;
    uint32_t pixel_height = 0;
};

HOLOSCAN_PARAM_SCHEMA(
    ImageProcessorParams,
    holoscan::param_schema("com.nvidia.hololink.module.operators.ImageProcessorParams", 1),
    holoscan::field<&ImageProcessorParams::pixel_format>("pixel_format")
        .description("Received pixel packing, as a hololink::module::csi::PixelFormat value."),
    holoscan::field<&ImageProcessorParams::bayer_format>("bayer_format")
        .description("CFA phase, as a hololink::module::csi::BayerFormat value."),
    holoscan::field<&ImageProcessorParams::optical_black>("optical_black")
        .description("Optical black level to subtract before auto white-balance."),
    holoscan::field<&ImageProcessorParams::cuda_device_ordinal>("cuda_device_ordinal")
        .description("CUDA device on which to run the image-processing kernels."),
    holoscan::field<&ImageProcessorParams::pixel_width>("pixel_width")
        .description("Width of the input image in pixels. With pixel_height, it sizes the output "
                     "tensor pool's slots; 0 (the default) keeps a 64 MiB slot."),
    holoscan::field<&ImageProcessorParams::pixel_height>("pixel_height")
        .description("Height of the input image in pixels; see pixel_width."));

/* Native module image-processing operator, ported to HSDK5's Operator<Params>/typed-port
 * shape. A plain holoscan::Operator (unlike CsiToBayerOp it implements no converter
 * interface). Applies optical-black correction and a Grey-World auto white-balance to the
 * 16-bit Bayer image produced upstream.
 *
 * Self-contained: it carries its own copy of the engine (the NVRTC applyBlackLevel /
 * histogram / calcWBGains / applyOperations kernels, the histogram sizing in start(), and
 * the sub-frame-aware white-balance accumulation in compute()), byte-identical to the
 * pre-port module kernels and to the already-ported native hololink::operators::ImageProcessorOp.
 * The pixel_format / bayer_format params are plain integer enumerator values, interpreted
 * against the module hololink::module::csi::PixelFormat / BayerFormat.
 *
 * Unlike the pre-port shell (which mutated the received gxf::Entity in place and re-emitted
 * it), HSDK5's runtime-pool model treats a received Input<T> sample as read-only under its
 * receive lease -- compute() allocates its own output tensor, copies the input into it, then
 * runs the mutating kernels on the copy. Mirrors native ImageProcessorOp's identical
 * adaptation of the same pre-port shell. */
class ImageProcessorOp : public holoscan::Operator<ImageProcessorParams> {
public:
    holoscan::Input<holoscan::schema::ImageT> input_tensor { "in_tensor" };
    holoscan::Input<hololink::CaptureMetadataT> input_metadata { "in_metadata" };
    holoscan::Output<holoscan::schema::ImageT> output_tensor { "tensor" };
    holoscan::Output<hololink::CaptureMetadataT> output_metadata { "metadata" };

    void setup(holoscan::OperatorSpec& spec) override;
    [[nodiscard]] holoscan::Contract contract() const override;
    void start() override;
    void stop() override;
    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
        holoscan::ExecutionContext& context) override;

private:
    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute_impl(
        holoscan::ExecutionContext& context);

    int pixel_format_ = 0;
    int bayer_format_ = 0;
    int32_t optical_black_ = 0;
    int cuda_device_ordinal_ = 0;

    CUcontext cuda_context_ = nullptr;
    CUdevice cuda_device_ = 0;
    bool is_integrated_ = false;
    bool host_memory_warning_ = false;

    std::shared_ptr<CudaFunctionLauncher> cuda_function_launcher_;

    hololink::module::UniqueCUdeviceptr histogram_memory_;
    hololink::module::UniqueCUdeviceptr white_balance_gains_memory_;

    uint32_t histogram_threadblock_size_ = 0;
    // Sub-frame mode: the frame whose bands' histograms are being accumulated, once there is one.
    int64_t expected_frame_number_ = 0;
    bool accumulating_histogram_ = false;
    // Frames dropped because downstream was behind, reported at most once per second.
    hololink::operators::DropCounter dropped_frames_;
    // Frames whose payload and metadata were stamped as different frames, reported at most once
    // per second.
    hololink::operators::DropCounter mismatched_frames_;
    void count_dropped_frame(int64_t sub_frame_offset);
};

} // namespace hololink::module::operators

#endif // HOLOLINK_MODULE_OPERATORS_IMAGE_PROCESSOR_OP_HPP
