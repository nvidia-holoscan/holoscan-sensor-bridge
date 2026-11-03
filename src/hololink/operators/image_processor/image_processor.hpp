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

#ifndef SRC_HOLOLINK_OPERATORS_IMAGE_PROCESSOR_IMAGE_PROCESSOR
#define SRC_HOLOLINK_OPERATORS_IMAGE_PROCESSOR_IMAGE_PROCESSOR

#include <chrono>
#include <memory>

#include <capture_metadata_traits.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/operators/capture_metadata_utils.hpp>

namespace hololink::operators {

// Params for HSDK5 graph.op<>() construction pattern.
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
    holoscan::param_schema("com.nvidia.hololink.operators.ImageProcessorParams", 1),
    holoscan::field<&ImageProcessorParams::pixel_format>("pixel_format")
        .description("Raw pixel-format code of the input image."),
    holoscan::field<&ImageProcessorParams::bayer_format>("bayer_format")
        .description("Raw Bayer CFA phase code of the input image."),
    holoscan::field<&ImageProcessorParams::optical_black>("optical_black")
        .description("Optical-black level subtracted during processing."),
    holoscan::field<&ImageProcessorParams::cuda_device_ordinal>("cuda_device_ordinal")
        .description("CUDA device on which to run the image-processing kernel."),
    holoscan::field<&ImageProcessorParams::pixel_width>("pixel_width")
        .description("Width of the input image in pixels. With pixel_height, it sizes the output "
                     "tensor pool's slots; 0 (the default) keeps a 64 MiB slot."),
    holoscan::field<&ImageProcessorParams::pixel_height>("pixel_height")
        .description("Height of the input image in pixels; see pixel_width."));

class ImageProcessorOp : public holoscan::Operator<ImageProcessorParams> {
public:
    void setup(holoscan::OperatorSpec& spec) override;
    holoscan::Contract contract() const override;
    void start() override;
    void stop() override;
    holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

    holoscan::Input<holoscan::schema::ImageT> input_tensor { "in_tensor" };
    holoscan::Input<hololink::CaptureMetadataT> input_metadata { "in_metadata" };
    holoscan::Output<holoscan::schema::ImageT> output_tensor { "tensor" };
    holoscan::Output<hololink::CaptureMetadataT> output_metadata { "metadata" };

private:
    // Frames dropped because the downstream queue was full, reported at most once per second.
    uint32_t backpressure_drops_ { 0 };
    // Frames whose payload and metadata were stamped as different frames, reported at most once
    // per second.
    hololink::operators::DropCounter mismatched_frames_;
    std::chrono::steady_clock::time_point backpressure_report_time_ {};
    void count_backpressure_drop(int64_t sub_frame_offset);
    // HOLOLINK_RX_TRACE=1: once-per-second pool-slot wait statistics.
    std::chrono::microseconds trace_max_alloc_ { 0 };
    uint32_t trace_frames_ { 0 };
    std::chrono::steady_clock::time_point trace_report_time_ {};
    holoscan::expected<void, holoscan::Error> compute_impl(holoscan::ExecutionContext& context);

    int pixel_format_ { 0 };
    int bayer_format_ { 0 };
    int32_t optical_black_ { 0 };
    int cuda_device_ordinal_ { 0 };

    CUcontext cuda_context_ = nullptr;
    CUdevice cuda_device_ = 0;
    bool is_integrated_ = false;
    bool host_memory_warning_ = false;

    std::shared_ptr<hololink::common::CudaFunctionLauncher> cuda_function_launcher_;

    hololink::common::UniqueCUdeviceptr histogram_memory_;
    hololink::common::UniqueCUdeviceptr white_balance_gains_memory_;

    uint32_t histogram_threadblock_size_;
    int64_t expected_frame_number_ = 0;
    // Sub-frame mode: the histogram holds the bands of frame expected_frame_number_. False until
    // the first band after start(): there is no previous frame to take gains from yet.
    bool accumulating_histogram_ = false;
};

} // namespace hololink::operators

#endif /* SRC_HOLOLINK_OPERATORS_IMAGE_PROCESSOR_IMAGE_PROCESSOR */
