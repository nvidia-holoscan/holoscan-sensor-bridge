/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <hololink/core/csi_controller.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/utils/cuda_stream_handler.hpp>

#include <cuda.h>

namespace hololink {
class DataChannel;
} // namespace hololink

namespace hololink::common {
class CudaFunctionLauncher;
} // namespace hololink::common

namespace hololink::operators {

/**
 * Operator that enables the HSB packetizer to output 10- and 12-bit packed pixel formats,
 * then converts the output to 16 bits per pixel.
 *   - 10-bit format is packed 3 pixels per 4 bytes as {2'b0, p3[9:0], p2[9:0], p1[9:0]}
 *   - 12-bit format is packed 2 pixels per 3 bytes as {p2[11:0], p1[11:0]}
 */
class PackedFormatConverterOp : public holoscan::Operator, public hololink::csi::CsiConverter {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(PackedFormatConverterOp);

    void start() override;
    void stop() override;
    void setup(holoscan::OperatorSpec& spec) override;
    void compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
        holoscan::ExecutionContext&) override;

    uint32_t receiver_start_byte() override;
    uint32_t received_line_bytes(uint32_t line_bytes) override;
    uint32_t transmitted_line_bytes(hololink::csi::PixelFormat pixel_format,
        uint32_t pixel_width) override;
    void configure(uint32_t start_byte, uint32_t bytes_per_line,
        uint32_t pixel_width, uint32_t pixel_height,
        hololink::csi::PixelFormat pixel_format,
        uint32_t trailing_bytes) override;

    size_t get_frame_size();

private:
    holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
    holoscan::Parameter<int> cuda_device_ordinal_;
    holoscan::Parameter<std::string> in_tensor_name_;
    holoscan::Parameter<std::string> out_tensor_name_;
    holoscan::Parameter<DataChannel*> hololink_channel_;

    CUcontext cuda_context_ = nullptr;
    CUdevice cuda_device_ = 0;
    bool is_integrated_ = false;
    bool host_memory_warning_ = false;

    holoscan::CudaStreamHandler cuda_stream_handler_;

    std::shared_ptr<hololink::common::CudaFunctionLauncher> cuda_function_launcher_;

    uint32_t start_byte_;
    uint32_t bytes_per_line_;
    uint32_t pixel_width_;
    uint32_t pixel_height_;
    hololink::csi::PixelFormat pixel_format_;
    uint32_t trailing_bytes_;
    bool configured_ = false;
};

} // namespace hololink::operators
