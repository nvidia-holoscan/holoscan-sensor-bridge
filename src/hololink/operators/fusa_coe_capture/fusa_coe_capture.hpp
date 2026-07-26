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

#include <functional>
#include <string>
#include <vector>

#include <hololink/core/csi_controller.hpp>
#include <holoscan/holoscan.hpp>

#include "fusa_coe_capture_core.hpp"

namespace hololink {
class DataChannel;
} // namespace hololink

namespace hololink::operators {

/**
 * @brief Operator class to capture images using NvFusaCaptureCoe
 */
class FusaCoeCaptureOp : public holoscan::Operator, public csi::CsiConverter {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(FusaCoeCaptureOp)

    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(holoscan::InputContext& op_input,
        holoscan::OutputContext& op_output,
        holoscan::ExecutionContext& context) override;

    uint32_t receiver_start_byte() override;
    uint32_t received_line_bytes(uint32_t line_bytes) override;
    uint32_t transmitted_line_bytes(csi::PixelFormat pixel_format,
        uint32_t pixel_width) override;
    void configure(uint32_t start_byte, uint32_t received_bytes_per_line,
        uint32_t pixel_width, uint32_t pixel_height,
        csi::PixelFormat pixel_format,
        uint32_t trailing_bytes) override;

    void configure_converter(csi::CsiConverter& converter);
    void configure_frame_size(uint32_t frame_size_bytes);

    static nvidia::gxf::Expected<void> buffer_release_callback(void* pointer);

private:
    holoscan::Parameter<std::string> interface_;
    holoscan::Parameter<std::vector<uint8_t>> mac_addr_;
    holoscan::Parameter<uint32_t> timeout_;
    holoscan::Parameter<std::string> out_tensor_name_;
    holoscan::Parameter<DataChannel*> hololink_channel_;
    holoscan::Parameter<std::function<void()>> device_start_;
    holoscan::Parameter<std::function<void()>> device_stop_;
    holoscan::Parameter<bool> cpu_output_;

    fusa_coe_capture::FusaCoeCaptureCore core_;
};

} // namespace hololink::operators
