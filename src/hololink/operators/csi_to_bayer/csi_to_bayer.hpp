/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>

#include <hololink/core/csi_controller.hpp>
#include <hololink/core/csi_formats.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/utils/cuda_stream_handler.hpp>

#include <cuda.h>

namespace hololink::common {

class CudaFunctionLauncher;

} // namespace hololink::common

namespace hololink::operators {

class CsiToBayerOp : public holoscan::Operator, public hololink::csi::CsiConverter {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(CsiToBayerOp);

    void start() override;
    void stop() override;
    void setup(holoscan::OperatorSpec& spec) override;
    void compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
        holoscan::ExecutionContext&) override;

    uint32_t receiver_start_byte() override;
    uint32_t received_line_bytes(uint32_t line_bytes) override;
    uint32_t transmitted_line_bytes(hololink::csi::PixelFormat pixel_format, uint32_t pixel_width) override;
    void configure(uint32_t start_byte, uint32_t bytes_per_line, uint32_t pixel_width, uint32_t pixel_height, hololink::csi::PixelFormat pixel_format, uint32_t trailing_bytes) override;
    size_t get_csi_length();

private:
    holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
    holoscan::Parameter<int> cuda_device_ordinal_;
    holoscan::Parameter<std::string> out_tensor_name_;

    CUcontext cuda_context_ = nullptr;
    CUdevice cuda_device_ = 0;
    bool is_integrated_ = false;
    bool host_memory_warning_ = false;

    holoscan::CudaStreamHandler cuda_stream_handler_;

    std::shared_ptr<hololink::common::CudaFunctionLauncher> cuda_function_launcher_;
    uint32_t pixel_width_ = 0;
    uint32_t pixel_height_ = 0;
    hololink::csi::PixelFormat pixel_format_ = hololink::csi::PixelFormat::RAW_8;
    uint32_t start_byte_ = 0;
    uint32_t bytes_per_line_ = 0;
    size_t csi_length_ = 0;
    bool configured_ = false;
};

} // namespace hololink::operators

#endif /* SRC_OPERATORS_CSI_TO_BAYER_CSI_TO_BAYER */
