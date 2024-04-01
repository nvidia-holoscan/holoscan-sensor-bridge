/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoscan/core/operator.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/utils/cuda_stream_handler.hpp>

#include <cuda.h>

namespace hololink::native {

class CudaFunctionLauncher;

} // namespace hololink::native

namespace hololink::operators {

class CsiToBayerOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(CsiToBayerOp);

    enum class PixelFormat {
        INVALID = -1,
        RAW_8 = 0,
        RAW_10 = 1,
        RAW_12 = 2,
    };

    void start() override;
    void stop() override;
    void setup(holoscan::OperatorSpec& spec) override;
    void compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
        holoscan::ExecutionContext&) override;

    void configure(uint32_t width, uint32_t height, PixelFormat pixel_format,
        uint32_t frame_start_size, uint32_t frame_end_size, uint32_t line_start_size,
        uint32_t line_end_size, uint32_t margin_left = 0, uint32_t margin_top = 0,
        uint32_t margin_right = 0, uint32_t margin_bottom = 0);
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

    std::shared_ptr<hololink::native::CudaFunctionLauncher> cuda_function_launcher_;

    uint32_t width_ = 0;
    uint32_t height_ = 0;
    PixelFormat pixel_format_ = PixelFormat::INVALID;
    uint32_t frame_start_size_ = 0;
    uint32_t frame_end_size_ = 0;
    uint32_t line_start_size_ = 0;
    uint32_t line_end_size_ = 0;

    uint32_t bytes_per_line_ = 0;
    size_t csi_length_ = 0;
};

} // namespace hololink::operators

#endif /* SRC_OPERATORS_CSI_TO_BAYER_CSI_TO_BAYER */
