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

#ifndef SRC_HOLOLINK_OPERATORS_IMAGE_PROCESSOR_IMAGE_PROCESSOR
#define SRC_HOLOLINK_OPERATORS_IMAGE_PROCESSOR_IMAGE_PROCESSOR

#include <memory>

#include <hololink/operators/csi_to_bayer/csi_to_bayer.hpp>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/utils/cuda_stream_handler.hpp>

#include <hololink/common/cuda_helper.hpp>

namespace hololink::operators {

class ImageProcessorOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(ImageProcessorOp);

    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(holoscan::InputContext&, holoscan::OutputContext& op_output, holoscan::ExecutionContext&) override;

private:
    holoscan::Parameter<int> pixel_format_;
    holoscan::Parameter<int> bayer_format_;
    holoscan::Parameter<int32_t> optical_black_;
    holoscan::Parameter<int> cuda_device_ordinal_;

    CUcontext cuda_context_ = nullptr;
    CUdevice cuda_device_ = 0;
    bool is_integrated_ = false;
    bool host_memory_warning_ = false;

    holoscan::CudaStreamHandler cuda_stream_handler_;

    std::shared_ptr<hololink::common::CudaFunctionLauncher> cuda_function_launcher_;

    hololink::common::UniqueCUdeviceptr histogram_memory_;
    hololink::common::UniqueCUdeviceptr white_balance_gains_memory_;

    uint32_t histogram_threadblock_size_;
};

} // namespace hololink::operators

#endif /* SRC_HOLOLINK_OPERATORS_IMAGE_PROCESSOR_IMAGE_PROCESSOR */
