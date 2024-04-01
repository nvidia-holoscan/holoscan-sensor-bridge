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

#ifndef SRC_HOLOLINK_OPERATORS_GAMMA_CORRECTION_GAMMA_CORRECTION
#define SRC_HOLOLINK_OPERATORS_GAMMA_CORRECTION_GAMMA_CORRECTION

#include <memory>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/utils/cuda_stream_handler.hpp>

#include <cuda.h>

namespace hololink::native {

class CudaFunctionLauncher;

} // namespace hololink::native

namespace hololink::operators {

class GammaCorrectionOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(GammaCorrectionOp);

    void start() override;
    void stop() override;
    void setup(holoscan::OperatorSpec& spec) override;
    void compute(holoscan::InputContext&, holoscan::OutputContext& op_output, holoscan::ExecutionContext&) override;

private:
    holoscan::Parameter<float> gamma_;
    holoscan::Parameter<int> cuda_device_ordinal_;

    CUcontext cuda_context_ = nullptr;
    CUdevice cuda_device_ = 0;
    bool is_integrated_ = false;
    bool host_memory_warning_ = false;

    holoscan::CudaStreamHandler cuda_stream_handler_;

    std::shared_ptr<hololink::native::CudaFunctionLauncher> cuda_function_launcher_;
};

} // namespace hololink::operators

#endif /* SRC_HOLOLINK_OPERATORS_GAMMA_CORRECTION_GAMMA_CORRECTION */
