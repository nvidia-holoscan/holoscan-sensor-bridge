/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#ifndef OPERATORS_IQ_ENC_IQ_ENC_OP_HPP
#define OPERATORS_IQ_ENC_IQ_ENC_OP_HPP

#include <mutex>

#include <hololink/common/gui_renderer.hpp>
#include <hololink/core/smart_object_pool.hpp>
#include <holoscan/holoscan.hpp>

namespace hololink::operators {

/***
 * The IQEncoderOp operator encode in-phase and quadrature signal components
 * and creates an encoded buffer.
 * A client may expect the receive a signal in a specific format
 * The IQEncoderOp operator will encode the signal to be in that format.
 * At the moment, the IQEncoder support only one format - ad9082
 * but in the future, more formats might be supported.
 *
 * Input:
 *  'input' - A signal buffer
 *
 * Output:
 *  'output' - The port the encoded buffer is sent to
 *
 *
 * Parameters:
 *  'scale'    - The signal will be scaled by this factor before it is encoded.
 *  'renderer' - Since Dear ImGui is not supported by Holoviz operators,
 *               a pointer to an ImGuiRenderer object is required.
 */
class IQEncoderOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(IQEncoderOp)

    IQEncoderOp() = default;

    void initialize() override;
    void setup(holoscan::OperatorSpec& spec) override;

    void start() override;
    void stop() override;

    void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
        holoscan::ExecutionContext& context) override;

private:
    using Tensor = std::shared_ptr<holoscan::Tensor>;
    using DLManagedTensorContextPool = SmartObjectPool<holoscan::DLManagedTensorContext>;
    DLManagedTensorContextPool dl_managed_tensor_context_pool_;
    std::mutex mutex_;
    ImGuiRenderer::Handle renderer_handle_;

    /**
     * Cuda implementation of the encode function.
     * encoded_output is a pointer to device memory where the encoded buffer is stored
     * count is the number of IQ pairs (samples)
     * scale is the scaling factor. The range of values is assumed to be [-scale, scale];
     */
    static void cuda_iq_encode(int16_t* encoded_output, const float* iq_components, size_t size, float max_abs_value);

    // Parameters
    holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
    holoscan::Parameter<float> scale_;
    holoscan::Parameter<hololink::ImGuiRenderer*> renderer_;
};

} // namespace hololink::operators

#endif /* OPERATORS_IQ_ENC_IQ_ENC_OP_HPP */
