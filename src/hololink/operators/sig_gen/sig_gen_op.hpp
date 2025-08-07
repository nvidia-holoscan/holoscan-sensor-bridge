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

#ifndef OPERATORS_SIG_GEN_SIG_GEN_OP_HPP
#define OPERATORS_SIG_GEN_SIG_GEN_OP_HPP

#include <memory>
#include <string>

#include <holoscan/holoscan.hpp>

#include <hololink/core/smart_object_pool.hpp>

#include "sig_gen_op.cuh"

namespace hololink {
class ImGuiRenderer;
namespace operators {

    /**
     * The operator generates a signal (a sequence of floats) in the GPU memory and emits it
     * The signal can be defined by passing a set of parameters to the operator
     * or by using the GUI (if enabled).
     * The IQ signal component samples are interleaved (I0Q0 I1Q1 I2Q2...)
     * The generated signal is also rendered in the GUI (if enabled)
     *
     * Output:
     *  'output' - The generated signal.
     *
     * Parameters:
     *  'samples_count'             - The number of samples (IQ pairs) to generate
     *  'sampling_interval'         - The interval between sequential samples
     *  'in_phase'                  - The in-phase expression
     *  'quadrature'                - The quadrature expression
     *  'cuda_toolkit_include_path' - A path to the cuda toolkit.
     *  'renderer'                  - Since Dear ImGui is not supported by Holoviz operators
     *                                a pointer to an ImGuiRenderer object is required.
     */
    class SignalGeneratorOp : public holoscan::Operator {
    public:
        HOLOSCAN_OPERATOR_FORWARD_ARGS(SignalGeneratorOp)

        SignalGeneratorOp() = default;

        void start() override;
        void stop() override;

        void initialize() override;
        void setup(holoscan::OperatorSpec& spec) override;

        void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
            holoscan::ExecutionContext& context) override;

        enum class SignalComponentType {
            InPhase,
            Quadrature
        };
        static constexpr unsigned number_of_signal_components = 2;

    private:
        using Tensor = std::shared_ptr<holoscan::Tensor>;
        class GUI;
        // The GUI is created when the `start` function is called and destroyed when the `stop` function
        // is called.
        std::shared_ptr<GUI> gui_;

        using DLManagedTensorContextPool = SmartObjectPool<holoscan::DLManagedTensorContext>;
        DLManagedTensorContextPool dl_managed_tensor_context_pool_;
        size_t index_ = 0; // The index of the current signal sample
        struct SignalExpression {
            SignalExpression() = default;
            void update_expression(const std::string& str);
            void set_cuda_toolkit_include_path(const std::string& cuda_toolkit_include_path);
            std::string str() const;

            // Returns the next samples index to be evaluated
            size_t evaluate(
                float* device_output, // Samples will be written here
                size_t count, // Number of samples to generate
                size_t samples_index, // The sample index to start with
                const Rational& sampling_interval, // The interval between samples
                size_t stride); // The stride between samples

        private:
            std::string str_;
            expr_eval::Parser parser_;
            expr_eval::Expression expr_;
        };
        std::array<SignalExpression, number_of_signal_components> signal_expressions_;

        // Parameters
        holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
        holoscan::Parameter<hololink::ImGuiRenderer*> renderer_;
        holoscan::Parameter<unsigned> samples_count_;
        holoscan::Parameter<Rational> sampling_interval_;
        std::array<holoscan::Parameter<std::string>, number_of_signal_components> signal_expression_strs_;
        holoscan::Parameter<std::string> cuda_toolkit_include_path_;
    };

} // namespace hololink::operators
} // namespace hololink

#endif /* OPERATORS_SIG_GEN_SIG_GEN_OP_HPP */
