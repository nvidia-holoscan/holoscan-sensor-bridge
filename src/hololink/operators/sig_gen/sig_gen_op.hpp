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

#ifndef OPERATORS_SIG_GEN_SIG_GEN_OP_HPP
#define OPERATORS_SIG_GEN_SIG_GEN_OP_HPP

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include <cuda.h>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/port.hpp>

#include <hololink/common/gui_renderer.hpp>

#include "sig_gen_op.cuh"

namespace hololink::operators {

/***
 * The operator generates a signal (a sequence of floats) in the GPU memory and emits it.
 * The signal can be defined by passing parameters to the constructor or adjusted via GUI.
 * IQ signal component samples are interleaved (I0Q0 I1Q1 I2Q2...).
 *
 * It's a source: it generates a buffer every period, as fast as it can when generating takes
 * longer. Each buffer continues the signal where the last one emitted left off; a buffer
 * dropped because downstream is full is regenerated. With count > 0 it completes after that
 * many activations -- including any whose buffer was dropped.
 *
 * Output:
 *  'output' - The generated float signal tensor.
 */
class SignalGeneratorOp : public holoscan::Operator<> {
public:
    SignalGeneratorOp(unsigned samples_count,
        Rational sampling_interval,
        std::string in_phase = "0",
        std::string quadrature = "0",
        std::string cuda_toolkit_include_path = "",
        std::shared_ptr<hololink::ImGuiRenderer> renderer = nullptr,
        std::chrono::nanoseconds period = std::chrono::milliseconds(1),
        uint64_t count = 0);
    ~SignalGeneratorOp();

    void setup(holoscan::OperatorSpec& spec) override;
    [[nodiscard]] holoscan::Contract contract() const override;
    void start() override;
    void stop() override;
    holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

    holoscan::Output<holoscan::Tensor> output { "output" };

    enum class SignalComponentType {
        InPhase,
        Quadrature
    };
    static constexpr unsigned number_of_signal_components = 2;

private:
    class GUI;
    std::shared_ptr<GUI> gui_;

    size_t index_ = 0;
    uint64_t frame_id_ = 0;
    uint64_t dropped_ = 0;
    struct SignalExpression {
        SignalExpression() = default;
        void update_expression(const std::string& str);
        void set_cuda_toolkit_include_path(const std::string& cuda_toolkit_include_path);
        std::string str() const;

        size_t evaluate(
            float* device_output,
            size_t count,
            size_t samples_index,
            const Rational& sampling_interval,
            size_t stride,
            SignalContext* device_context,
            cudaStream_t stream);

    private:
        std::string str_;
        expr_eval::Parser parser_;
        expr_eval::Expression expr_;
    };
    std::array<SignalExpression, number_of_signal_components> signal_expressions_;

    std::shared_ptr<hololink::ImGuiRenderer> renderer_;
    unsigned samples_count_;
    Rational sampling_interval_;
    std::array<std::string, number_of_signal_components> signal_expression_strs_;
    std::string cuda_toolkit_include_path_;
    std::chrono::nanoseconds period_;
    uint64_t count_;
    CUdevice cuda_device_ = 0;
    CUcontext cuda_context_ = nullptr;
    SignalContext* device_context_ = nullptr;
};

} // namespace hololink::operators

#endif /* OPERATORS_SIG_GEN_SIG_GEN_OP_HPP */
