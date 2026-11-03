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

#ifndef OPERATORS_IQ_ENC_IQ_ENC_OP_HPP
#define OPERATORS_IQ_ENC_IQ_ENC_OP_HPP

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>

#include <hololink/common/gui_renderer.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/port.hpp>

namespace hololink::operators {

/***
 * The IQEncoderOp operator encodes in-phase and quadrature signal components
 * and creates an encoded buffer.
 *
 * Input:
 *  'input' - A float signal buffer (interleaved IQ samples)
 *
 * Output:
 *  'output' - The encoded int16 buffer
 */
class IQEncoderOp : public holoscan::Operator<> {
public:
    explicit IQEncoderOp(float scale = 1.0f, std::shared_ptr<hololink::ImGuiRenderer> renderer = nullptr);

    void setup(holoscan::OperatorSpec& spec) override;
    [[nodiscard]] holoscan::Contract contract() const override;
    void start() override;
    void stop() override;
    holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

    holoscan::Input<holoscan::Tensor> input { "input" };
    holoscan::Output<holoscan::Tensor> output { "output" };

private:
    std::atomic<float> scale_;
    std::shared_ptr<hololink::ImGuiRenderer> renderer_;
    std::optional<ImGuiRenderer::Handle> renderer_handle_;
    uint64_t dropped_ = 0;
};

} // namespace hololink::operators

#endif /* OPERATORS_IQ_ENC_IQ_ENC_OP_HPP */
