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

#ifndef OPERATORS_SIG_VIEWER_SIG_VIEWER_OP_HPP
#define OPERATORS_SIG_VIEWER_SIG_VIEWER_OP_HPP

#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/port.hpp>

#include <hololink/common/gui_renderer.hpp>

namespace hololink {
class GuiStaging;
} // namespace hololink

namespace hololink::operators {

class SignalSpectrum;

/***
 * Visualizes an IQ signal tensor received on the 'input' port: time-domain (signal) and
 * frequency-domain (spectrum) views drawn with Dear ImGui. Without a running renderer it
 * consumes its input and does nothing else.
 *
 * Input:
 *  'input' - Float tensor of interleaved IQ samples (I0Q0 I1Q1 ...).
 */
class SignalViewerOp : public holoscan::Operator<> {
public:
    explicit SignalViewerOp(std::shared_ptr<hololink::ImGuiRenderer> renderer = nullptr);
    ~SignalViewerOp();

    void setup(holoscan::OperatorSpec& spec) override;
    [[nodiscard]] holoscan::Contract contract() const override;
    void start() override;
    void stop() override;
    holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override;

    holoscan::Input<holoscan::Tensor> input { "input" };

private:
    static constexpr int number_of_signal_components = 2;

    // What the GUI asks compute() to stage; written by the GUI, read by compute(), under mutex_.
    struct Settings {
        bool show_signal = false;
        int signal_from = 0; // plotted time window [signal_from, signal_to), in IQ samples
        int signal_to = 128;
        bool show_spectrum = true;
        int spectrum_size = 1024;
        int spectrum_output = 0; // index into the "Power"/"Amplitude" choices
    };
    // What one staged batch holds; travels with it as its tag.
    struct Layout {
        int signal_from = 0;
        int signal_samples = 0;
        int spectrum_size = 0;
        int total_samples = 0;
    };

    void draw();
    void draw_signal(Settings& settings);
    void draw_spectrum(Settings& settings);

    std::shared_ptr<hololink::ImGuiRenderer> renderer_;
    std::optional<ImGuiRenderer::Handle> renderer_handle_;
    std::mutex mutex_;
    Settings settings_;
    std::unique_ptr<hololink::GuiStaging> staging_;
    std::unique_ptr<SignalSpectrum> spectrum_; // compute() only

    // GUI thread only
    Layout layout_;
    std::vector<float> batch_;
    std::vector<float> signal_samples_;
    std::vector<float> spectrum_samples_;
    float max_value_ = 1.0f;
    int spectrum_from_ = 0;
    int spectrum_to_ = 512;
};

} // namespace hololink::operators

#endif /* OPERATORS_SIG_VIEWER_SIG_VIEWER_OP_HPP */
