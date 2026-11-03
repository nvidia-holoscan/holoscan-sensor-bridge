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
#include "sig_viewer_op.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>

#include <holoviz/imgui/ui.hpp>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>

#include <hololink/common/cuda_error.hpp>
#include <hololink/common/gui_staging.hpp>
#include <hololink/core/logging_internal.hpp>

#include "sig_viewer_spectrum.hpp"

namespace hololink::operators {
namespace {
    constexpr float plot_width = 1024;
    const char* const signal_component_names[] = { "In-Phase", "Quadrature" };
    const char* const spectrum_output_names = "Power\0Amplitude\0";
}

SignalViewerOp::SignalViewerOp(std::shared_ptr<hololink::ImGuiRenderer> renderer)
    : renderer_(std::move(renderer))
{
}

SignalViewerOp::~SignalViewerOp() = default;

void SignalViewerOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input(input, "input");
}

holoscan::Contract SignalViewerOp::contract() const
{
    holoscan::Contract contract;
    contract.trigger(holoscan::OnEach { input });
    return contract;
}

void SignalViewerOp::start()
{
    if (!renderer_)
        return;
    staging_ = std::make_unique<hololink::GuiStaging>();
    renderer_handle_ = renderer_->add_draw_function("Signal Viewer", [this] { draw(); });
}

void SignalViewerOp::stop()
{
    // Synchronous: no draw() runs after this, so the staging and spectrum can go.
    if (renderer_handle_) {
        renderer_->remove_draw_function(*renderer_handle_);
        renderer_handle_.reset();
    }
    staging_.reset();
    spectrum_.reset();
}

holoscan::expected<void, holoscan::Error> SignalViewerOp::compute(holoscan::ExecutionContext& context)
{
    auto maybe_sample = input.receive();
    if (!maybe_sample) {
        return holoscan::make_unexpected(std::move(maybe_sample).error());
    }
    const holoscan::Tensor& tensor = maybe_sample->data;
    if (tensor.size() % number_of_signal_components)
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kInvalidArgument, "Invalid IQ signal" });

    if (!renderer_ || !renderer_->is_running())
        return {};

    Settings settings;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        settings = settings_;
    }
    Layout layout { .total_samples = static_cast<int>(tensor.size() / number_of_signal_components) };
    if (settings.show_signal) {
        layout.signal_from = std::clamp(settings.signal_from, 0, layout.total_samples);
        layout.signal_samples = std::clamp(settings.signal_to, layout.signal_from, layout.total_samples)
            - layout.signal_from;
    }
    if (settings.show_spectrum) {
        const int size = std::min(settings.spectrum_size, layout.total_samples);
        layout.spectrum_size = (size >= 2) ? size : 0;
    }
    const size_t signal_floats = size_t(layout.signal_samples) * number_of_signal_components;
    const size_t floats = signal_floats + layout.spectrum_size;
    if (!floats)
        return {};

    // Everything below runs on the runtime's stream, so it's ordered after this input's
    // producer and before the input is released.
    const cudaStream_t stream = context.cuda_stream().get();
    const float* samples = static_cast<const float*>(tensor.data());
    staging_->stage(
        floats,
        [&](float* host) {
            if (signal_floats) {
                CUDA_CHECK(cudaMemcpyAsync(host,
                    samples + size_t(layout.signal_from) * number_of_signal_components,
                    signal_floats * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    stream));
            }
            if (layout.spectrum_size) {
                // Only staging when the GUI took the last batch means no earlier spectrum work is
                // still in flight, so replacing the spectrum here is safe.
                if (!spectrum_ || (spectrum_->count() != layout.spectrum_size))
                    spectrum_ = std::make_unique<SignalSpectrum>(layout.spectrum_size);
                spectrum_->calculate(samples, static_cast<SpectrumOutput>(settings.spectrum_output), stream);
                CUDA_CHECK(cudaMemcpyAsync(host + signal_floats,
                    spectrum_->data(),
                    layout.spectrum_size * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    stream));
            }
        },
        stream,
        { layout.signal_from, layout.signal_samples, layout.spectrum_size, layout.total_samples });
    return {};
}

void SignalViewerOp::draw()
{
    // Pick up the latest staged batch, if one landed.
    std::vector<int> tag;
    if (staging_->take(batch_, &tag)) {
        layout_ = Layout { tag[0], tag[1], tag[2], tag[3] };
        const auto signal_end = batch_.begin() + layout_.signal_samples * number_of_signal_components;
        signal_samples_.assign(batch_.begin(), signal_end);
        spectrum_samples_.assign(signal_end, signal_end + layout_.spectrum_size);
        if (!signal_samples_.empty()) {
            auto minmax = std::minmax_element(signal_samples_.begin(), signal_samples_.end());
            max_value_ = std::max(std::abs(*minmax.first), std::abs(*minmax.second));
        }
    }

    Settings settings;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        settings = settings_;
    }
    ImGui::Checkbox("Show Time Domain", &settings.show_signal);
    ImGui::Checkbox("Show Frequency Domain", &settings.show_spectrum);
    if (settings.show_signal)
        draw_signal(settings);
    if (settings.show_spectrum)
        draw_spectrum(settings);
    std::lock_guard<std::mutex> lock(mutex_);
    settings_ = settings;
}

void SignalViewerOp::draw_signal(Settings& settings)
{
    if (signal_samples_.empty())
        return;

    // The plotted window takes effect with the next staged batch.
    int from = settings.signal_from;
    int to = settings.signal_to;

    ImGui::PushID("signal");

    ImGui::SetNextItemWidth(42);
    ImGui::InputScalar("##from", ImGuiDataType_S32, &from, nullptr, nullptr, "%d", ImGuiInputTextFlags_EnterReturnsTrue);

    ImGui::SameLine();
    ImGui::BeginGroup();
    for (int component_index = 0; component_index < number_of_signal_components; ++component_index)
        ImGui::PlotLines("",
            signal_samples_.data() + component_index, // Offset to get the right plot data
            layout_.signal_samples,
            0,
            signal_component_names[component_index],
            -max_value_,
            max_value_,
            ImVec2(plot_width - 124, 128),
            number_of_signal_components * sizeof(float)); // Stride
    ImGui::EndGroup();
    ImGui::SameLine();

    ImGui::SetNextItemWidth(42);
    ImGui::InputScalar("##to", ImGuiDataType_S32, &to, nullptr, nullptr, "%d", ImGuiInputTextFlags_EnterReturnsTrue);

    ImGui::PopID();

    if ((from >= 0) && (from < to) && (to <= layout_.total_samples)) {
        settings.signal_from = from;
        settings.signal_to = to;
    }
}

void SignalViewerOp::draw_spectrum(Settings& settings)
{
    int size = settings.spectrum_size;
    ImGui::InputInt("Spectrum size", &size, 1, 0, ImGuiInputTextFlags_EnterReturnsTrue);
    size = std::max(size, 2);
    if (size != settings.spectrum_size) {
        settings.spectrum_size = size;
        spectrum_from_ = 0;
        spectrum_to_ = size / 2; // Only half of the spectrum is viewed
    }

    ImGui::SameLine();
    ImGui::Combo("Spectrum Output", &settings.spectrum_output, spectrum_output_names);

    const int count = static_cast<int>(spectrum_samples_.size());
    if (!count)
        return;
    // A batch staged with a different size than requested shows until the next one lands.
    spectrum_to_ = std::min(spectrum_to_, count);
    spectrum_from_ = std::min(spectrum_from_, spectrum_to_ - 1);
    int from = spectrum_from_;
    int to = spectrum_to_;

    ImGui::PushID("spectrum");

    ImGui::SetNextItemWidth(42);
    ImGui::InputScalar("##from", ImGuiDataType_S32, &from, nullptr, nullptr, "%d", ImGuiInputTextFlags_EnterReturnsTrue);

    ImGui::SameLine();

    ImGui::PlotLines("",
        spectrum_samples_.data() + spectrum_from_,
        spectrum_to_ - spectrum_from_,
        0,
        "Spectrum",
        0,
        FLT_MAX,
        ImVec2(plot_width - 124, 128));

    ImGui::SameLine();

    ImGui::SetNextItemWidth(42);
    ImGui::InputScalar("##to", ImGuiDataType_S32, &to, nullptr, nullptr, "%d", ImGuiInputTextFlags_EnterReturnsTrue);

    ImGui::PopID();

    if ((from >= 0) && (from < to) && (to <= count)) {
        spectrum_from_ = from;
        spectrum_to_ = to;
    }
}

} // namespace hololink::operators
