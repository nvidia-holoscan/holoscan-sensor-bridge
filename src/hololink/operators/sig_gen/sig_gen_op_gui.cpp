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

#include "sig_gen_op_gui.hpp"

#include <algorithm>
#include <cmath>

#include <holoviz/imgui/ui.hpp>

#include <hololink/common/cuda_error.hpp>

namespace hololink::operators {

constexpr float plot_width = 1024;

SignalGeneratorOp::GUI::SignalComponent::SignalComponent(const std::string& name, const std::string& expression)
    : name_(name)
    , expression_(expression)
    , cut_copy_paste_(expression_)
{
}

SignalGeneratorOp::GUI::SignalComponent::SignalComponent(SignalComponent&& other)
    : SignalComponent()
{
    swap(*this, other);
}

SignalGeneratorOp::GUI::SignalComponent& SignalGeneratorOp::GUI::SignalComponent::operator=(SignalComponent&& other)
{
    swap(*this, other);
    return *this;
}

void SignalGeneratorOp::GUI::SignalComponent::draw_show_checkbox()
{
    ImGui::PushID(name_.c_str());
    ImGui::Checkbox("Show Plot", &show_plot_);
    ImGui::PopID();
}

void SignalGeneratorOp::GUI::SignalComponent::draw_expression_input_text(std::mutex& mutex)
{
    ImGui::PushID(name_.c_str());
    auto signal_expression = expression_;
    signal_expression.resize(1024, '\0');
    if (ImGui::InputText(name_.c_str(),
            &signal_expression.front(),
            signal_expression.size(),
            ImGuiInputTextFlags_CallbackAlways,
            &CutCopyPaste::InputTextCallback,
            &cut_copy_paste_)
        && expression_ != &signal_expression.front()
        // Cannot use ImGuiInputTextFlags_EnterReturnsTrue
        // because of the InputTextCallback
        && ImGui::IsKeyPressed(ImGuiKey_Enter)) {
        std::lock_guard<std::mutex> lock(mutex);
        expression_ = std::string(&signal_expression.front());
    }

    if (ImGui::BeginPopupContextItem()) {
        ImGui::Text("%s", name_.c_str());
        ImGui::Separator();
        std::lock_guard<std::mutex> lock(mutex);
        expression_ = std::string(&signal_expression.front());
        cut_copy_paste_(50);
        ImGui::EndPopup();
    }
    ImGui::PopID();
}

SignalGeneratorOp::GUI::GUI(std::shared_ptr<hololink::ImGuiRenderer> renderer, const std::string& name,
    const std::string& expression_in_phase, const std::string& expression_quadrature,
    unsigned samples_count, const Rational& sampling_interval)
    : samples_count_(samples_count)
    , sampling_interval_(sampling_interval)
    , to_(std::min(default_plotted_samples, static_cast<int>(samples_count)))
    , signal_ {
        SignalComponent("In-Phase", expression_in_phase),
        SignalComponent("Quadrature", expression_quadrature)
    }
    , renderer_(std::move(renderer))
    , renderer_handle_(renderer_->add_draw_function(name, [this] { draw(); }))
{
}

SignalGeneratorOp::GUI::~GUI()
{
    // Synchronous: no draw() runs after this.
    renderer_->remove_draw_function(renderer_handle_);
}

void SignalGeneratorOp::GUI::stage_samples(const float* device_samples, cudaStream_t stream)
{
    int from, to;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        from = from_;
        to = to_;
    }
    const size_t floats = size_t(to - from) * number_of_signal_components;
    if (!floats)
        return;
    staging_.stage(
        floats,
        [&](float* host) {
            CUDA_CHECK(cudaMemcpyAsync(host,
                device_samples + size_t(from) * number_of_signal_components,
                floats * sizeof(float),
                cudaMemcpyDeviceToHost,
                stream));
        },
        stream);
}

void SignalGeneratorOp::GUI::draw()
{
    if (staging_.take(samples_) && !samples_.empty()) {
        auto minmax = std::minmax_element(samples_.begin(), samples_.end());
        max_value_ = std::max(std::abs(*minmax.first), std::abs(*minmax.second));
    }

    ImGui::LabelText("Samples Count", "%u", samples_count_);
    ImGui::Text("Sampling Interval");
    ImGui::SameLine();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (ImGui::InputInt("/", &sampling_interval_.num_, 1, 0))
            sampling_interval_.num_ = std::max(sampling_interval_.num_, 0);
        ImGui::SameLine();
        if (ImGui::InputInt("(Rational Number)", &sampling_interval_.den_, 1, 0, ImGuiInputTextFlags_EnterReturnsTrue))
            sampling_interval_.den_ = std::max(sampling_interval_.den_, 1);
    }
    ImGui::Separator();

    ImGui::BeginGroup();
    for (auto& signal_component : signal_)
        signal_component.draw_expression_input_text(mutex_);
    ImGui::EndGroup();
    ImGui::SameLine();
    ImGui::BeginGroup();
    for (auto& signal_component : signal_)
        signal_component.draw_show_checkbox();
    ImGui::EndGroup();

    draw_plots();
}

void SignalGeneratorOp::GUI::draw_plots()
{
    if (samples_.empty())
        return;

    // The plotted window takes effect with the next staged batch.
    int from, to;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        from = from_;
        to = to_;
    }

    ImGui::PushID("plots");

    ImGui::SetNextItemWidth(42);
    ImGui::InputScalar("##from", ImGuiDataType_S32, &from, nullptr, nullptr, "%d", ImGuiInputTextFlags_EnterReturnsTrue);

    ImGui::SameLine();
    ImGui::BeginGroup();
    const int count = static_cast<int>(samples_.size() / number_of_signal_components);
    for (size_t plot_index = 0; plot_index < number_of_signal_components; ++plot_index)
        if (signal_[plot_index].show_plot_)
            ImGui::PlotLines("",
                samples_.data() + plot_index, // Offset to get the right plot data
                count,
                0,
                signal_[plot_index].name_.c_str(),
                -max_value_,
                max_value_,
                ImVec2(plot_width - 124, 128),
                number_of_signal_components * sizeof(float)); // Stride
    ImGui::EndGroup();
    ImGui::SameLine();

    ImGui::SetNextItemWidth(42);
    ImGui::InputScalar("##to", ImGuiDataType_S32, &to, nullptr, nullptr, "%d", ImGuiInputTextFlags_EnterReturnsTrue);

    ImGui::PopID();

    if ((from >= 0) && (from < to) && (to <= static_cast<int>(samples_count_))) {
        std::lock_guard<std::mutex> lock(mutex_);
        from_ = from;
        to_ = to;
    }
}

bool SignalGeneratorOp::GUI::is_running() const
{
    return renderer_->is_running();
}

std::unique_lock<std::mutex> SignalGeneratorOp::GUI::lock()
{
    return std::unique_lock<std::mutex>(mutex_);
}

const std::string& SignalGeneratorOp::GUI::get_signal_component_expression(SignalComponentType signal_component_type) const
{
    return signal_[static_cast<int>(signal_component_type)].expression_;
}

unsigned SignalGeneratorOp::GUI::get_samples_count() const
{
    return static_cast<unsigned>(samples_count_);
}

const Rational& SignalGeneratorOp::GUI::get_sampling_interval() const
{
    return sampling_interval_;
}

} // namespace hololink::operators
