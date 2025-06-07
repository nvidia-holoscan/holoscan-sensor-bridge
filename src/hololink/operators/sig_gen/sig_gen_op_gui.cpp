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

#include "sig_gen_op_gui.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <imgui.h>

#include <hololink/common/cuda_error.hpp>

namespace hololink::operators {

constexpr size_t plot_width = 1024;

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

struct SignalGeneratorOp::GUI::OnTensorReadyData {
    SignalGeneratorOp::GUI* self_;
    SignalGeneratorOp::Tensor tensor_;
};

void SignalGeneratorOp::GUI::on_tensor_ready(void* data)
{
    auto on_tensor_ready_data = static_cast<OnTensorReadyData*>(data);
    auto self = on_tensor_ready_data->self_;
    std::lock_guard<std::mutex> lock(self->mutex_);
    auto& tensor = on_tensor_ready_data->tensor_;
    if (tensor != self->pending_tensor_queue_.front()) {
        HSB_LOG_ERROR("Synchronization error");
        TensorQueue empty_queue;
        std::swap(self->pending_tensor_queue_, empty_queue);
    } else {
        self->ready_tensor_ = std::move(self->pending_tensor_queue_.front());
        self->pending_tensor_queue_.pop();
    }
    delete on_tensor_ready_data;
}

SignalGeneratorOp::GUI::GUI(hololink::ImGuiRenderer* renderer, const std::string& name,
    const std::string& expression_in_phase, const std::string& expression_quadrature,
    unsigned samples_count, const Rational& sampling_interval)
    : signal_ {
        SignalComponent("In-Phase", expression_in_phase),
        SignalComponent("Quadrature", expression_quadrature)
    }
    , renderer_(renderer)
    , samples_count_(samples_count)
    , sampling_interval_(sampling_interval)
    , renderer_handle_(renderer_->add_draw_function(name, [this] {
        std::unique_lock<std::mutex> lock(mutex_);
        Tensor ready_tensor = std::move(ready_tensor_);
        lock.unlock();

        if (ready_tensor || force_update_) {
            if (ready_tensor)
                current_tensor_ = ready_tensor;

            if (current_tensor_) {
                auto size = number_of_signal_components * (to_ - from_);
                samples_.resize(size);

                auto device_src = reinterpret_cast<float*>(current_tensor_->data()) + number_of_signal_components * from_;
                CUDA_CHECK(cudaMemcpyAsync(&samples_.front(), device_src, size * sizeof(float), cudaMemcpyDeviceToHost, stream_.handle_));
                auto minmax = cuda_minmax(device_src, size, stream_.handle_);
                CUDA_CHECK(cudaStreamSynchronize(stream_.handle_));
                max_value_ = std::max(std::abs(minmax.first), std::abs(minmax.second));
                force_update_ = false;
            }
        }

        ImGui::LabelText("Samples Count", "%d", samples_count_);
        ImGui::Text("Sampling Interval");
        ImGui::SameLine();
        lock.lock();
        if (ImGui::InputInt("/", &sampling_interval_.num_, 1, 0))
            sampling_interval_.num_ = std::max(sampling_interval_.num_, 0);
        ImGui::SameLine();
        if (ImGui::InputInt("(Rational Number)", &sampling_interval_.den_, 1, 0, ImGuiInputTextFlags_EnterReturnsTrue))
            sampling_interval_.den_ = std::max(sampling_interval_.den_, 1);
        lock.unlock();
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
    }))
{
}

void SignalGeneratorOp::GUI::draw_plots()
{
    if (samples_.empty())
        return;

    auto from = from_;
    auto to = to_;

    ImGui::PushID(&samples_.front());

    ImGui::PushID(&from_);
    ImGui::SetNextItemWidth(42);
    ImGui::InputScalar("", ImGuiDataType_U32, &from, nullptr, nullptr, "%d", ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::PopID();

    ImGui::SameLine();
    ImGui::BeginGroup();
    for (size_t plot_index = 0; plot_index < number_of_signal_components; ++plot_index)
        if (signal_[plot_index].show_plot_)
            ImGui::PlotLines("",
                &samples_.front() + plot_index, // Offset to get the right plot data
                to_ - from_,
                from_,
                signal_[plot_index].name_.c_str(),
                -max_value_,
                max_value_,
                ImVec2(plot_width - 124, 128),
                number_of_signal_components * sizeof(float)); // Stride
    ImGui::EndGroup();
    ImGui::SameLine();

    ImGui::PushID(&to_);
    ImGui::SetNextItemWidth(42);
    ImGui::InputScalar("", ImGuiDataType_U32, &to, nullptr, nullptr, "%d", ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::PopID();

    ImGui::PopID();

    if (from < 0)
        from = from_;
    if (to > current_tensor_->size())
        to = to_;
    if (from >= to)
        from = from_;
    if (to <= from)
        to = to_;
    if (to != to_ || from != from)
        force_update_ = true;
    from_ = from;
    to_ = to;
}

SignalGeneratorOp::GUI::~GUI()
{
    renderer_->remove_draw_function(renderer_handle_);
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

void SignalGeneratorOp::GUI::push_samples(const Tensor& tensor)
{
    std::lock_guard<std::mutex> lock(mutex_);
    pending_tensor_queue_.emplace(tensor);
    CUDA_CHECK(cudaLaunchHostFunc(0, &on_tensor_ready, new OnTensorReadyData { this, tensor }));
}

} // namespace hololink::operators
