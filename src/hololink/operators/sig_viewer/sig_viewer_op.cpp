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
#include "sig_viewer_op.hpp"

#include <cmath>

#include <cuda.h>

#include <imgui.h>

#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoviz/holoviz.hpp>

#include <hololink/common/cuda_error.hpp>
#include <hololink/core/logging_internal.hpp>

namespace hololink::operators {
namespace {
    constexpr size_t plot_width = 1024;
    static const char* input_name = "input";
}

SignalViewerOp::SignalView::SignalView()
    : signal_ {
        SignalComponent("In-Phase"),
        SignalComponent("Quadrature")
    }
{
}

SignalViewerOp::SignalView::SignalComponent::SignalComponent(const std::string& name)
    : name_(name)
{
}

void SignalViewerOp::start()
{
    CUDA_CHECK(cudaStreamCreate(&stream_));
    if (renderer_.has_value() && renderer_.get())
        renderer_handle_ = renderer_->add_draw_function(name(), [this] {
            std::unique_lock<std::mutex> lock(mutex_);
            auto tensor_iq = std::move(input_ready_tensor_);
            lock.unlock();

            ImGui::Checkbox("Show Time Domain", &signal_view_.show_);
            ImGui::Checkbox("Show Frequency Domain", &spectrum_view_.show_);

            signal_view_.draw(tensor_iq, stream_);
            spectrum_view_.draw(tensor_iq, stream_);
        });
}

void SignalViewerOp::stop()
{
    if (renderer_.has_value() && renderer_.get())
        renderer_->remove_draw_function(renderer_handle_);
    CUDA_CHECK(cudaStreamDestroy(stream_));
}

void SignalViewerOp::initialize()
{
    // Create an allocator for the operator
    allocator_ = fragment()->make_resource<holoscan::UnboundedAllocator>("pool");
    // Add the allocator to the operator so that it is initialized
    add_arg(allocator_);

    // Call the base class initialize function
    Operator::initialize();
}

void SignalViewerOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input<Tensor>(input_name);

    /// Register converters for arguments not defined by Holoscan
    register_converter<hololink::ImGuiRenderer*>();

    spec.param(renderer_, "renderer", "ImGuiRenderer", "Pointer to ImGuiRenderer object");
}

struct OnInputTensorReadyData {
    SignalViewerOp* self_;
    std::shared_ptr<holoscan::Tensor> tensor_iq;
};

void SignalViewerOp::on_input_tensors_ready(void* data)
{
    auto on_input_tensor_ready_data = static_cast<OnInputTensorReadyData*>(data);
    auto self = on_input_tensor_ready_data->self_;
    std::lock_guard<std::mutex> lock(self->mutex_);
    if (on_input_tensor_ready_data->tensor_iq != self->input_pending_tensor_queue_.front()) {
        HSB_LOG_ERROR("Synchronization error");
        TensorsQueue empty_queue;
        std::swap(self->input_pending_tensor_queue_, empty_queue);
    } else {
        self->input_ready_tensor_ = std::move(self->input_pending_tensor_queue_.front());
        self->input_pending_tensor_queue_.pop();
    }
    delete on_input_tensor_ready_data;
}

void SignalViewerOp::compute(holoscan::InputContext& op_input, [[maybe_unused]] holoscan::OutputContext& op_output,
    [[maybe_unused]] holoscan::ExecutionContext& context)
{
    auto maybe_tensor_iq = op_input.receive<Tensor>(input_name);
    if (!maybe_tensor_iq) {
        HSB_LOG_ERROR("Failed to receive message from port 'in'");
        return;
    }

    auto tensor_iq = maybe_tensor_iq.value();
    if (tensor_iq->size() % number_of_signal_components)
        throw std::runtime_error("Invalid IQ signal");

    // If renderer is not enabled, or not running, return here
    if (renderer_.has_value() && renderer_.get() && renderer_->is_running()) {
        std::lock_guard<std::mutex> lock(mutex_);
        input_pending_tensor_queue_.emplace(tensor_iq);
        cudaLaunchHostFunc(0, &on_input_tensors_ready, new OnInputTensorReadyData { this, std::move(tensor_iq) });
    }
}

void SignalViewerOp::SignalView::draw(const std::shared_ptr<holoscan::Tensor>& tensor, cudaStream_t stream)
{
    if (!show_)
        return;
    // Update the data
    if (tensor || force_update_) {
        if (tensor)
            tensor_ = tensor;

        if (tensor_) {
            auto size = (to_ - from_) * number_of_signal_components;
            samples_.resize(size);

            CUDA_CHECK(cudaMemcpyAsync(
                &samples_.front(),
                reinterpret_cast<float*>(tensor_->data()) + from_ * number_of_signal_components,
                size * sizeof(float),
                cudaMemcpyDeviceToHost,
                stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto minmax = std::minmax_element(samples_.begin(), samples_.end());
            max_value_ = std::max(std::abs(*minmax.first), std::abs(*minmax.second));
            max_count_ = tensor_->size() / number_of_signal_components;
            force_update_ = false;
        }
    }
    // Nothing to draw
    if (samples_.empty())
        return;

    // Draw
    auto from = from_;
    auto to = to_;

    ImGui::PushID(&samples_.front());

    ImGui::PushID(&from_);
    ImGui::SetNextItemWidth(42);
    ImGui::InputScalar("", ImGuiDataType_U32, &from, nullptr, nullptr, "%d", ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::PopID();

    ImGui::SameLine();
    ImGui::BeginGroup();
    for (size_t component_index = 0; component_index < number_of_signal_components; ++component_index)
        ImGui::PlotLines("",
            &samples_.front() + component_index, // Offset to get the right plot data
            to_ - from_,
            0,
            signal_[component_index].name_.c_str(),
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
    if (to > max_count_)
        to = to_;
    if (from >= to)
        from = from_;
    if (to <= from)
        to = to_;
    if (to != to_ || from != from_)
        force_update_ = true;
    from_ = from;
    to_ = to;
}

const std::vector<std::pair<std::string, SignalViewerOp::SpectrumView::OutputType>> SignalViewerOp::SpectrumView::output_types_ {
    { "Power", OutputType::Power },
    { "Amplitude", OutputType::Amplitude }
};

bool SignalViewerOp::SpectrumView::type_combo_handler(void* data, int idx, const char** out_text)
{
    [[maybe_unused]] auto self = static_cast<SpectrumView*>(data);
    *out_text = output_types_[idx].first.c_str();
    return true;
}

void SignalViewerOp::SpectrumView::draw(const std::shared_ptr<holoscan::Tensor>& tensor_iq, cudaStream_t stream)
{
    if (!show_)
        return;

    int size = size_;
    ImGui::InputInt("Spectrum size", &size, 1, 0, ImGuiInputTextFlags_EnterReturnsTrue);
    if (size <= 1)
        size = 2;
    if (size_ != size) {
        size_ = size;
        from_ = 0;
        to_ = size_ / 2; // Only half of the spectrum is viewed
    }

    ImGui::SameLine();
    ImGui::Combo("Spectrum Output", &output_type_index_, &SignalViewerOp::SpectrumView::type_combo_handler, this, output_types_.size());

    // Time to update the lines
    if (tensor_iq || host_spectrum_.size() != size_) {
        // The requested spectrum size changed
        if (host_spectrum_.size() != size_)
            host_spectrum_.resize(size_, 0);

        // The input tensors changed
        if (tensor_iq) {
            tensor_iq_ = tensor_iq;
        }
        // Update the spectrum (only if tensors are available)
        if (tensor_iq_) {
            if (!device_spectrum_ || device_spectrum_->get_count() != size_)
                device_spectrum_.reset(new Spectrum(size_));
            device_spectrum_->calculate(
                reinterpret_cast<const float*>(tensor_iq_->data()),
                output_types_[output_type_index_].second,
                stream);
        }

        // If spectrum is available, copy it to the Host memory
        if (device_spectrum_ && device_spectrum_->get_data().size()) {
            CUDA_CHECK(cudaMemcpyAsync(&host_spectrum_.front(), device_spectrum_->get_data().data().get(), size_ * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }

    if (host_spectrum_.empty())
        return;

    auto data = &host_spectrum_.front();
    auto from = from_;
    auto to = to_;

    ImGui::PushID(data);

    ImGui::PushID(&from);
    ImGui::SetNextItemWidth(42);
    ImGui::InputScalar("", ImGuiDataType_U32, &from, nullptr, nullptr, "%d", ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::PopID();

    ImGui::SameLine();

    ImGui::PlotLines("",
        data + from,
        to_ - from_,
        from_,
        label_.c_str(),
        0,
        FLT_MAX,
        ImVec2(plot_width - 124, 128));

    ImGui::SameLine();

    ImGui::PushID(&to);
    ImGui::SetNextItemWidth(42);
    ImGui::InputScalar("", ImGuiDataType_U32, &to, nullptr, nullptr, "%d", ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::PopID();

    ImGui::PopID();

    if (from < 0)
        from = from_;
    if (to > size)
        to = to_;
    if (from >= to)
        from = from_;
    if (to <= from)
        to = to_;
    from_ = from;
    to_ = to;
}

} // namespace hololink::operators
