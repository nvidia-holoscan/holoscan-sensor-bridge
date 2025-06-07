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

#ifndef OPERATORS_SIG_GEN_SIG_GEN_OP_GUI_HPP
#define OPERATORS_SIG_GEN_SIG_GEN_OP_GUI_HPP

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <hololink/common/gui_renderer.hpp>

#include "sig_gen_op.hpp"

namespace hololink::operators {

using Samples = std::vector<float>;

/**
 * A wrapper class for Cuda Stream
 */
struct CudaStream {
    CudaStream()
    {
        CUDA_CHECK(cudaStreamCreate(&handle_));
    }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    ~CudaStream()
    try {
        CUDA_CHECK(cudaStreamDestroy(handle_));
    } catch (const std::exception&) {
        return;
    }

    cudaStream_t handle_;
};

/**
 * The GUI class is responsible for drawing the GUI for the SignalGeneratorOp
 */
class SignalGeneratorOp::GUI {
public:
    GUI(hololink::ImGuiRenderer* renderer, const std::string& name,
        const std::string& expression_in_phase, const std::string& expression_quadrature,
        unsigned samples_count, const Rational& sampling_interval);
    ~GUI();
    bool is_running() const;
    std::unique_lock<std::mutex> lock();
    const std::string& get_signal_component_expression(SignalComponentType signal_component_type) const;
    unsigned get_samples_count() const;
    const Rational& get_sampling_interval() const;

    using Tensor = std::shared_ptr<holoscan::Tensor>;
    void push_samples(const Tensor& tensor);

private:
    struct OnTensorReadyData;
    static void on_tensor_ready(void* data);

    mutable std::mutex mutex_;
    unsigned samples_count_;
    Rational sampling_interval_;
    CudaStream stream_ {};
    using TensorQueue = std::queue<Tensor>;
    TensorQueue pending_tensor_queue_;
    Tensor ready_tensor_;
    Tensor current_tensor_;
    int from_ = 0;
    int to_ = 128;
    float max_value_ = 1.0f;
    bool force_update_ = true;
    Samples samples_;
    void draw_plots();
    static constexpr unsigned number_of_signal_components = 2;
    struct SignalComponent {
        SignalComponent(const std::string& name = "", const std::string& expression = std::string());
        SignalComponent(const SignalComponent&) = delete;
        SignalComponent& operator=(const SignalComponent&) = delete;
        SignalComponent(SignalComponent&& other);
        SignalComponent& operator=(SignalComponent&& other);
        void draw_show_checkbox();
        void draw_expression_input_text(std::mutex& mutex);

        std::string name_;
        bool show_plot_ = true;
        std::string expression_;
        hololink::CutCopyPaste cut_copy_paste_;
        friend void swap(SignalComponent& lhs, SignalComponent& rhs) noexcept
        {
            using std::swap;
            swap(lhs.name_, rhs.name_);
            swap(lhs.show_plot_, rhs.show_plot_);
            swap(lhs.expression_, rhs.expression_);
        }
    };
    using Signal = std::array<SignalComponent, number_of_signal_components>;
    Signal signal_;
    hololink::ImGuiRenderer* renderer_;
    hololink::ImGuiRenderer::Handle renderer_handle_;

    static std::pair<float, float> cuda_minmax(const float* data, size_t size, cudaStream_t stream = 0);
};

} // namespace hololink::operators

#endif /* OPERATORS_SIG_GEN_SIG_GEN_OP_GUI_HPP */
