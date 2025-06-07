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

#ifndef OPERATORS_SIG_VIEWER_SIG_VIEWER_OP_HPP
#define OPERATORS_SIG_VIEWER_SIG_VIEWER_OP_HPP

#include <array>
#include <queue>
#include <string>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <holoscan/holoscan.hpp>

#include <hololink/common/gui_renderer.hpp>
#include <hololink/core/smart_object_pool.hpp>

namespace hololink::operators {

/***
 * The IQEncoderOp operator encodes in-phase and quadrature signal components
 * and creates a signal buffer.
 *
 * Input:
 *  'input' - The the signal to be viewed.
 *         The signal is expected to be an array of floats
 *         while each float represents a signal sample.
 *
 * Parameters:
 *  'renderer' - Since Dear ImGui is not supported by Holoviz operators,
 *               a pointer to an ImGuiRenderer object is required.
 */

class SignalViewerOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(SignalViewerOp)

    SignalViewerOp() = default;

    void initialize() override;
    void setup(holoscan::OperatorSpec& spec) override;

    void start() override;
    void stop() override;

    void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
        holoscan::ExecutionContext& context) override;
    using Samples = std::vector<float>;

private:
    static constexpr unsigned number_of_signal_components = 2;
    static void on_input_tensors_ready(void* data);

    std::shared_ptr<holoscan::UnboundedAllocator> allocator_;
    std::mutex mutex_;
    struct SignalView {
        SignalView();
        bool show_ = false;
        bool force_update_ = true;
        int from_ = 0;
        int to_ = 128;
        int max_count_ = 0; // Number of IQ pairs in the tensor
        std::shared_ptr<holoscan::Tensor> tensor_;
        float max_value_ = 1.0f;
        Samples samples_;

        struct SignalComponent {
            SignalComponent(const std::string& name = "");
            void draw_show_checkbox();

            std::string name_;
            bool show_plot_ = true;
        };
        using Signal = std::array<SignalComponent, number_of_signal_components>;
        Signal signal_;
        void draw(const std::shared_ptr<holoscan::Tensor>& tensor, cudaStream_t stream);
    };
    SignalView signal_view_;
    struct SpectrumView {
        bool show_ = true;
        std::string label_ = "Spectrum";
        int size_ = 1024;
        int from_ = 0;
        int to_ = size_ / 2;
        int output_type_index_ = 0;
        std::shared_ptr<holoscan::Tensor> tensor_iq_;

        enum class OutputType {
            Power,
            Amplitude
        };

        /**
         * The spectrum class can calculate spectrum from signal IQ pairs
         */
        class Spectrum {
        public:
            // The size of the spectrum (the number of sample pairs to use)
            Spectrum(int64_t count);
            ~Spectrum();

            int64_t get_count() const;

            void calculate(const float* device_samples_iq, OutputType output_type, cudaStream_t stream = 0);
            const thrust::device_vector<float>& get_data() const;

        private:
            struct Impl;
            std::unique_ptr<Impl> pimpl_;
        };
        std::shared_ptr<Spectrum> device_spectrum_;
        Samples host_spectrum_;

        void draw(const std::shared_ptr<holoscan::Tensor>& tensor_iq, cudaStream_t stream);
        static bool type_combo_handler(void* data, int idx, const char** out_text);
        static const std::vector<std::pair<std::string, OutputType>> output_types_;
    };
    SpectrumView spectrum_view_;

    ImGuiRenderer::Handle renderer_handle_;
    cudaStream_t stream_ {};

    using Tensor = std::shared_ptr<holoscan::Tensor>;
    using TensorsQueue = std::queue<Tensor>;
    TensorsQueue input_pending_tensor_queue_;
    Tensor input_ready_tensor_;

    static std::pair<float, float> cuda_minmax(const float* data, size_t size, cudaStream_t stream = 0);

    // Parameters
    holoscan::Parameter<hololink::ImGuiRenderer*> renderer_;
};

} // namespace hololink::operators

#endif /* OPERATORS_SIG_VIEWER_SIG_VIEWER_OP_HPP */
