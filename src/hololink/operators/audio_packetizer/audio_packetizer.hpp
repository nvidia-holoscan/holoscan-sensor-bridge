/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#ifndef SRC_HOLOLINK_OPERATORS_AUDIO_PACKETIZER_AUDIO_PACKETIZER_HPP
#define SRC_HOLOLINK_OPERATORS_AUDIO_PACKETIZER_AUDIO_PACKETIZER_HPP

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/core/gxf/entity.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>
#include <holoscan/holoscan.hpp>

#include <hololink/common/cuda_error.hpp>
#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/smart_object_pool.hpp>

namespace hololink::operators {

class AudioPacketizerOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(AudioPacketizerOp);

    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;
    void compute(holoscan::InputContext&,
        holoscan::OutputContext& op_output,
        holoscan::ExecutionContext&) override;
    void stop() override;

private:
    struct WAVHeader {
        char riff_header[4]; // Contains "RIFF"
        int32_t wav_size; // Size of WAV
        char wave_header[4]; // Contains "WAVE"
        char fmt_header[4]; // Contains "fmt "
        int32_t fmt_chunk_size;
        int16_t audio_format;
        int16_t num_channels;
        int32_t sample_rate;
        int32_t byte_rate;
        int16_t sample_alignment;
        int16_t bit_depth;
        char data[4];
        uint32_t data_size;
    };

    void load_wav_file();

    // Parameters
    holoscan::Parameter<std::string> wav_file_;
    holoscan::Parameter<uint32_t> chunk_size_;
    holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> pool_;

    // Internal state
    std::vector<char> audio_data_;
    std::vector<uint8_t> packet_buffer_;
    WAVHeader wav_header_;
    size_t current_pos_ { 0 };
    uint32_t num_of_packets_ { 0 };

    // Using SmartObjectPool for tensor context management
    using DLManagedTensorContextPool = SmartObjectPool<holoscan::DLManagedTensorContext>;
    DLManagedTensorContextPool dl_managed_tensor_context_pool_;
};

} // namespace hololink::operators

#endif // SRC_HOLOLINK_OPERATORS_AUDIO_PACKETIZER_AUDIO_PACKETIZER_HPP
