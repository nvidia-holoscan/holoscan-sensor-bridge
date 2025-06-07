/**
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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
 *
 * See README.md for detailed information.
 */

#include "audio_packetizer.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <zlib.h>

#include <fstream>
#include <string>
#include <vector>

namespace hololink::operators {

void AudioPacketizerOp::setup(holoscan::OperatorSpec& spec)
{
    spec.output<holoscan::gxf::Entity>("out");

    spec.param(wav_file_, "wav_file", "WAV File",
        "Path to the WAV file to transmit", std::string(""));
    spec.param(chunk_size_, "chunk_size", "Chunk Size",
        "Size of audio chunks to transmit", 1024u);
    spec.param(pool_, "pool", "Pool",
        "Allocator used to allocate tensor output");
}

void AudioPacketizerOp::start()
{
    // Load WAV file
    load_wav_file();

    packet_buffer_.clear();
    packet_buffer_.resize(chunk_size_.get() + 16);
    HOLOSCAN_LOG_INFO("AudioSendOp started with WAV file: {}", wav_file_.get());
    HOLOSCAN_LOG_INFO("  Sample Rate: {} Hz", wav_header_.sample_rate);
    HOLOSCAN_LOG_INFO("  Channels: {}", wav_header_.num_channels);
    HOLOSCAN_LOG_INFO("  Bits per Sample: {}", wav_header_.bit_depth);
}

void AudioPacketizerOp::compute(holoscan::InputContext& op_input,
    holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& context)
{
    // Calculate chunk size for this iteration
    size_t chunk = chunk_size_.get();
    const size_t ib_header_size = 12; // 3 * 4 bytes for the IB header
    const size_t crc_size = 4; // 4 bytes for CRC
    const size_t payload_size = std::min(chunk, audio_data_.size() - current_pos_);
    const size_t total_size = ib_header_size + chunk + crc_size;

    HOLOSCAN_LOG_DEBUG("Audio chunk size: {} samples, {} channels, {} bytes total",
        chunk,
        wav_header_.num_channels,
        total_size);

    // Create message entity
    auto message = holoscan::gxf::Entity::New(&context);
    if (!message) {
        HOLOSCAN_LOG_ERROR("Failed to create GXF entity");
        return;
    }

    // Get tensor context from pool
    std::shared_ptr<holoscan::DLManagedTensorContext> dl_managed_tensor_context = dl_managed_tensor_context_pool_.acquire();

    if (!dl_managed_tensor_context) {
        nvidia::gxf::Tensor gxf_tensor;
        auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
            context.context(),
            pool_->gxf_cid());

        // Reshape tensor to include space for IB header
        if (!gxf_tensor.reshape<uint8_t>( // Use uint8_t to handle both header and data
                nvidia::gxf::Shape { static_cast<int>(total_size) },
                nvidia::gxf::MemoryStorageType::kDevice,
                allocator.value())) {
            HOLOSCAN_LOG_ERROR("Failed to allocate CUDA memory");
            return;
        }

        // Convert to DLManagedTensorContext
        auto maybe_dl_ctx = gxf_tensor.toDLManagedTensorContext();
        if (!maybe_dl_ctx) {
            HOLOSCAN_LOG_ERROR("Failed to get DLManagedTensorContext from GXF tensor");
            return;
        }
        dl_managed_tensor_context = dl_managed_tensor_context_pool_.emplace(
            std::move(*maybe_dl_ctx.value()));
    }

    // Create tensor from context
    auto tensor = std::make_shared<holoscan::Tensor>(dl_managed_tensor_context);

    // Prepare IB header
    unsigned int opcode_n_flags = htonl(0x24 << 24 | 0x40FFFF);
    unsigned int fb_n_qp = htonl(0 | (2 & 0xFFFFFF));
    unsigned int a_n_psn = htonl(0 | (num_of_packets_ & 0xFFFFFF));

    memset(packet_buffer_.data(), 0, total_size);

    // Copy header to packet buffer
    memcpy(packet_buffer_.data(), &opcode_n_flags, 4);
    memcpy(packet_buffer_.data() + 4, &fb_n_qp, 4);
    memcpy(packet_buffer_.data() + 8, &a_n_psn, 4);

    // Copy audio data to packet buffer
    memcpy(packet_buffer_.data() + ib_header_size,
        audio_data_.data() + current_pos_,
        payload_size);

    // Zero-fill if needed
    if (payload_size < chunk) {
        memset(packet_buffer_.data() + ib_header_size + payload_size, 0, chunk - payload_size);
    }

    // Calculate and add CRC
    unsigned int crc = htonl(crc32(0xdebb20e3,
        packet_buffer_.data() + ib_header_size, // Calculate CRC of payload only
        chunk));
    memcpy(packet_buffer_.data() + ib_header_size + chunk, &crc, crc_size);

    // Copy complete packet to GPU
    CUDA_CHECK(cudaMemcpy(
        tensor->data(),
        packet_buffer_.data(),
        total_size,
        cudaMemcpyHostToDevice));

    // Emit the message
    op_output.emit(tensor);

    // Update position
    current_pos_ += payload_size;
    if (current_pos_ >= audio_data_.size()) {
        current_pos_ = 0; // Loop back to start
        HOLOSCAN_LOG_DEBUG("Looping back to start of audio file");
    }
    num_of_packets_++;
}

void AudioPacketizerOp::stop()
{
    // Clear audio data
    audio_data_.clear();
    audio_data_.shrink_to_fit();

    HOLOSCAN_LOG_INFO("AudioSendOp stopped and resources cleaned up");
}

void AudioPacketizerOp::load_wav_file()
{

    std::ifstream file(wav_file_.get(), std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open WAV file: " + wav_file_.get());
    }

    // Read WAV header
    file.read(reinterpret_cast<char*>(&wav_header_), sizeof(WAVHeader));

    // Validate WAV format
    if (std::strncmp(wav_header_.riff_header, "RIFF", 4) != 0 || std::strncmp(wav_header_.wave_header, "WAVE", 4) != 0 || std::strncmp(wav_header_.fmt_header, "fmt ", 4) != 0 || std::strncmp(wav_header_.data, "data", 4) != 0) {
        throw std::runtime_error("Invalid WAV format");
    }

    uint32_t num_bytes_per_sample = wav_header_.bit_depth / 8;

    // Calculate sizes
    size_t samples_per_channel = wav_header_.data_size / (num_bytes_per_sample * wav_header_.num_channels);
    size_t output_size = samples_per_channel * 4 * wav_header_.num_channels; // 4 bytes per sample with padding
    audio_data_.clear();
    audio_data_.resize(output_size);
    memset(audio_data_.data(), 0, output_size);

    // Read all input data
    std::vector<uint8_t> input_data(wav_header_.data_size);
    memset(input_data.data(), 0, wav_header_.data_size);
    if (!file.read(reinterpret_cast<char*>(input_data.data()), wav_header_.data_size)) {
        throw std::runtime_error("Failed to read WAV file data");
    }

    size_t input_pos = 0; // Position in input data
    size_t output_pos = 0; // Position in output data
    // Process each sample pair (left and right channels)
    for (size_t i = 0; i < samples_per_channel; i++) {
        // Process each channel
        for (size_t channel = 0; channel < wav_header_.num_channels; channel++) {
            // Copy 3 bytes of 24-bit sample
            audio_data_[output_pos++] = input_data[input_pos++]; // First byte
            audio_data_[output_pos++] = input_data[input_pos++]; // Second byte
            audio_data_[output_pos++] = input_data[input_pos++]; // Third byte
            audio_data_[output_pos++] = 0x00; // Add zero padding as fourth byte
        }
    }

    HOLOSCAN_LOG_INFO("Loaded WAV file: {} samples per channel, {} channels, {} Hz, 24-bit",
        samples_per_channel,
        wav_header_.num_channels,
        wav_header_.sample_rate);

    return;
}

} // namespace hololink::operators
