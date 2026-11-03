/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Template method bodies for ComputeCrcOp<PayloadT>.
//
// This file is NOT included from compute_crc.hpp. It is included by compute_crc.cpp
// alongside the explicit template instantiations for both supported PayloadT types, matching
// the base_receiver_op.hpp/.inl pattern used elsewhere in this codebase for a templated
// operator whose implementation is too large to want inline in the header:
//
//   #include "compute_crc.hpp"
//   #include "compute_crc.inl"
//   // ...
//   template class ComputeCrcOp<holoscan::Tensor>;
//   template class ComputeCrcOp<holoscan::schema::ImageT>;

#pragma once

#include <chrono>
#include <optional>
#include <stdexcept>

#include <nvcomp/crc32.h>

#include <hololink/common/cuda_error.hpp>
#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/networking.hpp>
#include <hololink/core/nvtx_trace.hpp>
#include <hololink/operators/capture_metadata_utils.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>

namespace hololink::operators {

template <typename PayloadT>
ComputeCrcOp<PayloadT>::ComputeCrcOp(
    std::shared_ptr<CrcChannel> crc_channel, int cuda_device_ordinal, uint64_t frame_size, bool emit_crc)
    : crc_channel_(std::move(crc_channel))
    , cuda_device_ordinal_(cuda_device_ordinal)
    , frame_size_(frame_size)
    , emit_crc_(emit_crc)
{
}

template <typename PayloadT>
void ComputeCrcOp<PayloadT>::setup(holoscan::OperatorSpec& spec)
{
    if (emit_crc_) {
        spec.output(output_crc, "crc");
    }
    // Input and output declaration names share one namespace per operator (Holoscan rejects
    // reusing a name across inputs/outputs/readiness sources), so the input ports here use an
    // "input_" prefix while the outputs keep the bare "tensor"/"metadata" names that downstream
    // consumers (add_flow, bind_tensor_output_device) wire against.
    if constexpr (std::is_same_v<PayloadT, holoscan::schema::ImageT>) {
        // A descriptor-schema port must declare which of its Tensor fields is the actual
        // materialized tensor -- the plain holoscan::Tensor ports below need no such call
        // (a Tensor port IS the tensor).
        (void)spec.input(input_tensor, "in_tensor").tensor_field(holoscan::schema::ImageTensorFields::data);
        spec.input(input_metadata, "in_metadata");
        (void)spec.output(output_tensor, "tensor").tensor_field(holoscan::schema::ImageTensorFields::data);
        spec.output(output_metadata, "metadata");
    } else {
        spec.input(input_tensor, "in_tensor");
        spec.input(input_metadata, "in_metadata");
        spec.output(output_tensor, "tensor");
        spec.output(output_metadata, "metadata");
    }
}

template <typename PayloadT>
holoscan::Contract ComputeCrcOp<PayloadT>::contract() const
{
    // Upstream stamps both ports with an identical capture_time, so OnAligned pairs them on an
    // exact match (see kMetadataPairingWindow).
    holoscan::Contract result;
    result
        .trigger(holoscan::OnAligned {
            .ports = holoscan::ports(input_tensor, input_metadata),
            .window = hololink::operators::kMetadataPairingWindow,
        })
        .input(input_tensor, holoscan::Exact {})
        .input(input_metadata, holoscan::Exact {});
    return result;
}

template <typename PayloadT>
void ComputeCrcOp<PayloadT>::start()
{
    if (!crc_channel_) {
        throw std::invalid_argument("ComputeCrcOp needs the CrcChannel its CheckCrcOp shares");
    }
    // A restarted run starts afresh.
    crc_channel_->clear();

    CUDA_CHECK(cudaSetDevice(cuda_device_ordinal_));

    int integrated = 0;
    CudaCheck(cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, cuda_device_ordinal_));
    is_integrated_ = (integrated != 0);

    // Initialize CRC options for nvcomp 5.0 API
    memset(&crc_opts_, 0, sizeof(crc_opts_));
    crc_opts_.spec = nvcompCRC32_JAMCRC; // Bit-inverted CRC32, matches existing behavior

    // Get heuristic configuration for optimal kernel settings.
    // Note that called this way, stream (0) isn't used.
    size_t num_chunks = 1;
    nvcompStatus_t status = nvcompBatchedCRC32GetHeuristicConf(
        nvcompCRC32IgnoredInputChunkBytes, num_chunks, &crc_opts_.kernel_conf, frame_size_, 0);
    if (status != nvcompSuccess) {
        throw std::runtime_error(
            fmt::format("nvcompBatchedCRC32GetHeuristicConf returned {}.", static_cast<int>(status)));
    }
}

template <typename PayloadT>
holoscan::expected<void, holoscan::Error> ComputeCrcOp<PayloadT>::compute(holoscan::ExecutionContext& context)
{
    // kNotReady is a normal, expected outcome from any call in compute_impl() below (a
    // clock-driven downstream consumer, e.g. HolovizOp, can trigger this operator's activation
    // -- or contend for its output pool -- before new data/capacity has actually arrived) --
    // drop this activation instead of failing the whole pipeline over it.
    auto result = compute_impl(context);
    if (!result && result.error().code == holoscan::ErrorCode::kNotReady) {
        return {};
    }
    return result;
}

template <typename PayloadT>
holoscan::expected<void, holoscan::Error> ComputeCrcOp<PayloadT>::compute_impl(holoscan::ExecutionContext& context)
{
    auto maybe_paired = hololink::operators::receive_paired_tensor_metadata(input_tensor, input_metadata);
    if (!maybe_paired) {
        return holoscan::make_unexpected(std::move(maybe_paired).error());
    }
    if (!*maybe_paired) {
        if (const uint64_t dropped = mismatched_frames_.count()) {
            HSB_LOG_WARN("ComputeCrcOp: dropped {} frame(s) in the last second whose payload and metadata "
                         "were stamped as different frames",
                dropped);
        }
        return {};
    }
    auto& paired = **maybe_paired;
    const holoscan::Tensor& in_tensor = detail::crc_payload_tensor<PayloadT>(paired.tensor.data);

    if (in_tensor.device().device_type == kDLCPU && !is_integrated_ && !host_memory_warning_) {
        host_memory_warning_ = true;
        HSB_LOG_WARN(
            "The input tensor is stored in host memory, this will reduce performance of this "
            "operator. For best performance store the input tensor in device memory.");
    }

    // compute() may run on any thread; the CRC buffers live on this device.
    CUDA_CHECK(cudaSetDevice(cuda_device_ordinal_));
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(context.cuda_stream().get());
    const int64_t frame_number = paired.metadata.data.frame_number;

    void* message = const_cast<void*>(in_tensor.data());
    size_t message_size = static_cast<size_t>(in_tensor.nbytes());
    auto message_value = reinterpret_cast<uint64_t>(message);
    hololink::core::NvtxTrace::event_u64(
        fmt::format("ComputeCrcOp buffer={:#x}", message_value).c_str(), message_value);
    CrcChannel::Slot* slot = crc_channel_->acquire(frame_number);
    CUDA_CHECK(cudaMemcpyAsync(slot->message_dptr, &message, sizeof(message), cudaMemcpyHostToDevice, cuda_stream));
    CUDA_CHECK(cudaMemcpyAsync(slot->message_size_dptr, &message_size, sizeof(message_size), cudaMemcpyHostToDevice, cuda_stream));

    uint32_t num_chunks = 1;
    nvcompStatus_t r = nvcompBatchedCRC32Async(
        (const void* const*)slot->message_dptr,
        slot->message_size_dptr,
        num_chunks,
        slot->result_dptr,
        crc_opts_,
        nvcompCRC32OnlySegment,
        nullptr,
        cuda_stream);
    if (r != nvcompSuccess) {
        crc_channel_->release(slot);
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kFailure,
            fmt::format("nvcompBatchedCRC32Async returned {}.", static_cast<int>(r)) });
    }
    crc_channel_->publish(slot, cuda_stream);

    // All outputs below share the ORIGINAL capture_time this frame arrived with (propagated,
    // not re-minted -- see propagate_emit_options()), so downstream consumers' own bounded
    // OnAligned pairing windows stay meaningful and capture_time keeps reflecting the real
    // capture instant all the way through the pipeline.
    const holoscan::EmitOptions emit_opts = hololink::operators::propagate_emit_options(
        paired.tensor.metadata, static_cast<uint64_t>(frame_number));
    std::optional<uint32_t> crc;
    if (emit_crc_) {
        // Keep the input loan alive until the GPU has finished reading this frame.
        // Carry the completed value through the graph rather than a callback side path.
        crc = crc_channel_->take(frame_number);
        if (!crc) {
            return holoscan::make_unexpected(holoscan::Error {
                holoscan::ErrorCode::kFailure, "No CRC for the current input frame" });
        }
    }
    auto emitted = hololink::operators::emit_tensor_and_metadata(
        [&] { return output_tensor.emit(std::move(paired.tensor.data), emit_opts); },
        output_metadata, paired.metadata.data, emit_opts);
    if (!emitted) {
        return holoscan::make_unexpected(std::move(emitted).error());
    }
    if (*emitted == hololink::operators::PairedEmit::kPublished && crc) {
        auto result = output_crc.emit(*crc, emit_opts);
        if (!result) {
            switch (result.error().code) {
            case holoscan::ErrorCode::kNotReady:
                break;
            case holoscan::ErrorCode::kBackpressured:
            case holoscan::ErrorCode::kPublicationPartial:
            case holoscan::ErrorCode::kPublicationIndeterminate:
                *emitted = hololink::operators::PairedEmit::kDropped;
                break;
            default:
                return holoscan::make_unexpected(std::move(result).error());
            }
        }
    }
    if (*emitted == hololink::operators::PairedEmit::kDropped) {
        if (const uint64_t dropped = dropped_frames_.count()) {
            HSB_LOG_WARN("ComputeCrcOp: dropped {} frame(s) in the last second because downstream "
                         "was behind",
                dropped);
        }
    }
    return {};
}

} // namespace hololink::operators
