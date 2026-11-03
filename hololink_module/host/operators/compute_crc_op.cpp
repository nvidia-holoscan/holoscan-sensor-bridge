/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hololink/module/operators/compute_crc_op.hpp"

#include <cstring>
#include <optional>
#include <stdexcept>

#include <fmt/format.h>

#include <hololink/operators/capture_metadata_utils.hpp>

#include "hololink/module/cuda_unique.hpp" // HOLOLINK_MODULE_CUDA_CHECK
#include "hololink/module/logging.hpp" // HSB_LOG_*

namespace hololink::module::operators {

void ComputeCrcOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input(input_tensor, "in_tensor");
    spec.input(input_metadata, "in_metadata");
    spec.output(output_tensor, "tensor");
    spec.output(output_metadata, "metadata");
}

holoscan::Contract ComputeCrcOp::contract() const
{
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

void ComputeCrcOp::start()
{
    if (!crc_channel_) {
        throw std::invalid_argument("ComputeCrcOp needs the CrcChannel its CheckCrcOp shares");
    }
    // A restarted run starts afresh.
    crc_channel_->clear();

    const auto& p = params();
    cuda_device_ordinal_ = p.cuda_device_ordinal;
    frame_size_ = p.frame_size;

    HOLOLINK_MODULE_CUDA_CHECK(cudaSetDevice(cuda_device_ordinal_));

    int integrated = 0;
    HOLOLINK_MODULE_CUDA_CHECK(cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, cuda_device_ordinal_));
    is_integrated_ = (integrated != 0);

    // Initialize CRC options for the nvcomp API.
    std::memset(&crc_opts_, 0, sizeof(crc_opts_));
    crc_opts_.spec = nvcompCRC32_JAMCRC; // Bit-inverted CRC32, matches existing behavior

    // Get heuristic configuration for optimal kernel settings. Note that called this way,
    // stream (0) isn't used.
    size_t num_chunks = 1;
    nvcompStatus_t status = nvcompBatchedCRC32GetHeuristicConf(
        nvcompCRC32IgnoredInputChunkBytes, num_chunks, &crc_opts_.kernel_conf, frame_size_, 0);
    if (status != nvcompSuccess) {
        throw std::runtime_error(
            fmt::format("nvcompBatchedCRC32GetHeuristicConf returned {}.", static_cast<int>(status)));
    }
}

holoscan::expected<void, holoscan::Error> ComputeCrcOp::compute(holoscan::ExecutionContext& context)
{
    auto result = compute_impl(context);
    if (!result && result.error().code == holoscan::ErrorCode::kNotReady) {
        return {};
    }
    return result;
}

holoscan::expected<void, holoscan::Error> ComputeCrcOp::compute_impl(holoscan::ExecutionContext& context)
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
    const holoscan::Tensor& in_tensor = paired.tensor.data;

    if (in_tensor.device().device_type == kDLCPU && !is_integrated_ && !host_memory_warning_) {
        host_memory_warning_ = true;
        HSB_LOG_WARN(
            "The input tensor is stored in host memory, this will reduce performance of this "
            "operator. For best performance store the input tensor in device memory.");
    }

    // compute() may run on any thread; the CRC buffers live on this device.
    HOLOLINK_MODULE_CUDA_CHECK(cudaSetDevice(cuda_device_ordinal_));
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(context.cuda_stream().get());
    const int64_t frame_number = paired.metadata.data.frame_number;

    void* message = const_cast<void*>(in_tensor.data());
    size_t message_size = static_cast<size_t>(in_tensor.nbytes());
    hololink::operators::CrcChannel::Slot* slot = crc_channel_->acquire(frame_number);
    HOLOLINK_MODULE_CUDA_CHECK(cudaMemcpyAsync(slot->message_dptr, &message, sizeof(message), cudaMemcpyHostToDevice, cuda_stream));
    HOLOLINK_MODULE_CUDA_CHECK(cudaMemcpyAsync(slot->message_size_dptr, &message_size, sizeof(message_size), cudaMemcpyHostToDevice, cuda_stream));

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

    const holoscan::EmitOptions emit_opts = hololink::operators::propagate_emit_options(
        paired.tensor.metadata, static_cast<uint64_t>(frame_number));
    auto emitted = hololink::operators::emit_tensor_and_metadata(
        [&] { return output_tensor.emit(std::move(paired.tensor.data), emit_opts); },
        output_metadata, paired.metadata.data, emit_opts);
    if (!emitted) {
        return holoscan::make_unexpected(std::move(emitted).error());
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

void CheckCrcOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input(input_metadata, "in_metadata");
    spec.output(output_metadata, "metadata");
}

holoscan::Contract CheckCrcOp::contract() const
{
    holoscan::Contract result;
    result.trigger(holoscan::OnEach { input_metadata });
    return result;
}

holoscan::expected<void, holoscan::Error> CheckCrcOp::compute(holoscan::ExecutionContext& /*context*/)
{
    auto sample = input_metadata.receive();
    if (!sample) {
        // kNotReady: activated without new metadata.
        if (sample.error().code == holoscan::ErrorCode::kNotReady) {
            return {};
        }
        return holoscan::make_unexpected(std::move(sample).error());
    }
    if (!crc_channel_) {
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kInvalidArgument,
            "CheckCrcOp needs the CrcChannel its ComputeCrcOp was given" });
    }

    const int64_t frame_number = sample->data.frame_number;
    if (const std::optional<uint32_t> computed_crc = crc_channel_->take(frame_number)) {
        check_crc(*computed_crc, sample->data);
    } else if (const uint64_t missing = missing_crcs_.count()) {
        // The frame was dropped before ComputeCrcOp saw it.
        HSB_LOG_WARN("CheckCrcOp: no CRC for {} frame(s) in the last second (latest frame {})",
            missing, frame_number);
    }

    auto emitted = hololink::operators::emit_metadata(output_metadata, sample->data,
        hololink::operators::propagate_emit_options(sample->metadata, static_cast<uint64_t>(frame_number)));
    if (!emitted) {
        return holoscan::make_unexpected(std::move(emitted).error());
    }
    if (*emitted == hololink::operators::PairedEmit::kDropped) {
        if (const uint64_t dropped = dropped_frames_.count()) {
            HSB_LOG_WARN("CheckCrcOp: dropped {} frame(s) in the last second because downstream "
                         "was behind",
                dropped);
        }
    }
    return {};
}

void CheckCrcOp::check_crc(uint32_t computed_crc)
{
    last_computed_crc_ = computed_crc;
}

void CheckCrcOp::check_crc(uint32_t computed_crc, const hololink::CaptureMetadataT& /*metadata*/)
{
    check_crc(computed_crc);
}

} // namespace hololink::module::operators
