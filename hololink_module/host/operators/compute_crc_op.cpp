/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hololink/module/operators/compute_crc_op.hpp"

#include <cstdint>
#include <cstring> // memset
#include <stdexcept>
#include <string>

#include <cuda.h>
#include <nvcomp/crc32.h>

#include <fmt/format.h>
#include <yaml-cpp/yaml.h>

#include <holoscan/holoscan.hpp>
#include <holoscan/utils/cuda_macros.hpp> // HOLOSCAN_CUDA_CALL

#include "hololink/module/cuda_unique.hpp" // HOLOLINK_MODULE_CUDA_CHECK
#include "hololink/module/logging.hpp" // HSB_LOG_WARN
#include "hololink/module/page_size.hpp" // PAGE_SIZE, round_up

// See roce_receiver_op.cpp: this YAML converter exists only to satisfy
// Holoscan's register_converter<T>() requirement for a parameter type that
// is never set from a YAML config (the CheckCrcOp's compute_crc_op handle
// is always passed via a C++ holoscan::Arg). It throws if ever invoked.
#define HOLOLINK_MODULE_YAML_CONVERTER_UNSUPPORTED(TYPE)                                    \
    template <>                                                                             \
    struct YAML::convert<TYPE> {                                                            \
        static Node encode(TYPE&) { throw std::runtime_error("Unsupported"); }              \
        static bool decode(const Node&, TYPE&) { throw std::runtime_error("Unsupported"); } \
    };

HOLOLINK_MODULE_YAML_CONVERTER_UNSUPPORTED(
    std::shared_ptr<hololink::module::operators::ComputeCrcOp>);

namespace hololink::module::operators {

void ComputeCrcOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input<holoscan::gxf::Entity>("input");
    spec.output<holoscan::gxf::Entity>("output");

    spec.param(cuda_device_ordinal_, "cuda_device_ordinal", "CudaDeviceOrdinal",
        "Device to use for CUDA operations", 0);
    spec.param(frame_size_, "frame_size", "FrameSize",
        "Upper bound for the size of region to check");
}

void ComputeCrcOp::start()
{
    HOLOSCAN_CUDA_CALL(cudaSetDevice(cuda_device_ordinal_.get()));

    int integrated = 0;
    HOLOLINK_MODULE_CUDA_CHECK(cuDeviceGetAttribute(
        &integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, cuda_device_ordinal_.get()));
    is_integrated_ = (integrated != 0);

    // Per-frame device scratch and events are allocated lazily by
    // acquire_slot() (and recycled), so there's nothing to pre-allocate here.

    // Initialize CRC options for the nvcomp 5.0 API.
    std::memset(&crc_opts_, 0, sizeof(crc_opts_));
    crc_opts_.spec = nvcompCRC32_JAMCRC; // Bit-inverted CRC32, matches HSB.

    // Get heuristic configuration for optimal kernel settings. Called this
    // way, no stream is used.
    size_t num_chunks = 1;
    nvcompStatus_t status = nvcompBatchedCRC32GetHeuristicConf(
        nvcompCRC32IgnoredInputChunkBytes, num_chunks, &crc_opts_.kernel_conf,
        frame_size_.get(), 0);
    if (status != nvcompSuccess) {
        throw std::runtime_error(fmt::format(
            "nvcompBatchedCRC32GetHeuristicConf returned {}.", static_cast<int>(status)));
    }
}

void ComputeCrcOp::stop()
{
    // The current CUDA device is thread-local and Holoscan may run this on a
    // different worker thread than start(), so re-select it before any CUDA call.
    HOLOSCAN_CUDA_CALL(cudaSetDevice(cuda_device_ordinal_.get()));

    // slot_storage_ owns every slot regardless of which queue it's in.
    std::lock_guard<std::mutex> guard(slots_mutex_);
    for (CrcSlot& slot : slot_storage_) {
        HOLOSCAN_CUDA_CALL(cudaFree(slot.message_size_dptr));
        HOLOSCAN_CUDA_CALL(cudaFree(slot.message_dptr));
        HOLOSCAN_CUDA_CALL(cudaFree(slot.result_dptr));
        HOLOSCAN_CUDA_CALL(cudaEventDestroy(slot.event));
    }
    free_slots_ = std::queue<CrcSlot*>();
    pending_slots_ = std::queue<CrcSlot*>();
    slot_storage_.clear();
}

ComputeCrcOp::CrcSlot* ComputeCrcOp::acquire_slot()
{
    std::lock_guard<std::mutex> guard(slots_mutex_);
    if (!free_slots_.empty()) {
        CrcSlot* slot = free_slots_.front();
        free_slots_.pop();
        return slot;
    }
    // Grow the pool by one. The pool self-limits to the number of frames that
    // are in flight between ComputeCrcOp and CheckCrcOp (a small constant), so
    // this only allocates during warm-up. Buffers are page-rounded to avoid
    // alignment issues, matching the original single-buffer allocation.
    slot_storage_.emplace_back();
    CrcSlot& slot = slot_storage_.back();
    HOLOSCAN_CUDA_CALL(cudaMalloc(&slot.message_size_dptr,
        hololink::module::round_up(sizeof(*slot.message_size_dptr), hololink::module::PAGE_SIZE)));
    HOLOSCAN_CUDA_CALL(cudaMalloc(&slot.message_dptr,
        hololink::module::round_up(sizeof(*slot.message_dptr), hololink::module::PAGE_SIZE)));
    HOLOSCAN_CUDA_CALL(cudaMalloc(&slot.result_dptr,
        hololink::module::round_up(sizeof(*slot.result_dptr), hololink::module::PAGE_SIZE)));
    HOLOSCAN_CUDA_CALL(cudaEventCreate(&slot.event));
    return &slot;
}

void ComputeCrcOp::compute(holoscan::InputContext& input, holoscan::OutputContext& output,
    holoscan::ExecutionContext& /*context*/)
{
    // The current CUDA device is thread-local and Holoscan may run this on a
    // different worker thread than start(), so re-select it before any CUDA call.
    HOLOSCAN_CUDA_CALL(cudaSetDevice(cuda_device_ordinal_.get()));

    auto maybe_entity = input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) {
        throw std::runtime_error("Failed to receive input");
    }
    auto& entity = static_cast<nvidia::gxf::Entity&>(maybe_entity.value());

    const auto maybe_tensor = entity.get<nvidia::gxf::Tensor>();
    if (!maybe_tensor) {
        throw std::runtime_error("Tensor not found in message");
    }

    const auto input_tensor = maybe_tensor.value();

    if (input_tensor->storage_type() == nvidia::gxf::MemoryStorageType::kHost) {
        if (!is_integrated_ && !host_memory_warning_) {
            host_memory_warning_ = true;
            HSB_LOG_WARN(
                "The input tensor is stored in host memory, this will reduce performance of this "
                "operator. For best performance store the input tensor in device memory.");
        }
    } else if (input_tensor->storage_type() != nvidia::gxf::MemoryStorageType::kDevice) {
        throw std::runtime_error(
            fmt::format("Unsupported storage type {}", (int)input_tensor->storage_type()));
    }

    // Get the CUDA stream from the input message if present, otherwise
    // generate one. This stream is also transmitted on the output port.
    const cudaStream_t cuda_stream = input.receive_cuda_stream();

    // Per-frame slot: its event and readback buffer aren't shared with any
    // other in-flight frame, so the work below stays fully asynchronous and
    // the synchronization is deferred to get_computed_crc() (CheckCrcOp).
    CrcSlot* slot = acquire_slot();

    void* message = input_tensor->pointer();
    size_t message_size = input_tensor->size();
    HOLOSCAN_CUDA_CALL(cudaMemcpyAsync(
        slot->message_dptr, &message, sizeof(message), cudaMemcpyHostToDevice, cuda_stream));
    HOLOSCAN_CUDA_CALL(cudaMemcpyAsync(slot->message_size_dptr, &message_size, sizeof(message_size),
        cudaMemcpyHostToDevice, cuda_stream));
    uint32_t num_chunks = 1;

    // nvcomp 5.0 API: options by value, plus segment_kind and
    // device_statuses parameters.
    nvcompStatus_t r = nvcompBatchedCRC32Async(
        (const void* const*)slot->message_dptr, // device_input_chunk_ptrs
        slot->message_size_dptr, // device_input_chunk_bytes
        num_chunks, // num_chunks
        slot->result_dptr, // device_crc32_ptr
        crc_opts_, // opts (by value)
        nvcompCRC32OnlySegment, // segment_kind (complete message)
        nullptr, // device_statuses (optional)
        cuda_stream); // stream
    if (r != nvcompSuccess) {
        throw std::runtime_error(
            fmt::format("nvcompBatchedCRC32Async returned {}.", static_cast<int>(r)));
    }
    HOLOSCAN_CUDA_CALL(cudaMemcpyAsync(&slot->computed_crc, slot->result_dptr,
        sizeof(slot->computed_crc), cudaMemcpyDeviceToHost, cuda_stream));
    HOLOSCAN_CUDA_CALL(cudaEventRecord(slot->event, cuda_stream));

    // Hand the in-flight slot to get_computed_crc(), which will synchronize on
    // slot->event (deferred, off this compute path) and read slot->computed_crc.
    {
        std::lock_guard<std::mutex> guard(slots_mutex_);
        pending_slots_.push(slot);
    }

    // Emit the input tensor.
    output.emit(entity, "output");
}

uint32_t ComputeCrcOp::get_computed_crc()
{
    // Called from CheckCrcOp::compute (a different operator, possibly a
    // different worker thread); the current CUDA device is thread-local, so
    // re-select it before synchronizing on the slot's event.
    HOLOSCAN_CUDA_CALL(cudaSetDevice(cuda_device_ordinal_.get()));

    CrcSlot* slot = nullptr;
    {
        std::lock_guard<std::mutex> guard(slots_mutex_);
        if (pending_slots_.empty()) {
            throw std::runtime_error(
                "get_computed_crc() called with no pending CRC; CheckCrcOp reads "
                "must be 1:1 with ComputeCrcOp frames.");
        }
        slot = pending_slots_.front();
        pending_slots_.pop();
    }

    // Deferred synchronization: wait for this frame's CRC, then read its value.
    // CRC-32/JAMCRC (bit-inverted CRC32) matches what the FPGA sends.
    HOLOSCAN_CUDA_CALL(cudaEventSynchronize(slot->event));
    const uint32_t crc = slot->computed_crc;

    {
        std::lock_guard<std::mutex> guard(slots_mutex_);
        free_slots_.push(slot);
    }
    return crc;
}

void CheckCrcOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input<holoscan::gxf::Entity>("input");
    spec.output<holoscan::gxf::Entity>("output");

    register_converter<std::shared_ptr<ComputeCrcOp>>();

    spec.param(compute_crc_op_, "compute_crc_op", "ComputeCrcOp",
        "Operator that computed the CRC");
    spec.param(computed_crc_metadata_name_, "computed_crc_metadata_name",
        "ComputedCrcMetadataName",
        "When specified, we'll save the computed CRC under this name.",
        std::string("computed_crc"));
}

void CheckCrcOp::compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& /*context*/)
{
    auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) {
        throw std::runtime_error("Failed to receive input");
    }

    std::shared_ptr<ComputeCrcOp> compute_crc_op = compute_crc_op_.get();
    if (!compute_crc_op) {
        throw std::runtime_error(fmt::format(
            "ComputeCrcOp wasn't specified; \"{}\" isn't properly configured.", name()));
    }

    uint32_t computed_crc = compute_crc_op->get_computed_crc();
    check_crc(computed_crc);

    // Emit the input tensor; this has to be done AFTER any metadata updates.
    op_output.emit(maybe_entity.value(), "output");
}

void CheckCrcOp::check_crc(uint32_t computed_crc)
{
    const int64_t value = static_cast<int64_t>(computed_crc);
    // Holoscan pipeline metadata; visible to Python `message.metadata` and
    // downstream operators that merge operator metadata.
    auto const& pipeline_meta = metadata();
    pipeline_meta->set(computed_crc_metadata_name_.get(), value);
}

} // namespace hololink::module::operators
