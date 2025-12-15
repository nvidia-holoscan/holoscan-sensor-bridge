/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "compute_crc.hpp"

#include <nvcomp/crc32.h>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/networking.hpp>
#include <hololink/core/nvtx_trace.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/utils/cuda_macros.hpp>

/**
 * @brief This macro defining a YAML converter which throws for unsupported types.
 *
 * Background: Holoscan supports setting parameters through YAML files. But for some parameters
 * it makes no sense to specify them in YAML files. Therefore use a converter which throws for these types.
 *
 * @tparam TYPE
 */
#define YAML_CONVERTER(TYPE)                                                                \
    template <>                                                                             \
    struct YAML::convert<TYPE> {                                                            \
        static Node encode(TYPE&) { throw std::runtime_error("Unsupported"); }              \
        static bool decode(const Node&, TYPE&) { throw std::runtime_error("Unsupported"); } \
    };

YAML_CONVERTER(std::shared_ptr<hololink::operators::ComputeCrcOp>);

namespace hololink::operators {

void ComputeCrcOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input<holoscan::gxf::Entity>("input");
    spec.output<holoscan::gxf::Entity>("output");

    spec.param(cuda_device_ordinal_, "cuda_device_ordinal", "CudaDeviceOrdinal",
        "Device to use for CUDA operations", 0);
    spec.param(frame_size_, "frame_size", "FrameSize",
        "Upper bound for the size of region to check");
    cuda_stream_handler_.define_params(spec);
}

void ComputeCrcOp::start()
{
    HOLOSCAN_CUDA_CALL(cudaSetDevice(cuda_device_ordinal_.get()));

    int integrated = 0;
    CudaCheck(cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, cuda_device_ordinal_.get()));
    is_integrated_ = (integrated != 0);

    // Make sure we don't have page alignment issues with allocated buffers
    const size_t message_size_dptr_size = hololink::core::round_up(sizeof(*message_size_dptr_), hololink::core::PAGE_SIZE);
    HOLOSCAN_CUDA_CALL(cudaMalloc(&message_size_dptr_, message_size_dptr_size));
    const size_t message_dptr_size = hololink::core::round_up(sizeof(*message_dptr_), hololink::core::PAGE_SIZE);
    HOLOSCAN_CUDA_CALL(cudaMalloc(&message_dptr_, message_dptr_size));
    const size_t result_dptr_size = hololink::core::round_up(sizeof(*result_dptr_), hololink::core::PAGE_SIZE);
    HOLOSCAN_CUDA_CALL(cudaMalloc(&result_dptr_, result_dptr_size));

    // We use this to synchronize CRC readout.
    HOLOSCAN_CUDA_CALL(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

    // Initialize CRC options for nvcomp 5.0 API
    memset(&crc_opts_, 0, sizeof(crc_opts_));
    crc_opts_.spec = nvcompCRC32_JAMCRC; // Bit-inverted CRC32, matches existing behavior

    // Get heuristic configuration for optimal kernel settings
    // Note that called this way, stream_ isn't used.
    size_t num_chunks = 1;
    nvcompStatus_t status = nvcompBatchedCRC32GetHeuristicConf(nvcompCRC32IgnoredInputChunkBytes, num_chunks, &crc_opts_.kernel_conf,
        frame_size_, stream_);
    if (status != nvcompSuccess) {
        throw std::runtime_error(fmt::format("nvcompBatchedCRC32GetHeuristicConf returned {}.", static_cast<int>(status)));
    }
}

void ComputeCrcOp::stop()
{
    HOLOSCAN_CUDA_CALL(cudaFree(message_size_dptr_));
    message_size_dptr_ = nullptr;
    HOLOSCAN_CUDA_CALL(cudaFree(message_dptr_));
    message_dptr_ = nullptr;
    HOLOSCAN_CUDA_CALL(cudaFree(result_dptr_));
    result_dptr_ = nullptr;
    HOLOSCAN_CUDA_CALL(cudaStreamDestroy(stream_));
    stream_ = 0;
}

void ComputeCrcOp::compute(holoscan::InputContext& input, holoscan::OutputContext& output,
    holoscan::ExecutionContext& context)
{
    auto maybe_entity = input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) {
        throw std::runtime_error("Failed to receive input");
    }
    auto& entity = static_cast<nvidia::gxf::Entity&>(maybe_entity.value());

    // get the CUDA stream from the input message
    gxf_result_t stream_handler_result
        = cuda_stream_handler_.from_message(context.context(), entity);
    if (stream_handler_result != GXF_SUCCESS) {
        throw std::runtime_error(fmt::format("Failed to get the CUDA stream from incoming messages: {}", GxfResultStr(stream_handler_result)));
    }

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

    void* message = input_tensor->pointer();
    size_t message_size = input_tensor->size();
    hololink::core::NvtxTrace::event_u64("ComputeCrcOp", reinterpret_cast<uint64_t>(message));
    HOLOSCAN_CUDA_CALL(cudaMemcpyAsync(message_dptr_, &message, sizeof(message), cudaMemcpyHostToDevice, stream_));
    HOLOSCAN_CUDA_CALL(cudaMemcpyAsync(message_size_dptr_, &message_size, sizeof(message_size), cudaMemcpyHostToDevice, stream_));
    uint32_t num_chunks = 1;

    // nvcomp 5.0 API: pass options by value, add segment_kind and device_statuses parameters
    nvcompStatus_t r = nvcompBatchedCRC32Async(
        (const void* const*)message_dptr_, // device_input_chunk_ptrs
        message_size_dptr_, // device_input_chunk_bytes
        num_chunks, // num_chunks
        result_dptr_, // device_crc32_ptr
        crc_opts_, // opts (by value, not pointer)
        nvcompCRC32OnlySegment, // segment_kind (complete message, not streaming)
        nullptr, // device_statuses (optional)
        stream_); // stream
    if (r != nvcompSuccess) {
        throw std::runtime_error(fmt::format("nvcompBatchedCRC32Async returned {}.", static_cast<int>(r)));
    }
    HOLOSCAN_CUDA_CALL(cudaMemcpyAsync(&computed_crc_, result_dptr_, sizeof(computed_crc_), cudaMemcpyDeviceToHost, stream_));
    // Emit the input tensor
    output.emit(maybe_entity.value(), "output");
}

uint32_t ComputeCrcOp::get_computed_crc()
{
    HOLOSCAN_CUDA_CALL(cudaStreamSynchronize(stream_));
    hololink::core::NvtxTrace::event_u64("Done", computed_crc_);
    // Using CRC-32/JAMCRC preset which returns bit-inverted CRC32.
    // This directly matches what the FPGA sends (JAMCRC format).
    return computed_crc_;
}

//
void CheckCrcOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input<holoscan::gxf::Entity>("input");
    spec.output<holoscan::gxf::Entity>("output");

    // Register converter for ComputeCrcOp shared_ptr
    register_converter<std::shared_ptr<ComputeCrcOp>>();

    spec.param(compute_crc_op_, "compute_crc_op", "ComputeCrcOp",
        "Operator that computed the CRC");
    cuda_stream_handler_.define_params(spec);
}

void CheckCrcOp::compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& context)
{
    auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) {
        throw std::runtime_error("Failed to receive input");
    }
    auto& entity = static_cast<nvidia::gxf::Entity&>(maybe_entity.value());

    gxf_result_t stream_handler_result
        = cuda_stream_handler_.from_message(context.context(), entity);
    if (stream_handler_result != GXF_SUCCESS) {
        throw std::runtime_error(fmt::format("Failed to get the CUDA stream from incoming messages: {}", GxfResultStr(stream_handler_result)));
    }

    // Emit the input tensor
    op_output.emit(maybe_entity.value(), "output");

    //
    std::shared_ptr<ComputeCrcOp> compute_crc_op = compute_crc_op_.get();
    if (!compute_crc_op) {
        throw std::runtime_error(fmt::format("ComputeCrcOp wasn't specified; \"{}\" isn't properly configured.", name()));
    }

    uint32_t computed_crc = compute_crc_op->get_computed_crc();
    check_crc(computed_crc);
}

} // namespace hololink::operators
