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

#include "packed_format_converter.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>

#include <fmt/format.h>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/networking.hpp>
#include <hololink/core/packetizer_program.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>

#include <hololink/operators/capture_metadata_utils.hpp>
#include <hololink/operators/image_encoding_utils.hpp>

namespace {

const char* source = R"(
extern "C" {

// Converts packed 8-bit data into 16-bit data.
__global__ void packed8bitTo16bit(unsigned short* output, const unsigned char* input,
                                  int bytes_per_line, int width, int height) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx_x >= width) || (idx_y >= height))
        return;

    int input_index = (idx_y * bytes_per_line) + idx_x;
    int output_index = (idx_y * width) + idx_x;

    output[output_index] = (((unsigned short)input[input_index]) << 8) & 0xFF00;
}

// Converts packed 10-bit data, where each 4-byte dword contains 3 pixels, into 16-bit data.
__global__ void packed10bitTo16bit(unsigned short* output, const unsigned int* input,
                                   int bytes_per_line, int width, int height) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    // 3 pixels per 4-byte group; round up so the final partial group (width
    // not a multiple of 3) is processed instead of dropped, and guard each
    // write so a partial group doesn't spill into the next row.
    if ((idx_x >= (width + 2) / 3) || (idx_y >= height))
        return;

    int input_index = (idx_y * bytes_per_line / 4) + idx_x;
    int output_index = (idx_y * width) + (idx_x * 3);
    int col = idx_x * 3;

    unsigned int packed = input[input_index];
    if (col + 0 < width) output[output_index]     = (packed << 6) & 0xFFC0;
    if (col + 1 < width) output[output_index + 1] = (packed >> 4) & 0xFFC0;
    if (col + 2 < width) output[output_index + 2] = (packed >> 14) & 0xFFC0;
}

// Converts packed 12-bit data, where 2 pixels are stored in every 3 bytes.
// Byte layout (matches the COE/FUSA RAW12 stream and is consistent with
// packed10bitTo16bit's pixel-0-first ordering):
//   byte[0]      = pixel0[7:0]
//   byte[1][3:0] = pixel0[11:8]
//   byte[1][7:4] = pixel1[3:0]
//   byte[2]      = pixel1[11:4]
__global__ void packed12bitTo16bit(unsigned short* output, const unsigned char* input,
                                   int bytes_per_line, int width, int height) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    // 2 pixels per 3-byte group; round up so the final partial group (odd
    // width) is processed instead of dropped, and guard each write so a
    // partial group doesn't spill into the next row.
    if ((idx_x >= (width + 1) / 2) || (idx_y >= height))
        return;

    int input_index = (idx_y * bytes_per_line) + (idx_x * 3);
    int output_index = (idx_y * width) + (idx_x * 2);
    int col = idx_x * 2;

    if (col + 0 < width) output[output_index]     = ((input[input_index + 1] & 0x0F) << 12 |
                                                    input[input_index]            <<  4) & 0xFFF0;
    if (col + 1 < width) output[output_index + 1] = (input[input_index + 2]        <<  8 |
                                                    (input[input_index + 1] & 0xF0)    ) & 0xFFF0;
}

})";

} // anonymous namespace

namespace hololink::operators {

namespace {

    // The bytes of one full frame of 16-bit samples: the output tensor pool's slot size. The compiler
    // evaluates it once, against the Params given to graph.op(); HSDK5 wants the geometry's nested
    // fields read through a projection like this one, not as parameter_byte_span() operands.
    holoscan::expected<std::uint64_t, holoscan::Error> output_frame_bytes(
        const PackedFormatConverterParams& params) noexcept
    {
        // Without a geometry, keep the ceiling of earlier releases so that the run compiles and
        // start() reports the missing geometry, as it always has.
        static constexpr std::uint64_t kDefaultMaxOutputBytes = 64ULL * 1024 * 1024;
        const auto& geometry = params.geometry;
        if (geometry.pixel_width == 0 || geometry.pixel_height == 0) {
            return kDefaultMaxOutputBytes;
        }
        return std::uint64_t { geometry.pixel_width } * geometry.pixel_height * sizeof(uint16_t);
    }

} // namespace

void PackedFormatConverterOp::setup(holoscan::OperatorSpec& spec)
{
    static constexpr DLDataType kUInt8 { static_cast<uint8_t>(kDLUInt), 8, 1 };
    static constexpr DLDataType kUInt16 { static_cast<uint8_t>(kDLUInt), 16, 1 };

    // Input and output declaration names share one namespace per operator (Holoscan rejects
    // reusing a name across inputs/outputs/readiness sources), so the input ports here use an
    // "in_" prefix while the outputs keep the bare "tensor"/"metadata" names that downstream
    // consumers (add_flow, bind_tensor_output_device) wire against.
    spec.input(input_tensor, "in_tensor")
        .expects_tensor(holoscan::TensorInputSpec {
            .representation = holoscan::TensorRepresentation {
                .memory_kind = holoscan::MemoryKind::kCudaDevice,
                .dtype = kUInt8,
            },
        });
    spec.input(input_metadata, "in_metadata");
    spec.output(output_tensor, "tensor")
        .max_emits_per_compute(1U)
        .tensor_field(holoscan::schema::ImageTensorFields::data)
        .produces_tensor(holoscan::TensorOutputSpec {
                             .representation = holoscan::TensorRepresentation {
                                 .memory_kind = holoscan::MemoryKind::kCudaDevice,
                                 .dtype = kUInt16,
                             },
                             .bounds = holoscan::TensorBounds {
                                 .max_rank = 3U,
                             },
                             .storage = holoscan::TensorOutputStorage::kRuntimePool,
                         },
            // One slot holds one full frame of 16-bit samples, so the pool is sized by the
            // configured geometry rather than by a fixed ceiling: the compiler resolves this
            // expression from the Params each graph.op() call supplies.
            holoscan::parameter_projected_byte_span<&output_frame_bytes>("com.nvidia.hololink.operators.PackedFormatConverterOp.output_frame_bytes", holoscan::param_ref<&PackedFormatConverterParams::geometry, &hololink::csi::CsiGeometry::pixel_width>(), holoscan::param_ref<&PackedFormatConverterParams::geometry, &hololink::csi::CsiGeometry::pixel_height>()));
    spec.output(output_metadata, "metadata").max_emits_per_compute(1U);
}

holoscan::Contract PackedFormatConverterOp::contract() const
{
    // Upstream (the receiver) stamps both ports with an identical capture_time, so OnAligned
    // pairs them on an exact match (see kMetadataPairingWindow).
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

void PackedFormatConverterOp::start()
{
    const PackedFormatConverterParams& p = params();
    const hololink::csi::CsiGeometry& g = p.geometry;
    if (g.pixel_width == 0 || g.pixel_height == 0) {
        throw std::invalid_argument(
            "PackedFormatConverterOp needs PackedFormatConverterParams::geometry (from "
            "FusaCoeCaptureOp::make_geometry_adapter()).");
    }
    const uint32_t line_bytes = transmitted_line_bytes(g.pixel_format, g.pixel_width);
    if (g.bytes_per_line < line_bytes) {
        throw std::invalid_argument(fmt::format(
            "PackedFormatConverterOp: bytes_per_line={} is shorter than a packed line of {} pixels "
            "({} bytes).",
            g.bytes_per_line, g.pixel_width, line_bytes));
    }
    HSB_LOG_INFO("bytes_per_line={}, width={}, height={}", g.bytes_per_line, g.pixel_width,
        g.pixel_height);
    bytes_per_line_ = g.bytes_per_line;
    pixel_width_ = g.pixel_width;
    pixel_height_ = g.pixel_height;
    pixel_format_ = g.pixel_format;

    CudaCheck(cuInit(0));
    CudaCheck(cuDeviceGet(&cuda_device_, p.cuda_device_ordinal));
    CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));
    int integrated = 0;
    CudaCheck(cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, cuda_device_));
    is_integrated_ = (integrated != 0);

    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);

    cuda_function_launcher_.reset(new hololink::common::CudaFunctionLauncher(
        source, { "packed8bitTo16bit", "packed10bitTo16bit", "packed12bitTo16bit" }));
}

void PackedFormatConverterOp::stop()
{
    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);

    cuda_function_launcher_.reset();

    CudaCheck(cuDevicePrimaryCtxRelease(cuda_device_));
    cuda_context_ = nullptr;
}

holoscan::expected<void, holoscan::Error> PackedFormatConverterOp::compute(
    holoscan::ExecutionContext& context)
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

void PackedFormatConverterOp::count_dropped_frame(const char* reason)
{
    if (const uint64_t dropped = dropped_frames_.count()) {
        HSB_LOG_WARN("PackedFormatConverterOp: dropped {} frame(s) in the last second ({})",
            dropped, reason);
    }
}

holoscan::expected<void, holoscan::Error> PackedFormatConverterOp::compute_impl(
    holoscan::ExecutionContext& context)
{
    auto maybe_paired = hololink::operators::receive_paired_tensor_metadata(input_tensor, input_metadata);
    if (!maybe_paired) {
        return holoscan::make_unexpected(std::move(maybe_paired).error());
    }
    if (!*maybe_paired) {
        if (const uint64_t dropped = mismatched_frames_.count()) {
            HSB_LOG_WARN("PackedFormatConverterOp: dropped {} frame(s) in the last second whose payload and metadata "
                         "were stamped as different frames",
                dropped);
        }
        return {};
    }
    auto& paired = **maybe_paired;
    const holoscan::Tensor& in_tensor = paired.tensor.data;
    const hololink::CaptureMetadataT& in_frame = paired.metadata.data;

    if (in_tensor.device().device_type == kDLCPU && !is_integrated_ && !host_memory_warning_) {
        host_memory_warning_ = true;
        HSB_LOG_WARN(
            "The input tensor is stored in host memory, this will reduce performance of this "
            "operator. For best performance store the input tensor in device memory.");
    }

    if (in_tensor.ndim() != 1) {
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kInvalidArgument, "Tensor must be one dimensional" });
    }
    // The kernels read every line up to the last one's packed pixels.
    const std::size_t source_offset = params().header_included ? params().geometry.start_byte : 0;
    const std::size_t needed = source_offset + std::size_t(bytes_per_line_) * (pixel_height_ - 1)
        + transmitted_line_bytes(pixel_format_, pixel_width_);
    if (static_cast<std::size_t>(in_tensor.nbytes()) < needed) {
        return holoscan::make_unexpected(holoscan::Error { holoscan::ErrorCode::kInvalidArgument,
            fmt::format("PackedFormatConverterOp: frame {} has {} bytes, less than the {} its "
                        "geometry needs",
                in_frame.frame_number, in_tensor.nbytes(), needed) });
    }

    static constexpr DLDataType kUInt16 { static_cast<uint8_t>(kDLUInt), 16, 1 };
    const std::array<int64_t, 3> shape { int64_t(pixel_height_), int64_t(pixel_width_), 1 };
    auto loan = output_tensor.allocate_tensor(
        holoscan::TensorLoanRequest { .shape = shape, .dtype = kUInt16 });
    if (!loan) {
        if (loan.error().code == holoscan::ErrorCode::kResourceExhausted) {
            // Every pool slot is still held downstream (the consumer is behind): drop this
            // frame, like the receiver does, rather than failing the session.
            count_dropped_frame("output pool exhausted");
            return {};
        }
        return holoscan::make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
        return holoscan::make_unexpected(std::move(writer).error());
    }
    void* const out_ptr = writer->data();

    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);
    const CUstream cuda_stream = context.cuda_stream().get();

    // CRC paths supply the full frame; normal capture already strips the CSI header.
    const uint8_t* input_ptr = static_cast<const uint8_t*>(in_tensor.data()) + source_offset;

    switch (pixel_format_) {
    case hololink::csi::PixelFormat::RAW_8:
        cuda_function_launcher_->launch("packed8bitTo16bit",
            { pixel_width_, pixel_height_, 1 },
            cuda_stream,
            out_ptr,
            input_ptr,
            bytes_per_line_, pixel_width_, pixel_height_);
        break;
    case hololink::csi::PixelFormat::RAW_10:
        cuda_function_launcher_->launch("packed10bitTo16bit",
            { (pixel_width_ + 2) / 3, // outputs 3 pixels per shader invocation (round up for the tail)
                pixel_height_, 1 },
            cuda_stream,
            out_ptr,
            input_ptr,
            bytes_per_line_, pixel_width_, pixel_height_);
        break;
    case hololink::csi::PixelFormat::RAW_12:
        cuda_function_launcher_->launch("packed12bitTo16bit",
            { (pixel_width_ + 1) / 2, // outputs 2 pixels per shader invocation (round up for the tail)
                pixel_height_, 1 },
            cuda_stream,
            out_ptr,
            input_ptr,
            bytes_per_line_, pixel_width_, pixel_height_);
        break;
    default:
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kInvalidArgument, "Unsupported pixel format" });
    }

    auto committed = std::move(*writer).commit();
    if (!committed) {
        return holoscan::make_unexpected(std::move(committed).error());
    }

    // Pass every metadata field through unchanged (this operator only transforms the
    // tensor payload's packing, not the frame's hardware metadata).
    hololink::CaptureMetadataT out_metadata = hololink::operators::forward_metadata(in_frame);

    // Both emits below share the ORIGINAL capture_time this frame arrived with (propagated,
    // not re-minted -- see propagate_emit_options()), so downstream consumers' own bounded
    // OnAligned pairing windows stay meaningful and capture_time keeps reflecting the real
    // capture instant all the way through the pipeline.
    const holoscan::EmitOptions emit_opts = hololink::operators::propagate_emit_options(
        paired.tensor.metadata, static_cast<uint64_t>(in_frame.frame_number));
    holoscan::schema::ImageT out_descriptor;
    out_descriptor.height = static_cast<int32_t>(pixel_height_);
    out_descriptor.width = static_cast<int32_t>(pixel_width_);
    out_descriptor.encoding = hololink::operators::bayer_image_encoding(params().bayer_format);
    auto emitted = hololink::operators::emit_tensor_and_metadata(
        [&] { return output_tensor.emit_tensor(std::move(*loan), out_descriptor, emit_opts); },
        output_metadata, out_metadata, emit_opts);
    if (!emitted) {
        return holoscan::make_unexpected(std::move(emitted).error());
    }
    if (*emitted == hololink::operators::PairedEmit::kDropped) {
        count_dropped_frame("downstream queue full");
    }
    return {};
}

uint32_t PackedFormatConverterOp::transmitted_line_bytes(hololink::csi::PixelFormat pixel_format,
    uint32_t pixel_width)
{
    // The sensor's CSI-2 line, MIPI packed...
    uint32_t csi_line_bytes = 0;
    switch (pixel_format) {
    case hololink::csi::PixelFormat::RAW_8:
        csi_line_bytes = pixel_width;
        break;
    case hololink::csi::PixelFormat::RAW_10:
        // 4 pixels per 5 bytes
        csi_line_bytes = pixel_width * 5 / 4;
        break;
    case hololink::csi::PixelFormat::RAW_12:
        // 2 pixels per 3 bytes
        csi_line_bytes = pixel_width * 3 / 2;
        break;
    default:
        throw std::invalid_argument("PackedFormatConverterOp: unsupported pixel format");
    }
    // ...as the HSB's packetizer program for that format reshapes it. The program's
    // schedule sizes the output (a whole number of its loops), not the pixel count: 1280
    // RAW_10 pixels take 1792 bytes, not 3 pixels per 4 bytes = 1708. RAW_8 has no program
    // and passes through unchanged.
    return hololink::csi::get_packetizer_program(pixel_format)->get_output_size(csi_line_bytes);
}

} // namespace hololink::operators
