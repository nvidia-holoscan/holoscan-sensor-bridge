/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Hardware-free stand-ins for a receiver and a display, shared by the operator tests:
//
// - ChunkSourceOp delivers synthetic CSI-2 frames the way a receiver does: a uint8 device tensor
//   plus its CaptureMetadataT per receiver frame. In sub-frame mode (Geometry::sub_frame_rows
//   set) each frame arrives as consecutive chunks of bytes_per_line * sub_frame_rows bytes, each
//   its own receiver frame with its own frame_number, as the HSB delivers them; otherwise every
//   frame arrives whole. Chunks can be dropped, delivered short or delivered with a hole.
// - FrameCaptureOp copies every image it receives, with its descriptor and metadata, to the host,
//   and can hold on to the images themselves, as a consumer that keeps its frames does.

#ifndef HOLOLINK_TESTS_SYNTHETIC_CSI_SOURCE_HPP
#define HOLOLINK_TESTS_SYNTHETIC_CSI_SOURCE_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include <capture_metadata_traits.hpp>
#include <hololink/operators/capture_metadata_utils.hpp>

namespace hololink::tests::synthetic {

constexpr DLDataType kUInt8 { kDLUInt, 8, 1 };
constexpr std::size_t kMaxChunkBytes = 64 * 1024;

// A RAW10 sensor frame and how the HSB chunks it in sub-frame mode (sub_frame_rows set) or
// delivers it whole (sub_frame_rows 0).
struct Geometry {
    uint32_t width = 72; // multiple of 4 (RAW10 packs 4 pixels in 5 bytes)
    uint32_t height = 48;
    uint32_t sub_frame_rows = 16;
    // Bytes ahead of the first pixel line (embedded data + optical-black lines on a real
    // sensor). Not a multiple of the chunk size, as on a real sensor.
    uint32_t start_byte = 944;
    // Bytes after the last pixel line (a sensor's trailing status lines).
    uint32_t trailing_bytes = 0;

    // 90 transmitted bytes per line, received padded to 8 bytes
    [[nodiscard]] uint32_t bytes_per_line() const { return (width * 5 / 4 + 7) & ~7u; }
    [[nodiscard]] std::size_t chunk_bytes() const
    {
        return sub_frame_rows ? std::size_t(bytes_per_line()) * sub_frame_rows : csi_length();
    }
    [[nodiscard]] std::size_t csi_length() const
    {
        return start_byte + std::size_t(bytes_per_line()) * height + trailing_bytes;
    }
    [[nodiscard]] uint64_t chunks_per_frame() const
    {
        return (csi_length() + chunk_bytes() - 1) / chunk_bytes();
    }
};

// Distinct per frame, row and byte, so a band at the wrong place or from the wrong frame shows.
inline uint8_t pattern(uint32_t frame, uint32_t row, uint32_t byte)
{
    return static_cast<uint8_t>(frame * 101u + row * 7u + byte * 3u + 1u);
}

// The CSI-2 byte stream of one frame: header, then packed lines padded to bytes_per_line.
inline std::vector<uint8_t> csi_frame(const Geometry& g, uint32_t frame)
{
    std::vector<uint8_t> data(g.csi_length(), 0);
    std::fill_n(data.begin(), g.start_byte, uint8_t { 0xA5 });
    std::fill(data.end() - g.trailing_bytes, data.end(), uint8_t { 0x5A });
    for (uint32_t row = 0; row < g.height; ++row) {
        uint8_t* line = data.data() + g.start_byte + std::size_t(row) * g.bytes_per_line();
        for (uint32_t byte = 0; byte < g.width * 5 / 4; ++byte) {
            line[byte] = pattern(frame, row, byte);
        }
    }
    return data;
}

// What CsiToBayerOp's frameReconstruction10 kernel makes of csi_frame(): [height, width].
inline std::vector<uint16_t> expected_bayer(const Geometry& g, uint32_t frame)
{
    const std::vector<uint8_t> csi = csi_frame(g, frame);
    std::vector<uint16_t> out(std::size_t(g.width) * g.height);
    for (uint32_t row = 0; row < g.height; ++row) {
        const uint8_t* in = csi.data() + g.start_byte + std::size_t(row) * g.bytes_per_line();
        uint16_t* o = out.data() + std::size_t(row) * g.width;
        for (uint32_t group = 0; group < g.width / 4; ++group) {
            const uint8_t* b = in + group * 5;
            const uint16_t lsbs = b[4];
            o[group * 4 + 0] = uint16_t(((b[0] << 2) | (lsbs & 0x03)) << 6);
            o[group * 4 + 1] = uint16_t(((b[1] << 4) | (lsbs & 0x0C)) << 4);
            o[group * 4 + 2] = uint16_t(((b[2] << 6) | (lsbs & 0x30)) << 2);
            o[group * 4 + 3] = uint16_t((b[3] << 8) | (lsbs & 0xC0));
        }
    }
    return out;
}

struct SourceConfig {
    Geometry geometry;
    uint32_t frames = 3;
    // The receiver frame_number of the first chunk; a multiple of chunks_per_frame() so that
    // it starts a frame, whose frame number is then first_chunk / chunks_per_frame().
    uint64_t first_chunk = 0;
    // Chunks, counted from the first one across frames, that are lost (never delivered),
    // delivered short (only their first half: the HSB wrote less, and the receiver trims a
    // chunk to the HSB's bytes_written), or delivered with a hole (a lost packet: full size,
    // but the receiver reports fewer bytes received than written).
    std::set<uint64_t> dropped_chunks;
    std::set<uint64_t> truncated_chunks;
    std::set<uint64_t> holed_chunks;
    // Chunks delivered flagged kDegraded, as a receiver may flag data it knows to be damaged.
    std::set<uint64_t> degraded_chunks;
    // Chunks whose metadata carries the next chunk's frame_number, so it no longer matches the
    // tensor's frame_id.
    std::set<uint64_t> misnumbered_chunks;
    // The frames' bytes. Without sub_frame_rows each one is delivered whole, whatever its size.
    std::vector<std::vector<uint8_t>> csi_frames;
    // Called at the start of every activation with the chunk it is about to deliver (or lose), so
    // a test can change what downstream holds at an exact point of the stream.
    std::function<void(uint64_t chunk)> before_chunk;
};

struct SourceStats {
    // Chunks the source itself could not deliver (its pool or queue full); tests require none.
    std::atomic<uint32_t> dropped { 0 };
};

// Delivers one chunk per activation: a uint8 device tensor plus its CaptureMetadataT, the
// chunk's number being its frame_number (the HSB counts receiver frames, not sensor frames).
class ChunkSourceOp : public holoscan::Operator<> {
public:
    ChunkSourceOp(std::shared_ptr<const SourceConfig> config, std::shared_ptr<SourceStats> stats)
        : config_(std::move(config))
        , stats_(std::move(stats))
    {
    }

    holoscan::Output<holoscan::Tensor> tensor { "tensor" };
    holoscan::Output<hololink::CaptureMetadataT> metadata { "metadata" };

    void setup(holoscan::OperatorSpec& spec) override
    {
        spec.output(tensor, "tensor")
            .max_emits_per_compute(1U)
            .produces_tensor(holoscan::TensorOutputSpec {
                .representation = holoscan::TensorRepresentation {
                    .memory_kind = holoscan::MemoryKind::kCudaDevice,
                    .dtype = kUInt8,
                    .rank = 1U,
                },
                .bounds = holoscan::tensor_bounds(kMaxChunkBytes),
                .storage = holoscan::TensorOutputStorage::kRuntimePool,
            });
        spec.output(metadata, "metadata").max_emits_per_compute(1U);
    }

    [[nodiscard]] holoscan::Contract contract() const override
    {
        holoscan::Contract contract;
        contract.trigger(holoscan::OnClock { .period = std::chrono::milliseconds(1) });
        contract.complete_after(holoscan::ComputeCalls {
            .limit = config_->frames * config_->geometry.chunks_per_frame() });
        return contract;
    }

    holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override
    {
        const Geometry& g = config_->geometry;
        const uint64_t chunk = next_chunk_++;
        if (config_->before_chunk) {
            config_->before_chunk(chunk);
        }
        if (config_->dropped_chunks.count(chunk)) {
            return {};
        }
        const std::vector<uint8_t>& csi = config_->csi_frames.at(chunk / g.chunks_per_frame());
        const std::size_t begin = (chunk % g.chunks_per_frame()) * g.chunk_bytes();
        std::size_t size = g.sub_frame_rows ? std::min(g.chunk_bytes(), csi.size() - begin) : csi.size();
        if (config_->truncated_chunks.count(chunk)) {
            size /= 2;
        }

        const std::array<int64_t, 1> shape { static_cast<int64_t>(size) };
        auto loan = tensor.allocate_tensor(holoscan::TensorLoanRequest { .shape = shape, .dtype = kUInt8 });
        if (!loan) {
            if (loan.error().code == holoscan::ErrorCode::kResourceExhausted) {
                stats_->dropped++;
                return {};
            }
            return holoscan::make_unexpected(std::move(loan).error());
        }
        auto writer = loan->write(context.cuda_stream());
        if (!writer) {
            return holoscan::make_unexpected(std::move(writer).error());
        }
        if (cudaMemcpyAsync(writer->data(), csi.data() + begin, size, cudaMemcpyHostToDevice,
                context.cuda_stream().get())
            != cudaSuccess) {
            return holoscan::make_unexpected(
                holoscan::Error { holoscan::ErrorCode::kFailure, "cudaMemcpyAsync failed" });
        }
        auto committed = std::move(*writer).commit();
        if (!committed) {
            return holoscan::make_unexpected(std::move(committed).error());
        }

        const uint64_t frame_number = config_->first_chunk + chunk;
        hololink::CaptureMetadataT chunk_metadata;
        chunk_metadata.frame_number = static_cast<int64_t>(frame_number + config_->misnumbered_chunks.count(chunk));
        chunk_metadata.bytes_written = static_cast<int64_t>(size);
        chunk_metadata.frame_bytes_received = static_cast<int64_t>(size);
        if (config_->holed_chunks.count(chunk)) {
            chunk_metadata.frame_bytes_received -= 1408; // one packet's payload
        }
        const holoscan::EmitOptions options {
            .capture_time = context.activation_time(),
            .frame_id = frame_number,
            .flags = config_->degraded_chunks.count(chunk) ? holoscan::SampleFlags::kDegraded
                                                           : holoscan::SampleFlags::kNone,
        };
        auto emitted = tensor.emit(std::move(*loan), options);
        if (emitted) {
            emitted = metadata.emit(std::move(chunk_metadata), options);
        }
        if (!emitted && emitted.error().code == holoscan::ErrorCode::kBackpressured) {
            stats_->dropped++;
            return {};
        }
        return emitted;
    }

private:
    std::shared_ptr<const SourceConfig> config_;
    std::shared_ptr<SourceStats> stats_;
    uint64_t next_chunk_ = 0;
};

struct CapturedFrame {
    int64_t frame_number = -1;
    int64_t sub_frame_offset = -1;
    uint32_t sub_frame_rows = 0;
    int32_t image_width = 0;
    int32_t image_height = 0;
    holoscan::schema::ImageEncoding encoding = holoscan::schema::ImageEncoding_CUSTOM;
    std::vector<int64_t> shape;
    DLDataType dtype {};
    std::vector<uint8_t> bytes;
    bool degraded = false; // the image sample's kDegraded flag
    std::size_t plane_layouts = 0; // how many plane layouts the image descriptor carries
};

struct Captures {
    std::mutex mutex;
    std::vector<CapturedFrame> frames;
    // How long FrameCaptureOp takes over each image, to model a consumer slower than the source.
    std::chrono::milliseconds delay { 0 };
    // While set, FrameCaptureOp keeps every image tensor it receives, so the producer's pool slots
    // behind them stay in use until release_retained().
    bool retain = false;
    std::vector<holoscan::Tensor> retained;

    // Lets go of the retained tensors and stops retaining.
    void release_retained()
    {
        std::vector<holoscan::Tensor> released;
        {
            std::lock_guard lock(mutex);
            retain = false;
            released.swap(retained);
        }
    }
};

// Copies every assembled frame, with its descriptor and metadata, to the host.
class FrameCaptureOp : public holoscan::Operator<> {
public:
    explicit FrameCaptureOp(std::shared_ptr<Captures> captures)
        : captures_(std::move(captures))
    {
    }

    holoscan::Input<holoscan::schema::ImageT> image { "image" };
    holoscan::Input<hololink::CaptureMetadataT> metadata { "metadata" };

    void setup(holoscan::OperatorSpec& spec) override
    {
        (void)spec.input(image, "image").tensor_field(holoscan::schema::ImageTensorFields::data);
        spec.input(metadata, "metadata");
    }

    [[nodiscard]] holoscan::Contract contract() const override
    {
        holoscan::Contract contract;
        contract
            .trigger(holoscan::OnAligned {
                .ports = holoscan::ports(image, metadata),
                .window = hololink::operators::kMetadataPairingWindow,
            })
            .input(image, holoscan::Exact {})
            .input(metadata, holoscan::Exact {});
        return contract;
    }

    holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override
    {
        auto maybe_paired = hololink::operators::receive_paired_tensor_metadata(image, metadata);
        if (!maybe_paired) {
            return holoscan::make_unexpected(std::move(maybe_paired).error());
        }
        if (!*maybe_paired) {
            // The operators under test drop such pairs, so none may reach the end of the chain.
            return holoscan::make_unexpected(
                holoscan::Error { holoscan::ErrorCode::kInvalidArgument, "frame identity mismatch" });
        }
        auto& paired = **maybe_paired;
        const holoscan::schema::ImageT& received = paired.tensor.data;
        if (!received.data) {
            return holoscan::make_unexpected(
                holoscan::Error { holoscan::ErrorCode::kInvalidArgument, "image without tensor" });
        }
        const holoscan::Tensor& frame = *received.data;
        CapturedFrame captured;
        captured.frame_number = paired.metadata.data.frame_number;
        captured.sub_frame_offset = paired.metadata.data.sub_frame_offset;
        captured.sub_frame_rows = paired.metadata.data.sub_frame_rows;
        captured.image_width = received.width;
        captured.image_height = received.height;
        captured.encoding = received.encoding;
        captured.degraded = paired.tensor.metadata.degraded();
        captured.plane_layouts = received.plane_layouts.size();
        captured.shape.assign(frame.shape_span().begin(), frame.shape_span().end());
        captured.dtype = frame.dtype();
        captured.bytes.resize(static_cast<std::size_t>(frame.nbytes()));
        auto copied = frame.copy_to_host(captured.bytes.data(), captured.bytes.size(), context.cuda_stream());
        if (!copied) {
            return copied;
        }
        std::this_thread::sleep_for(captures_->delay);
        std::lock_guard lock(captures_->mutex);
        if (captures_->retain) {
            captures_->retained.push_back(frame); // a copy keeps its claim on the pool slot
        }
        captures_->frames.push_back(std::move(captured));
        return {};
    }

    // Retained images must not outlive the run that owns their pool.
    void stop() override { captures_->release_retained(); }

private:
    std::shared_ptr<Captures> captures_;
};

} // namespace hololink::tests::synthetic

#endif // HOLOLINK_TESTS_SYNTHETIC_CSI_SOURCE_HPP
