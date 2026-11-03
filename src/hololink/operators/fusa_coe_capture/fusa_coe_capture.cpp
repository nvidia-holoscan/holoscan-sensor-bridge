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

#include "fusa_coe_capture.hpp"

#include <array>
#include <cerrno>
#include <climits>
#include <cstring>
#include <ctime>

#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>

#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include <hololink/core/hololink.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/on_scope_exit.hpp>
#include <hololink/core/packetizer_program.hpp>

namespace hololink::operators {

using hololink::core::OnScopeExit;

namespace {

    class LegacyCoeChannelConfig : public fusa_coe_capture::CoeChannelConfig {
    public:
        explicit LegacyCoeChannelConfig(hololink::DataChannel* channel)
            : channel_(channel)
        {
        }

        void set_packetizer(
            bool enabled, hololink::csi::PixelFormat pixel_format) override
        {
            std::shared_ptr<hololink::PacketizerProgram> packetizer_program;
            if (enabled) {
                packetizer_program = hololink::csi::get_packetizer_program(pixel_format);
            } else {
                packetizer_program = std::make_shared<hololink::NullPacketizerProgram>();
            }
            channel_->set_packetizer_program(packetizer_program);
        }

        void configure_coe(
            uint8_t coe_channel, size_t frame_size, uint32_t bytes_per_line) override
        {
            channel_->configure_coe(coe_channel, frame_size, bytes_per_line);
        }

        void unconfigure() override
        {
            channel_->unconfigure();
        }

    private:
        hololink::DataChannel* channel_;
    };

    class LegacyCoeMetadataDecoder : public fusa_coe_capture::CoeMetadataDecoder {
    public:
        bool decode(
            const void* host_memory, size_t host_memory_size,
            fusa_coe_capture::CoeFrameMetadata& out) const override
        {
            auto frame_metadata = Hololink::deserialize_metadata(
                static_cast<const uint8_t*>(host_memory), host_memory_size);
            out.timestamp_s = static_cast<int64_t>(frame_metadata.timestamp_s);
            out.timestamp_ns = static_cast<int64_t>(frame_metadata.timestamp_ns);
            out.metadata_s = static_cast<int64_t>(frame_metadata.metadata_s);
            out.metadata_ns = static_cast<int64_t>(frame_metadata.metadata_ns);
            out.crc = static_cast<int64_t>(frame_metadata.crc);
            out.frame_number = static_cast<int64_t>(frame_metadata.frame_number);
            out.bytes_written = static_cast<int64_t>(frame_metadata.bytes_written);
            return true;
        }
    };

} // namespace

void FusaCoeCaptureOp::setup(holoscan::OperatorSpec& spec)
{
    static constexpr DLDataType kUInt8 { static_cast<uint8_t>(kDLUInt), 8, 1 };
    // Tensor output port: raw pixel data, a 1-D byte tensor. kCudaDevice by default; kHost
    // when cpu_output is set (see FusaCoeCaptureParams::cpu_output). kMaxFrameBytes is fixed
    // at setup() time because frame geometry is only known after configure() is called later.
    spec.output(tensor, "tensor")
        .max_emits_per_compute(1U)
        .produces_tensor(holoscan::TensorOutputSpec {
            .representation = holoscan::TensorRepresentation {
                .memory_kind = cpu_output_
                    ? holoscan::MemoryKind::kHost
                    : holoscan::MemoryKind::kCudaDevice,
                .dtype = kUInt8,
            },
            .bounds = holoscan::TensorBounds {
                .max_rank = 1U,
                .max_byte_span = kMaxFrameBytes,
            },
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        });

    // Metadata output port: per-capture hardware metadata, correlated with the
    // tensor above via the shared capture_time/frame_id stamped on both in
    // compute().
    spec.output(metadata, "metadata")
        .max_emits_per_compute(1U);

    // Device-driven scheduling: the acquire thread posts a notification here for each
    // frame compute() can take (FusaCoeCaptureCore keeps only the newest frame and posts
    // when it stores one after compute() has taken the last), which wakes the HSDK5
    // scheduler to call compute(). A notification counts against the capacity until the
    // activation it triggered retires, so at most two are outstanding: the running
    // activation's and the next frame's. The rest is headroom; a full source makes the
    // acquire thread retry (frame_ready()).
    static constexpr uint32_t kFrameReadyNotificationCapacity = 8;
    spec.notification_source(source_, "frame-ready")
        .capacity(kFrameReadyNotificationCapacity)
        .sender_reference_capacity(1);

    // kArm runs after graph compile but before start(), so the NotificationSender
    // is bound to the current run generation before the acquire thread can post.
    spec.lifecycle()
        .stage(holoscan::LifecycleStage::kArm,
            static_cast<holoscan::LifecycleStatus (FusaCoeCaptureOp::*)(holoscan::LifecycleContext&) noexcept>(
                &FusaCoeCaptureOp::on_arm));
}

holoscan::Contract FusaCoeCaptureOp::contract() const
{
    holoscan::Contract result;
    result.trigger(holoscan::OnNotified { .event = source_ });
    if (const int64_t frame_limit = this->params().frame_limit; frame_limit > 0) {
        result.complete_after(holoscan::ComputeCalls { .limit = static_cast<uint64_t>(frame_limit) });
    }
    return result;
}

void FusaCoeCaptureOp::start()
{
    const auto& p = this->params();

    if (p.cpu_output != cpu_output_) {
        throw std::invalid_argument("FusaCoeCaptureOp: cpu_output must match the constructor memory kind");
    }

    if (!p.hololink_channel) {
        throw std::invalid_argument("FusaCoeCaptureOp::start: hololink_channel must be set");
    }
    if (p.mac_addr.size() != 6U) {
        throw std::invalid_argument("FusaCoeCaptureOp::start: mac_addr must contain 6 bytes");
    }
    if (!p.device_start || !p.device_stop) {
        throw std::invalid_argument("FusaCoeCaptureOp::start: device_start/device_stop must be set");
    }

    if (p.frame_size_bytes) {
        // Opaque byte blob: no CFA, no packetizer -- see FusaCoeCaptureParams::frame_size_bytes.
        core_.configure_frame_size(*p.frame_size_bytes);
    } else {
        // p.geometry.bytes_per_line was computed by the camera driver from whatever
        // packetizer_enabled value was passed to make_geometry_adapter() -- independently
        // of p.packetizer_enabled here, which the caller must also set to the same value
        // (see FusaCoeCaptureParams::packetizer_enabled and make_geometry_adapter()'s doc
        // comment). If the two diverge, NvSciBuf gets the wrong color format for the
        // actual wire layout, silently. Recompute the expected received line size the same
        // way GeometryCaptureConverter did and fail loudly on a mismatch instead.
        const uint32_t expected_bytes_per_line = fusa_coe_capture::FusaCoeCaptureCore::received_line_bytes(
            fusa_coe_capture::FusaCoeCaptureCore::transmitted_line_bytes(
                p.pixel_format, p.geometry.pixel_width, p.packetizer_enabled),
            p.packetizer_enabled);
        if (expected_bytes_per_line != p.geometry.bytes_per_line) {
            throw std::invalid_argument(fmt::format(
                "FusaCoeCaptureOp::start: geometry.bytes_per_line ({}) does not match what "
                "pixel_format/geometry.pixel_width/packetizer_enabled predict ({}) -- "
                "packetizer_enabled likely differs from the value passed to "
                "make_geometry_adapter() when geometry was captured",
                p.geometry.bytes_per_line, expected_bytes_per_line));
        }

        // Apply the capture format and geometry the caller resolved (via the sensor driver
        // and make_geometry_adapter()) before declaring this operator — there is no live
        // instance to configure earlier than this.
        core_.request_format(p.pixel_format, p.bayer_format, p.packetizer_enabled);
        core_.configure(
            p.geometry.start_byte, p.geometry.bytes_per_line,
            p.geometry.pixel_width, p.geometry.pixel_height,
            p.geometry.pixel_format, p.geometry.trailing_bytes);
    }

    LegacyCoeChannelConfig channel(p.hololink_channel);
    core_.start(
        p.interface,
        p.mac_addr,
        p.timeout,
        channel,
        p.device_start,
        p.device_stop,
        [this]() { return frame_ready(); },
        p.stall_timeout_ms,
        p.name);
    frame_limit_run_end_.started(p.frame_limit, p.frame_limit_ends_run);
}

void FusaCoeCaptureOp::stop()
{
    frame_limit_run_end_.stopped();
    // compute() enqueues a cudaMemcpyAsync (and a cudaLaunchHostFunc release) that read
    // from NvSciBuf-backed device memory on the runtime-assigned compute stream, and
    // core_.stop() below frees that memory synchronously. stop() has no ExecutionContext
    // (so no access to context.cuda_stream()), and HSDK5's own CUDA-stream quiescing
    // (RunSessionState::~RunSessionState() -> cuda_execution_resources_->quiesce()) is a
    // teardown step that runs strictly *after* every operator's stop() callback has
    // already returned -- not before or during it. So without a synchronization point of
    // our own here, a copy still in flight when stop() runs would read freed memory.
    // cudaDeviceSynchronize() is coarser than synchronizing just this operator's stream,
    // but stop() is teardown-only (not a hot path) and this avoids depending on stream
    // handle identity/lifetime assumptions across activations.
    cudaDeviceSynchronize();

    const auto& p = this->params();
    LegacyCoeChannelConfig channel(p.hololink_channel);
    core_.stop(channel, p.device_stop);
    // Invalidate the sender after the acquire thread has been joined (inside
    // core_.stop(), above), so that any in-flight frame_ready() calls complete
    // before we close.
    sender_ = holoscan::NotificationSender {};
}

holoscan::expected<void, holoscan::Error> FusaCoeCaptureOp::compute(
    holoscan::ExecutionContext& context)
{
    const auto& p = this->params();
    // With frame_limit_ends_run, the call that reaches frame_limit may end the whole run.
    const FrameLimitRunEnd::Call frame_limit_call(frame_limit_run_end_, context);
    if (frame_limit_call.reached()) {
        HSB_LOG_INFO("Frame limit {} reached{}", p.frame_limit,
            frame_limit_call.ends_run() ? ": ending the run" : "; the run ends once the other receivers reach theirs");
    }

    fusa_coe_capture::FusaCoeCaptureCore::BufferView buffer {};
    if (!core_.take_acquired_buffer(buffer)) {
        if (core_.stalled()) {
            // The acquire thread already logged this at ERROR (see
            // acquire_buffer_thread_func()); only compute() can end the run, so it
            // was woken via frame_ready_callback_ to do so here. Fail it, rather than
            // request a stop that reads like Ctrl+C: the run ends kFailed and its
            // diagnostic records the stall, as HSDK4's callers of hololink_module's
            // FusaCoeCaptureOp failed on its capture_stalled().
            return holoscan::make_unexpected(holoscan::Error {
                holoscan::ErrorCode::kTimeout,
                fmt::format("CoE capture stalled on \"{}\" for {}ms without a frame", p.name,
                    core_.stall_duration_ms()) });
        }
        // No frame: compute() runs once per frame it can take, so this does not happen
        // short of a stall.
        return {};
    }
    // Frames the acquire thread replaced with a newer one because compute() did not run
    // before it arrived: lost to a pipeline that is behind the camera, like a full queue.
    if (const uint64_t replaced = core_.take_replaced_frame_count()) {
        count_dropped_frames(replaced_frames_, replaced, "replaced by a newer frame");
    }

    // Host reception time, as close to acquisition as possible -- mirrors the
    // other ported receivers (LinuxReceiver, RoceReceiver, LinuxCoeReceiver),
    // whose received_s/received_ns this operator's output previously left at 0.
    struct timespec received_time { };
    if (clock_gettime(CLOCK_REALTIME, &received_time) != 0) {
        HSB_LOG_ERROR("FusaCoeCaptureOp: clock_gettime failed, errno={}", errno);
    }

    // Releases the acquired buffer on any early return below -- it's a single-slot
    // resource in a 4-buffer/leg pool (buffer_in_compute_, fusa_coe_capture_core.hpp),
    // so an un-released error path permanently removes one of 4 buffers from the
    // leg's pool. Dismissed once the buffer's release becomes the async host
    // callback's responsibility instead (see below).
    OnScopeExit release_fusa_buffer([this, &buffer] {
        auto* release_ptr = static_cast<uint8_t*>(buffer.cuda_device_ptr) + core_.start_byte();
        core_.register_pending_output(release_ptr);
        fusa_coe_capture::FusaCoeCaptureCore::buffer_release_callback(release_ptr);
    });

    // Decode metadata from the CPU-mapped NvSciBuf region.
    hololink::CaptureMetadataT frame_metadata_out;
    frame_metadata_out.received_s = received_time.tv_sec;
    frame_metadata_out.received_ns = received_time.tv_nsec;
    {
        fusa_coe_capture::CoeFrameMetadata frame_metadata {};
        LegacyCoeMetadataDecoder decoder;
        if (core_.decode_metadata(buffer, decoder, frame_metadata)) {
            frame_metadata_out.frame_number = frame_metadata.frame_number;
            frame_metadata_out.timestamp_s = frame_metadata.timestamp_s;
            frame_metadata_out.timestamp_ns = frame_metadata.timestamp_ns;
            frame_metadata_out.metadata_s = frame_metadata.metadata_s;
            frame_metadata_out.metadata_ns = frame_metadata.metadata_ns;
            frame_metadata_out.crc = frame_metadata.crc;
            frame_metadata_out.bytes_written = frame_metadata.bytes_written;
        }
    }

    const bool full_csi = p.include_csi_frame && !p.frame_size_bytes;
    const size_t source_offset = full_csi ? 0 : core_.start_byte();
    const size_t frame_data_size = static_cast<size_t>(core_.bytes_per_line()) * core_.pixel_height()
        + (full_csi ? static_cast<size_t>(core_.start_byte()) + p.geometry.trailing_bytes : 0);
    if (frame_data_size > kMaxFrameBytes) {
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kInvalidArgument,
            fmt::format("frame_data_size {:#x} exceeds pool ceiling kMaxFrameBytes {:#x}",
                frame_data_size, kMaxFrameBytes) });
    }
    if (frame_data_size > static_cast<size_t>(INT_MAX)) {
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kFailure,
            "frame_data_size exceeds INT_MAX" });
    }

    static constexpr DLDataType kUInt8 { static_cast<uint8_t>(kDLUInt), 8, 1 };
    const std::array<int64_t, 1> shape { static_cast<int64_t>(frame_data_size) };

    // Borrow a pre-allocated pool slot.
    auto loan = tensor.allocate_tensor(
        holoscan::TensorLoanRequest { .shape = shape, .dtype = kUInt8 });
    if (!loan) {
        const auto code = loan.error().code;
        if (code == holoscan::ErrorCode::kResourceExhausted) {
            // Back-pressure: release the FuSa buffer (scope guard above); the acquire thread
            // notifies again for the next frame.
            count_dropped_frames(dropped_frames_, 1, "tensor pool exhausted");
            return {};
        }
        HSB_LOG_ERROR("FusaCoeCaptureOp: allocate_tensor failed: {}",
            holoscan::error_code_name(code));
        return holoscan::make_unexpected(std::move(loan).error());
    }

    if (p.cpu_output) {
        // Host write phase: buffer.cpu_ptr is NvSciBuf's own CPU-visible mapping of the same
        // memory buffer.cuda_device_ptr maps on the device side (FusaCoeCaptureCore::BufferView
        // carries both) -- a plain host memcpy, no CUDA copy or stream involved.
        auto writer = loan->write_host();
        if (!writer) {
            HSB_LOG_ERROR("FusaCoeCaptureOp: loan->write_host() failed: {}",
                holoscan::error_code_name(writer.error().code));
            return holoscan::make_unexpected(std::move(writer).error());
        }
        const void* src = static_cast<const uint8_t*>(buffer.cpu_ptr) + source_offset;
        std::memcpy(writer->data(), src, frame_data_size);
        auto committed = std::move(*writer).commit();
        if (!committed) {
            HSB_LOG_ERROR("FusaCoeCaptureOp: commit() failed: {}",
                holoscan::error_code_name(committed.error().code));
            return holoscan::make_unexpected(std::move(committed).error());
        }
    } else {
        // Enter write phase on the compute stream.
        auto writer = loan->write(context.cuda_stream());
        if (!writer) {
            HSB_LOG_ERROR("FusaCoeCaptureOp: loan->write() failed: {}",
                holoscan::error_code_name(writer.error().code));
            return holoscan::make_unexpected(std::move(writer).error());
        }

        // Copy image data from NvSciBuf-backed device memory to the pool slot.
        const void* src = static_cast<const uint8_t*>(buffer.cuda_device_ptr) + source_offset;
        if (cudaMemcpyAsync(
                writer->data(), src, frame_data_size,
                cudaMemcpyDeviceToDevice,
                context.cuda_stream().get())
            != cudaSuccess) {
            return holoscan::make_unexpected(holoscan::Error {
                holoscan::ErrorCode::kFailure,
                "cudaMemcpyAsync failed in FusaCoeCaptureOp::compute" });
        }

        // Commit: records CUDA readiness event after the copy so downstream
        // consumers can enqueue work without explicit synchronisation.
        auto committed = std::move(*writer).commit();
        if (!committed) {
            HSB_LOG_ERROR("FusaCoeCaptureOp: commit() failed: {}",
                holoscan::error_code_name(committed.error().code));
            return holoscan::make_unexpected(std::move(committed).error());
        }
    }

    // Register the FuSa buffer as pending, then schedule its release via a host
    // callback that fires after the cudaMemcpyAsync completes on the GPU stream.
    // This allows the NvFusaCapture buffer pool to be recycled as soon as the
    // copy is done, without blocking the compute thread. From here on, the async
    // callback owns the release -- dismiss the scope guard above.
    release_fusa_buffer.dismiss();
    auto* release_ptr = static_cast<uint8_t*>(buffer.cuda_device_ptr) + core_.start_byte();
    core_.register_pending_output(release_ptr);
    if (cudaLaunchHostFunc(
            context.cuda_stream().get(),
            [](void* data) { fusa_coe_capture::FusaCoeCaptureCore::buffer_release_callback(data); },
            release_ptr)
        != cudaSuccess) {
        // Fallback: synchronise and release immediately.
        cudaStreamSynchronize(context.cuda_stream().get());
        fusa_coe_capture::FusaCoeCaptureCore::buffer_release_callback(release_ptr);
    }

    holoscan::EmitOptions emit_opts;
    // Stamp the time compute() took the frame.
    auto now = context.clock().now();
    emit_opts.capture_time = now ? *now : context.activation_time();
    emit_opts.frame_id = static_cast<uint64_t>(frame_metadata_out.frame_number);

    // Two correlated emits, both stamped with the same capture_time/frame_id so a
    // downstream consumer can pair them on an exact capture_time match (see
    // kMetadataPairingWindow in capture_metadata_utils.hpp) and check that they
    // describe the same frame (receive_paired_tensor_metadata()).
    auto emitted = hololink::operators::emit_tensor_and_metadata(
        [&] { return tensor.emit(std::move(*loan), emit_opts); }, metadata, frame_metadata_out,
        emit_opts);
    if (!emitted) {
        HSB_LOG_ERROR("FusaCoeCaptureOp: publishing frame {} failed: {}",
            frame_metadata_out.frame_number, holoscan::runtime_error_message(emitted.error()));
        return holoscan::make_unexpected(std::move(emitted).error());
    }
    if (*emitted == hololink::operators::PairedEmit::kDropped) {
        // A downstream queue is full: the frame is dropped rather than failing the run.
        count_dropped_frames(dropped_frames_, 1, "downstream queue full");
    }

    return {};
}

void FusaCoeCaptureOp::count_dropped_frames(
    hololink::operators::DropCounter& counter, uint64_t frames, const char* reason)
{
    if (const uint64_t dropped = counter.count(frames)) {
        HSB_LOG_WARN("FusaCoeCaptureOp: dropped {} frame(s) in the last second ({}): the pipeline "
                     "is behind the camera",
            dropped, reason);
    }
}

bool FusaCoeCaptureOp::frame_ready()
{
    auto result = sender_.post();
    if (result) {
        return true;
    }
    const auto code = result.error().code;
    switch (code) {
    case holoscan::ErrorCode::kNotReady:
        // Sender not armed yet -- ask the caller to retry shortly.
        return false;
    case holoscan::ErrorCode::kBackpressured:
        // The source is full. compute() runs once per accepted notification and the
        // acquire thread posts only one per frame compute() can take, so this one must be
        // accepted later -- ask the caller to retry shortly.
        return false;
    case holoscan::ErrorCode::kClosed:
    case holoscan::ErrorCode::kStaleEndpoint:
        // Terminal: the run is ending (sources close when dispatch ends, before stop()). Not
        // retried, so a frame left in the slot is never taken; stop() returns it to the driver.
        break;
    default:
        HSB_LOG_ERROR("FusaCoeCaptureOp::frame_ready: sender_.post() failed: {}",
            holoscan::error_code_name(code));
        break;
    }
    return true;
}

holoscan::LifecycleStatus FusaCoeCaptureOp::on_arm(
    holoscan::LifecycleContext& context) noexcept
{
    auto sender = context.notification_sender(source_);
    if (!sender) {
        HSB_LOG_ERROR("FusaCoeCaptureOp::on_arm: failed to acquire NotificationSender: {}",
            holoscan::error_code_name(sender.error().code));
        return holoscan::LifecycleStatus::kFatalFailure;
    }
    sender_ = std::move(*sender);
    return holoscan::LifecycleStatus::kOk;
}

namespace {

    // Adapter: captures the geometry a camera driver computes via the CsiConverter
    // protocol, using FusaCoeCaptureOp/HSB-packetizer alignment rules (distinct from
    // CameraSensor's built-in CsiToBayerOp-flavored adapter — see the declaration
    // comment on FusaCoeCaptureOp::make_geometry_adapter()).
    class GeometryCaptureConverter : public hololink::csi::CsiConverter {
    public:
        GeometryCaptureConverter(hololink::csi::CsiGeometry& out, bool packetizer_enabled)
            : out_(out)
            , packetizer_enabled_(packetizer_enabled)
        {
        }

        uint32_t receiver_start_byte() override
        {
            return fusa_coe_capture::FusaCoeCaptureCore::receiver_start_byte();
        }

        uint32_t received_line_bytes(uint32_t line_bytes) override
        {
            return fusa_coe_capture::FusaCoeCaptureCore::received_line_bytes(
                line_bytes, packetizer_enabled_);
        }

        uint32_t transmitted_line_bytes(
            hololink::csi::PixelFormat pixel_format, uint32_t pixel_width) override
        {
            return fusa_coe_capture::FusaCoeCaptureCore::transmitted_line_bytes(
                pixel_format, pixel_width, packetizer_enabled_);
        }

        void configure(uint32_t start_byte, uint32_t received_bytes_per_line,
            uint32_t pixel_width, uint32_t pixel_height,
            hololink::csi::PixelFormat pixel_format,
            uint32_t trailing_bytes = 0) override
        {
            out_ = { start_byte, received_bytes_per_line, pixel_width, pixel_height,
                pixel_format, trailing_bytes };
        }

    private:
        hololink::csi::CsiGeometry& out_;
        bool packetizer_enabled_;
    };

} // namespace

std::shared_ptr<hololink::csi::CsiConverter> FusaCoeCaptureOp::make_geometry_adapter(
    hololink::csi::CsiGeometry& out, bool packetizer_enabled)
{
    return std::make_shared<GeometryCaptureConverter>(out, packetizer_enabled);
}

} // namespace hololink::operators
