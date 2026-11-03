// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>

#include <fmt/format.h>

#include <capture_metadata_traits.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/sensor_io/provenance.hpp>

namespace hololink::operators {

// The OnAligned window of every two-port Hololink operator's input (see each operator's
// contract()): a payload sample pairs only with the metadata sample that has the identical
// capture time. A producer stamps a frame's -- or a sub-frame band's -- payload and metadata with
// the time of the activation that emits them, and every later hop passes that time on unchanged
// (propagate_emit_options() below, sensor_io::derive_from() in SDK operators), so samples of
// different frames or bands never align, whichever route each sample took. A sample whose partner
// never comes, such as a payload whose metadata emit was dropped, is retired by OnAligned once
// the other port's head is newer, and the next frame pairs as usual.
inline constexpr std::chrono::nanoseconds kMetadataPairingWindow { 0 };

// One correlated payload + metadata pair, received and identity-validated together by
// receive_paired_tensor_metadata() below. PayloadT is whatever the operator's own tensor-like
// input port carries -- holoscan::Tensor for a Tensor-domain operator, holoscan::schema::ImageT
// for an image-domain one -- so this stays the single shared pairing helper across both.
template <typename PayloadT>
struct PairedTensorMetadata {
    holoscan::Sample<PayloadT> tensor;
    holoscan::Sample<hololink::CaptureMetadataT> metadata;
};

namespace detail {
    // Recovers PayloadT from an Input<PayloadT> argument so callers never have to name it
    // explicitly -- receive_paired_tensor_metadata(input_tensor, input_metadata) keeps working
    // unchanged whether input_tensor is Input<Tensor> or Input<schema::ImageT>.
    template <typename T>
    struct InputPayloadType; // no primary definition -- only Input<T> specializes it, so
                             // instantiating this with anything else is a compile error.
    template <typename T>
    struct InputPayloadType<holoscan::Input<T>> {
        using type = T;
    };
    template <typename InputT>
    using input_payload_type_t = typename InputPayloadType<InputT>::type;
} // namespace detail

// Receives the payload sample and the metadata sample that OnAligned paired on an operator's two
// input ports, and checks that they describe the same frame: the payload's frame_id, stamped at
// the upstream emit, must equal the metadata's own frame_number. Since the two already share a
// capture time, a mismatch means an upstream operator stamped the wrong identity. The pair is
// then dropped rather than passed on or allowed to fail the run: the result is std::nullopt, and
// the caller counts the drop and reports it. Any receive() error is returned.
template <typename InputTensorT, typename InputMetadataT>
[[nodiscard]] inline holoscan::expected<
    std::optional<PairedTensorMetadata<detail::input_payload_type_t<InputTensorT>>>,
    holoscan::Error>
receive_paired_tensor_metadata(InputTensorT& input_tensor, InputMetadataT& input_metadata)
{
    using PayloadT = detail::input_payload_type_t<InputTensorT>;
    auto maybe_tensor = input_tensor.receive();
    if (!maybe_tensor) {
        return holoscan::make_unexpected(std::move(maybe_tensor).error());
    }
    auto maybe_metadata = input_metadata.receive();
    if (!maybe_metadata) {
        return holoscan::make_unexpected(std::move(maybe_metadata).error());
    }
    if (maybe_tensor->metadata.frame_id
        != static_cast<std::uint64_t>(maybe_metadata->data.frame_number)) {
        return std::nullopt;
    }
    return PairedTensorMetadata<PayloadT> { std::move(*maybe_tensor), std::move(*maybe_metadata) };
}

// Builds EmitOptions that propagate the ORIGINAL capture_time this frame was received with
// (from a sample already paired via receive_paired_tensor_metadata() above), rather than
// stamping a fresh per-hop timestamp -- so capture_time keeps meaning the real capture instant
// all the way through the pipeline (needed for both the next hop's own pairing window and for
// genuine downstream diagnostics/safety use), along with the frame's identity. The source's
// validity flags (e.g. kDegraded) and trace carry over too, and a source without a capture time
// yields none rather than an invalid one.
[[nodiscard]] inline holoscan::EmitOptions propagate_emit_options(
    const holoscan::SampleMetadata& source, std::uint64_t frame_id)
{
    holoscan::EmitOptions options = holoscan::sensor_io::derive_from(source);
    options.frame_id = frame_id;
    return options;
}

// What emit_tensor_and_metadata() did with one frame.
enum class PairedEmit {
    kPublished, // the payload and its metadata went out (an unconnected output is skipped)
    kDropped, // a downstream queue was full: the frame was dropped or only partly published
};

// Publishes a frame's metadata alone, for an operator that passes on only metadata, under the
// metadata half of emit_tensor_and_metadata()'s policy below: kNotReady (the output isn't
// connected) is ignored; kBackpressured, kPublicationPartial and kPublicationIndeterminate make
// the frame count as dropped; any other error is returned.
[[nodiscard]] inline holoscan::expected<PairedEmit, holoscan::Error> emit_metadata(
    holoscan::Output<hololink::CaptureMetadataT>& metadata_output,
    const hololink::CaptureMetadataT& metadata, const holoscan::EmitOptions& options)
{
    auto emitted = metadata_output.emit(metadata, options);
    if (emitted) {
        return PairedEmit::kPublished;
    }
    switch (emitted.error().code) {
    case holoscan::ErrorCode::kNotReady:
        return PairedEmit::kPublished;
    case holoscan::ErrorCode::kBackpressured:
    case holoscan::ErrorCode::kPublicationPartial:
    case holoscan::ErrorCode::kPublicationIndeterminate:
        return PairedEmit::kDropped;
    default:
        return holoscan::make_unexpected(std::move(emitted).error());
    }
}

// Publishes one frame -- \p emit_payload publishes the payload (tensor or image), then the
// metadata goes out with the same \p options -- under the policy every hololink producer shares,
// so a consumer that falls behind costs frames rather than the run (returning kBackpressured from
// compute() fails the run):
// - payload kBackpressured: the frame is dropped; its metadata is not emitted.
// - payload kNotReady: the payload output isn't connected; the metadata still goes out.
// - payload kPublicationPartial or kPublicationIndeterminate: the payload reached (or may have
//   reached) only some consumers, so the metadata still goes out for them to pair it with, and
//   the frame counts as dropped.
// - metadata kNotReady: the metadata output isn't connected, which apps may do.
// - metadata kBackpressured: a metadata consumer is behind, so the frame counts as dropped. The
//   payload is already out; a consumer pairing it waits until the next frame's metadata arrives
//   and OnAligned retires it (see kMetadataPairingWindow). Connect consumers that take only the
//   metadata, such as printers and frame counters, with kDropOldest so that they never fill up
//   (after a publication that reached only some consumers, SDK 5.0.0.5 also fails the run when
//   the stream ends).
// Any other error is returned. kDropped tells the caller to count the drop.
template <typename EmitPayload>
[[nodiscard]] holoscan::expected<PairedEmit, holoscan::Error> emit_tensor_and_metadata(
    EmitPayload&& emit_payload, holoscan::Output<hololink::CaptureMetadataT>& metadata_output,
    const hololink::CaptureMetadataT& metadata, const holoscan::EmitOptions& options)
{
    PairedEmit outcome = PairedEmit::kPublished;
    holoscan::expected<void, holoscan::Error> payload = std::forward<EmitPayload>(emit_payload)();
    if (!payload) {
        switch (payload.error().code) {
        case holoscan::ErrorCode::kBackpressured:
            return PairedEmit::kDropped;
        case holoscan::ErrorCode::kNotReady:
            break;
        case holoscan::ErrorCode::kPublicationPartial:
        case holoscan::ErrorCode::kPublicationIndeterminate:
            outcome = PairedEmit::kDropped;
            break;
        default:
            return holoscan::make_unexpected(std::move(payload).error());
        }
    }
    auto emitted = emit_metadata(metadata_output, metadata, options);
    if (!emitted) {
        return holoscan::make_unexpected(holoscan::Error { emitted.error().code,
            fmt::format("metadata.emit() failed for frame_number={} after its payload was "
                        "published: {}",
                metadata.frame_number, emitted.error().message) });
    }
    return *emitted == PairedEmit::kDropped ? PairedEmit::kDropped : outcome;
}

// Counts the frames an operator drops and tells it when to report them: at most once per second,
// so sustained drops stay visible without a log line per frame.
class DropCounter {
public:
    // Counts \p drops drops (one by default). Returns how many drops to report now, or 0 while the
    // last report is less than a second old.
    [[nodiscard]] std::uint64_t count(std::uint64_t drops = 1)
    {
        pending_ += drops;
        const auto now = std::chrono::steady_clock::now();
        if (now - last_report_ < std::chrono::seconds(1)) {
            return 0;
        }
        last_report_ = now;
        return std::exchange(pending_, 0);
    }

private:
    std::uint64_t pending_ = 0;
    std::chrono::steady_clock::time_point last_report_ {};
};

// frame_limit_ends_run, a parameter of the receivers and FusaCoeCaptureOp. Their contract()
// completes them after frame_limit compute() calls (complete_after(), which counts every
// successful call, with or without a frame), and HSDK5 then ends the run once every other operator
// has completed too. An operator triggered by one input never completes while another of its
// inputs holds a sample that came after the triggering input's stream ended -- a HolovizOp that
// renders on the left camera's layer and gets the right camera's last frame, or one that draws
// inference overlays that come after the video's last frame -- so such a run never ends by itself.
//
// The operators that set frame_limit_ends_run form one group, and the last of them to reach its
// frame_limit ends the whole run (ExecutionContext::request_completion(): samples still in the
// pipeline are dropped, and the run reports kRequestedCompleted unless a stop or failure takes
// precedence). A stereo run thus ends once both cameras have delivered their frames, and a camera
// that never does keeps the run going, as without frame_limit_ends_run, so that a stall timeout or
// a test's timeout still catches it. HSDK5 gives an operator nothing that names its run, so there
// is one group per program image: per executable, and per Python extension module (the receiver
// libraries are static, so the hololink and hololink_module modules each have their own). Run one
// frame-limited graph at a time, and in Python take its receivers from one of the two modules. An
// operator joins the group once started (every start() completes before the first compute()) and
// leaves it when stopped.
class FrameLimitRunGroup {
public:
    void join()
    {
        const std::lock_guard lock(mutex_);
        ++legs_;
    }

    // \p reached: the leaving operator had reached its frame_limit.
    void leave(bool reached)
    {
        const std::lock_guard lock(mutex_);
        --legs_;
        if (reached) {
            --reached_;
        }
    }

    // An operator reached its frame_limit. Returns true for the group's last one.
    [[nodiscard]] bool reach()
    {
        const std::lock_guard lock(mutex_);
        return ++reached_ == legs_;
    }

private:
    std::mutex mutex_;
    int legs_ = 0;
    int reached_ = 0;
};

// This program image's FrameLimitRunGroup.
[[nodiscard]] inline FrameLimitRunGroup& frame_limit_run_group()
{
    static FrameLimitRunGroup group;
    return group;
}

// One operator's frame_limit_ends_run (see FrameLimitRunGroup): call started() at the end of a
// successful start(), stopped() at the top of stop(), and make a Call at the top of compute().
class FrameLimitRunEnd {
public:
    void started(std::int64_t frame_limit, bool ends_run)
    {
        frame_limit_ = ends_run && frame_limit > 0 ? static_cast<std::uint64_t>(frame_limit) : 0;
        calls_ = 0;
        reached_ = false;
        if (frame_limit_ > 0) {
            frame_limit_run_group().join();
        }
    }

    void stopped()
    {
        if (frame_limit_ > 0) {
            frame_limit_run_group().leave(reached_);
            frame_limit_ = 0;
        }
    }

    // One compute() call. On the call that reaches frame_limit, the operator has reached it, and
    // if it is the group's last, the run ends as compute() returns.
    class Call {
    public:
        Call(FrameLimitRunEnd& end, holoscan::ExecutionContext& context)
            : context_(context)
        {
            if (end.frame_limit_ > 0 && !end.reached_ && ++end.calls_ == end.frame_limit_) {
                end.reached_ = true;
                reached_ = true;
                ends_run_ = frame_limit_run_group().reach();
            }
        }
        Call(const Call&) = delete;
        Call& operator=(const Call&) = delete;
        ~Call()
        {
            if (ends_run_) {
                context_.request_completion();
            }
        }

        // This call reaches the operator's frame_limit.
        [[nodiscard]] bool reached() const noexcept { return reached_; }
        // ... and ends the run.
        [[nodiscard]] bool ends_run() const noexcept { return ends_run_; }

    private:
        holoscan::ExecutionContext& context_;
        bool reached_ = false;
        bool ends_run_ = false;
    };

private:
    std::uint64_t frame_limit_ = 0;
    std::uint64_t calls_ = 0;
    bool reached_ = false;
};

// Copies every CaptureMetadata field through unchanged. Unlike HololinkFrameT,
// CaptureMetadataT holds no Tensor (no pointer/ownership fields at all -- every
// field is a plain scalar), so a whole-struct copy is always safe and complete; this
// helper exists so every operator uses the same "copy through by default" idiom
// instead of hand-copying fields individually, which has previously dropped fields
// silently (e.g. operator_entry_ns went missing from more than one operator's
// hand-written HololinkFrameT copy). Operators that need to change specific fields
// (e.g. CsiToBayerOp recomputing frame_number/sub_frame_offset/sub_frame_rows/
// operator_entry_ns, or SubFrameCombinerOp resetting sub_frame_offset/sub_frame_rows)
// should call this first, then assign the fields they own.
[[nodiscard]] inline hololink::CaptureMetadataT forward_metadata(
    const hololink::CaptureMetadataT& in)
{
    return in;
}

} // namespace hololink::operators
