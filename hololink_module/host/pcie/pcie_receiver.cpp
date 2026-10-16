/**
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

// PcieReceiver: module-native port of the reference bring-up tree's receiver.
// Same start / monitor / accounting logic (including the b2daa252 newest-slot fix) and the
// same metadata decoder (parse_rx_metadata, from dma_rx_ring.cpp); the register programming
// that used to sit next to them lives in the hsb_pcie module's PcieDataChannelV1, which
// reads this receiver through PcieReceiverAttachmentV1. The device fd is a
// PcieDeviceAccessInterface.

#include "hololink/module/pcie/pcie_receiver.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <string>

#include <fmt/format.h>

#include <hololink/core/deserializer.hpp>
#include <hololink/module/cuda_unique.hpp>
#include <hololink/module/logging.hpp>

namespace hololink::module::pcie {

namespace {

    // Bounded monitor wait, so stop() / close() are noticed promptly.
    constexpr uint32_t MONITOR_WAIT_MS = 100;
    constexpr unsigned METADATA_SETTLE_MS = 50; ///< bound on metadata landing after the MSI-X
    constexpr unsigned METADATA_POLL_MS = 1;
    /// Completions in a row that found nothing deliverable while at least one slot held a
    /// decoded, unseen frame that was only "older" than the last delivered one: at that point
    /// the endpoint's frame counter has evidently restarted (sensor or packetizer reset, a
    /// design reset, a block of another run picked first) and the last delivered number is
    /// no longer a reference. ~0.5 s at the settle bound.
    constexpr unsigned RESYNC_AFTER_MISSES = 8;
    /// Full per-slot dumps accompany the first few misses and then every Nth.
    constexpr uint64_t MISS_DUMP_EVERY = 100;

    /// Fixed-offset field reads out of a slot's metadata block, bounds-checked by the
    /// (header-only) core Deserializer: a fetch past the block fails and leaves the field
    /// untouched.
    struct MetadataFields {
        explicit MetadataFields(const uint8_t* raw, size_t size)
            : block(raw, size)
        {
        }
        bool be32(size_t offset, uint32_t& value) { return block.reset(offset) && block.next_uint32_be(value); }
        bool be16(size_t offset, uint16_t& value) { return block.reset(offset) && block.next_uint16_be(value); }
        bool be64(size_t offset, uint64_t& value) { return block.reset(offset) && block.next_uint64_be(value); }
        bool le32(size_t offset, uint32_t& value) { return block.reset(offset) && block.next_uint32_le(value); }
        bool le64(size_t offset, uint64_t& value) { return block.reset(offset) && block.next_uint64_le(value); }
        hololink::core::Deserializer block;
    };

} // namespace

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

RxMetadata parse_rx_metadata(const uint8_t* raw, size_t size, uint32_t slot, uint8_t fill)
{
    RxMetadata metadata;
    metadata.slot = slot;
    if (!raw || size < RX_METADATA_BYTES) {
        return metadata;
    }

    // An untouched slot still holds the fill pattern the ring was primed with.
    if (std::all_of(raw, raw + RX_METADATA_BYTES, [fill](uint8_t b) { return b == fill; })) {
        return metadata;
    }
    metadata.parsed = true;

    MetadataFields fields(raw, RX_METADATA_BYTES);

    // Current layout, identified by its zero padding from 0x30 to the end of the block.
    if (std::all_of(raw + 0x30, raw + RX_METADATA_BYTES, [](uint8_t b) { return b == 0; })) {
        metadata.layout = RxMetadataLayout::CURRENT;
        fields.be32(0x00, metadata.flags);
        fields.be64(0x18, metadata.valid_bytes);
        fields.be16(0x22, metadata.frame_number);
        return metadata;
    }

    // Legacy HSB WR-IMM payload.
    metadata.layout = RxMetadataLayout::LEGACY;
    fields.be32(0x50, metadata.flags);
    fields.be64(0x68, metadata.valid_bytes);
    fields.be16(0x72, metadata.frame_number);

    // Older bitfiles wrote these little-endian at the head of the block.
    if (metadata.valid_bytes == 0) {
        fields.le64(0x18, metadata.valid_bytes);
    }
    if (metadata.frame_number == 0) {
        uint32_t frame_number = 0;
        if (fields.le32(0x20, frame_number)) {
            metadata.frame_number = static_cast<uint16_t>(frame_number & 0xFFFFu);
        }
    }
    if (metadata.flags == 0) {
        fields.le32(0x00, metadata.flags);
    }
    return metadata;
}

// ---------------------------------------------------------------------------
// PcieReceiver
// ---------------------------------------------------------------------------

PcieReceiver::PcieReceiver(std::shared_ptr<PcieDeviceAccessInterface> access,
    const RxFrameGeometry& geometry,
    const RxRingConfig& config,
    CUdeviceptr buffer,
    size_t buffer_size,
    FrameBufferSource source)
    : access_(std::move(access))
    , buffer_(buffer)
    , buffer_size_(buffer_size)
    , source_(source)
    , geometry_(geometry)
    , config_(config)
    , layout_(RxRingLayout::compute(geometry, config.num_buffers, config.window_bytes))
{
    if (!access_) {
        throw std::runtime_error("PcieReceiver: no device access");
    }
    // The requester id the channel programs into PCIE_DP_REQUESTER_ID; taken once, here, so
    // the attachment values are complete from construction (only iova / msix wait on start()).
    identity_ = access_->identity();
}

PcieReceiver::~PcieReceiver()
{
    close();
}

bool PcieReceiver::start()
{
    if (const char* error = layout_.validate(geometry_)) {
        HSB_LOG_ERROR("PcieReceiver: {}", error);
        return false;
    }
    if (config_.disable_metadata) {
        // blocking_monitor() finds completed frames only by scanning the per-slot metadata
        // blocks; with MD_DISABLE programmed the endpoint writes none and every completion
        // would be a miss. The attachment interface carries the flag for receivers that
        // track frames another way; this one cannot honour it.
        HSB_LOG_ERROR("PcieReceiver: disable_metadata is not supported by this receiver (frames "
                      "are found through the metadata blocks)");
        return false;
    }

    // Selected explicitly by the caller, not inferred from buffer_ being null -- see
    // FrameBufferSource. Endpoints that cannot reach GPU VRAM take the DeviceAccess path.
    PcieStatus status;
    if (source_ == FrameBufferSource::DeviceAccess) {
        buffer_size_ = layout_.required_bytes();
        status = access_->alloc_host_staged(buffer_size_, buffer_, mapping_);
    } else {
        status = access_->import_dmabuf(buffer_, buffer_size_, mapping_);
    }
    if (status != PcieStatus::SUCCESS) {
        HSB_LOG_ERROR("PcieReceiver: could not make the frame buffer addressable by {}: {}",
            access_->description(), to_string(status));
        return false;
    }
    mapped_ = true;

    // The address-generator constraints on the mapping the channel is about to program.
    if (const char* error = layout_.validate_buffer(mapping_.iova, buffer_size_)) {
        HSB_LOG_ERROR("PcieReceiver: {} (iova={:#x} size={:#x})", error, mapping_.iova, buffer_size_);
        close();
        return false;
    }

    // This sensor's vector: check it exists and that MSI-X is up. reset_count zeroes the
    // vector's counter so the baseline wait below starts from a known value.
    const uint32_t vector = config_.sensor;
    MsixVectorInfo info;
    status = access_->msix_info(vector, true, info);
    if (status != PcieStatus::SUCCESS) {
        HSB_LOG_ERROR("PcieReceiver: MSIX_INFO for vector {} on {} failed: {}", vector,
            access_->description(), to_string(status));
        close();
        return false;
    }
    msix_vectors_ = info.num_vectors;
    if (vector >= info.num_vectors) {
        HSB_LOG_ERROR("PcieReceiver: sensor {} needs MSI-X vector {}, but only {} were allocated on {}",
            vector, vector, info.num_vectors, access_->description());
        close();
        return false;
    }
    if (!info.enabled) {
        HSB_LOG_ERROR("PcieReceiver: MSI-X is not enabled on {}", access_->description());
        close();
        return false;
    }

    // Baseline the interrupt counter before the channel arms, so the first wait cannot miss a
    // completion that lands between arming and the first wait_frame().
    status = access_->wait_frame(vector, 0, 0, last_irq_count_);
    if (status == PcieStatus::FAILED) {
        // TIMEOUT only says no completion has landed yet. FAILED is the MSIX_WAIT ioctl
        // itself failing, which every wait in blocking_monitor() would hit too: fail here
        // rather than return success and let the monitor discover it.
        HSB_LOG_ERROR("PcieReceiver: could not baseline MSI-X vector {} on {}: {}", vector,
            access_->description(), to_string(status));
        close();
        return false;
    }

    // Turn the vector on. The kernel owns the MSI-X table, so this is the only interrupt
    // step anywhere in userspace: the driver enable_irq()s the vector for this fd (refcounted
    // across fds) and disables it again in close() or when the fd goes away. Done before the
    // channel arms; a completion that lands before the monitor's first wait is pending in
    // hardware, not lost. An older driver without the ioctl cannot deliver completions to a
    // receiver that no longer programs the table itself, so that is a hard failure.
    uint32_t refs = 0;
    status = access_->msix_ctrl(vector, true, refs);
    if (status != PcieStatus::SUCCESS) {
        HSB_LOG_ERROR("PcieReceiver: could not enable MSI-X vector {} on {}: {}", vector,
            access_->description(), to_string(status));
        close();
        return false;
    }
    vector_enabled_ = true;
    if (refs > 1) {
        HSB_LOG_INFO("PcieReceiver: vector {} on {} is now held enabled by {} fds", vector,
            access_->description(), refs);
    }

    // The monitor finds written slots by scanning metadata rather than tracking the
    // endpoint's write pointer (which persists across runs and is not reset by programming
    // BUF_PTR_RANGE), so no stale block may look like a frame of this run.
    clear_metadata_blocks();
    frames_ = 0;
    dropped_ = 0;
    extended_frame_number_ = 0;
    have_last_frame_ = false;
    delivered_ = 0;
    miss_streak_ = 0;
    misses_ = 0;
    delivered_frame_per_slot_.assign(layout_.num_buffers, -1);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        frame_pending_ = false;
    }
    running_ = true;
    HSB_LOG_INFO("PcieReceiver: ready on {} sensor {} at iova={:#x} (requester_id={:#x} "
                 "window={:#x} slot_stride={:#x} buffers={} msix_vector={} enabled); awaiting "
                 "the channel's attach",
        access_->description(), vector, mapping_.iova, identity_.requester_id,
        layout_.window_bytes, layout_.slot_stride, layout_.num_buffers, vector);
    return true;
}

void PcieReceiver::stop()
{
    // The running_ transition happens under mutex_ so a get_next_frame() caller cannot test
    // [frame_pending_ || !running_], see it false, and then miss the notify below (a lost
    // wakeup that would have stranded it for the full timeout). No register is touched: the
    // endpoint is disarmed by the channel's detach_receiver().
    {
        std::lock_guard<std::mutex> lock(mutex_);
        running_ = false;
    }
    // Wake anyone parked in get_next_frame; the monitor rechecks running_ on return.
    frame_available_.notify_all();
}

void PcieReceiver::close()
{
    stop();

    // A completion that landed before the channel disarmed and the monitor's wait_frame()
    // returned is still being processed: read_slot_metadata() reads the host-staged mapping
    // directly, so the mapping must outlive that read. Wait for the monitor to leave
    // blocking_monitor() before releasing -- unless this IS the monitor thread (a
    // frame_ready callback closing the receiver), which cannot wait for itself.
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (monitor_active_ && monitor_thread_id_ != std::this_thread::get_id()) {
            monitor_exited_.wait(lock, [this] { return !monitor_active_; });
        }
    }

    // Interrupt off before the memory goes: the driver's disable waits for a handler in
    // flight, so nothing can wake a waiter on this vector afterwards. The channel has
    // disarmed by now (detach_receiver precedes close in the operator), so no new completion
    // is in flight either.
    if (vector_enabled_) {
        uint32_t refs = 0;
        const PcieStatus status = access_->msix_ctrl(config_.sensor, false, refs);
        if (status != PcieStatus::SUCCESS) {
            HSB_LOG_WARN("PcieReceiver: could not disable MSI-X vector {} on {}: {} (the driver "
                         "drops it when the fd closes)",
                config_.sensor, access_->description(), to_string(status));
        }
        vector_enabled_ = false;
    }

    if (mapped_) {
        access_->release(mapping_.handle);
        mapped_ = false;
        mapping_ = DmaMapping {};
        if (source_ == FrameBufferSource::DeviceAccess) {
            buffer_ = 0;
        }
    }
}

void PcieReceiver::read_slot_metadata(uint32_t slot, uint8_t* block)
{
    const size_t offset = layout_.slot_offset(slot) + layout_.metadata_offset;

    // Staging path: the frame buffer is host memory that CUDA only maps, so read the block
    // directly rather than routing 128 bytes through the copy engine per frame.
    if (void* host = access_->host_pointer()) {
        std::memcpy(block, static_cast<const uint8_t*>(host) + offset, RX_METADATA_BYTES);
        return;
    }

    // VRAM path: the endpoint wrote the block into device memory alongside the payload, so it
    // has to come out through CUDA. Needs a current context on this thread.
    HOLOLINK_MODULE_CUDA_CHECK(cuMemcpyDtoH(block, buffer_ + offset, RX_METADATA_BYTES));
}

void PcieReceiver::clear_metadata_blocks()
{
    void* host = access_->host_pointer();
    if (host == nullptr) {
        // VRAM path: needs a current CUDA context on this thread, as the import that
        // preceded this call did. A device access that mapped nothing CUDA-visible (tests)
        // has no blocks to clear.
        CUcontext context = nullptr;
        if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || context == nullptr) {
            HSB_LOG_DEBUG("PcieReceiver: no CUDA context current; metadata blocks not cleared");
            return;
        }
    }
    for (uint32_t slot = 0; slot < layout_.num_buffers; slot++) {
        const size_t offset = layout_.slot_offset(slot) + layout_.metadata_offset;
        if (host != nullptr) {
            std::memset(static_cast<uint8_t*>(host) + offset, 0, RX_METADATA_BYTES);
        } else {
            HOLOLINK_MODULE_CUDA_CHECK(cuMemsetD8(buffer_ + offset, 0, RX_METADATA_BYTES));
        }
    }
}

uint16_t PcieReceiver::frame_distance(uint16_t frame) const
{
    return static_cast<uint16_t>(frame - last_frame_number_);
}

bool PcieReceiver::find_newest_frame(
    std::array<uint8_t, RX_METADATA_BYTES>& block, uint32_t& slot, RxMetadata& rx)
{
    // The endpoint's write pointer is not known to the host (it survives re-programming), so
    // the written slot is discovered from the metadata the endpoint leaves in each slot: a
    // decoded block with a non-zero byte count whose (slot, frame_number) has not been
    // delivered yet. Among several candidates (a batched completion), the one furthest
    // ahead of the last delivered frame number wins -- the same rule as the reference
    // design's PcieRxRing::wait_metadata().
    bool found = false;
    uint16_t best_distance = 0;
    std::array<uint8_t, RX_METADATA_BYTES> candidate {};
    for (uint32_t s = 0; s < layout_.num_buffers; s++) {
        candidate.fill(0);
        read_slot_metadata(s, candidate.data());
        const RxMetadata parsed = parse_rx_metadata(candidate.data(), candidate.size(), s, 0);
        if (!parsed.parsed || parsed.valid_bytes == 0) {
            continue;
        }
        if (slot_delivered(s, parsed.frame_number)) {
            continue;
        }
        const uint16_t distance = have_last_frame_ ? frame_distance(parsed.frame_number)
                                                   : static_cast<uint16_t>(parsed.frame_number + 1);
        if (have_last_frame_ && (distance == 0 || distance >= 0x8000)) {
            continue; // older than (or equal to) what was already delivered
        }
        if (!found || distance > best_distance) {
            found = true;
            best_distance = distance;
            slot = s;
            rx = parsed;
            block = candidate;
        }
    }
    return found;
}

std::string PcieReceiver::describe_slots(bool& saw_behind)
{
    saw_behind = false;
    const auto slots_delivered = std::count_if(delivered_frame_per_slot_.begin(),
        delivered_frame_per_slot_.end(), [](int32_t f) { return f >= 0; });
    std::string out = fmt::format("last_delivered={} ({}) slots_delivered={} delivered={} frames={} dropped={}",
        have_last_frame_ ? std::to_string(last_frame_number_) : std::string("none"),
        have_last_frame_ ? "valid" : "no reference yet", slots_delivered, delivered_, frames_,
        dropped_);
    std::array<uint8_t, RX_METADATA_BYTES> block {};
    for (uint32_t s = 0; s < layout_.num_buffers; s++) {
        block.fill(0);
        read_slot_metadata(s, block.data());
        const RxMetadata parsed = parse_rx_metadata(block.data(), block.size(), s, 0);
        const char* reason = "candidate";
        std::string detail;
        if (!parsed.parsed) {
            reason = "untouched (all zero)";
        } else if (parsed.valid_bytes == 0) {
            reason = "zero valid_bytes";
        } else if (slot_delivered(s, parsed.frame_number)) {
            reason = "already delivered from this slot";
        } else if (have_last_frame_) {
            const uint16_t distance = frame_distance(parsed.frame_number);
            if (distance == 0) {
                reason = "same frame number as last delivered";
                saw_behind = true;
            } else if (distance >= 0x8000) {
                reason = "older than last delivered";
                detail = fmt::format(" (behind by {})", 0x10000u - distance);
                saw_behind = true;
            }
        }
        std::string head;
        for (size_t i = 0; i < 0x30; i++) {
            head += fmt::format("{:02x}", block[i]);
            if ((i & 3) == 3 && i + 1 < 0x30) {
                head += ' ';
            }
        }
        out += fmt::format("\n  slot {}: layout={} valid_bytes={} frame={} flags={:#x} -> {}{} | head[0x00..0x2f]={}",
            s,
            parsed.layout == RxMetadataLayout::CURRENT      ? "CURRENT"
                : parsed.layout == RxMetadataLayout::LEGACY ? "LEGACY"
                                                            : "NONE",
            parsed.valid_bytes, parsed.frame_number, parsed.flags, reason, detail, head);
    }
    return out;
}

void PcieReceiver::blocking_monitor()
{
    // Mark the monitor live for the whole loop so close() waits for it (see close()).
    {
        std::lock_guard<std::mutex> lock(mutex_);
        monitor_active_ = true;
        monitor_thread_id_ = std::this_thread::get_id();
    }
    struct MonitorExit {
        PcieReceiver& self;
        ~MonitorExit()
        {
            {
                std::lock_guard<std::mutex> lock(self.mutex_);
                self.monitor_active_ = false;
                self.monitor_thread_id_ = std::thread::id {};
            }
            self.monitor_exited_.notify_all();
        }
    } monitor_exit { *this };

    const uint32_t vector = config_.sensor;
    const uint32_t num_buffers = layout_.num_buffers;
    std::array<uint8_t, RX_METADATA_BYTES> block {};

    while (running_) {
        uint32_t count = last_irq_count_;
        // Bounded rather than infinite so close() is observed promptly without needing to
        // interrupt a blocked wait.
        const PcieStatus status = access_->wait_frame(vector, MONITOR_WAIT_MS, last_irq_count_, count);
        struct timespec received_at { };
        clock_gettime(CLOCK_REALTIME, &received_at);
        if (status == PcieStatus::TIMEOUT) {
            continue;
        }
        if (status != PcieStatus::SUCCESS) {
            HSB_LOG_ERROR("PcieReceiver: wait_frame on {} failed ({}); monitor exiting",
                access_->description(), to_string(status));
            // No frame can arrive after this. Report the receiver stopped so running() says
            // so and get_next_frame() returns at once instead of running out its timeout.
            {
                std::lock_guard<std::mutex> lock(mutex_);
                running_ = false;
            }
            frame_available_.notify_all();
            break;
        }

        // Unsigned wraparound gives the right delta across the counter rolling over.
        const uint32_t completions = count - last_irq_count_;
        last_irq_count_ = count;
        if (completions == 0) {
            continue;
        }

        // More than one completion means frames were produced faster than consumed; only the
        // newest is reported, the rest count as dropped.
        if (completions > 1) {
            dropped_ += completions - 1;
        }
        frames_ += completions;

        // Find the slot the endpoint actually wrote. The metadata block can land a moment
        // after the interrupt, so give it a bounded settle time.
        uint32_t newest_slot = 0;
        RxMetadata rx;
        bool found = false;
        for (unsigned waited_ms = 0;; waited_ms += METADATA_POLL_MS) {
            found = find_newest_frame(block, newest_slot, rx);
            if (found || waited_ms >= METADATA_SETTLE_MS || !running_) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(METADATA_POLL_MS));
        }
        if (!found && !running_) {
            // Stopped while waiting: the completion is a frame cut off by the shutdown (the
            // sensor halted or the channel disarmed under it), whose metadata block will never
            // land. Not a miss worth diagnosing; leave the loop.
            break;
        }
        if (!found) {
            miss_streak_++;
            misses_++;
            bool saw_behind = false;
            const bool dump = misses_ <= 3 || (misses_ % MISS_DUMP_EVERY) == 0;
            std::string slots;
            if (dump || miss_streak_ >= RESYNC_AFTER_MISSES) {
                slots = describe_slots(saw_behind);
            }
            if (dump) {
                HSB_LOG_WARN("PcieReceiver: {} completion(s) on {} vector {} but no new complete "
                             "metadata block in any of {} slots within {} ms (miss {} in a row, {} "
                             "total); {}",
                    completions, access_->description(), vector, num_buffers, METADATA_SETTLE_MS,
                    miss_streak_, misses_, slots);
            } else {
                HSB_LOG_WARN("PcieReceiver: {} completion(s) on {} vector {} but no new complete "
                             "metadata block in any of {} slots within {} ms (miss {} in a row)",
                    completions, access_->description(), vector, num_buffers, METADATA_SETTLE_MS,
                    miss_streak_);
            }
            // Every completed frame has looked older than the last delivered one for a while:
            // the endpoint's frame counter restarted, or the reference came from a block that
            // was not this run's. Drop the reference (keep the delivered keys, so a block
            // already handed out is still not handed out twice) and take the newest of what
            // is there now. The extended frame number restarts from the wire value.
            if (saw_behind && miss_streak_ >= RESYNC_AFTER_MISSES) {
                HSB_LOG_WARN("PcieReceiver: frames on {} vector {} have been older than the last "
                             "delivered frame number {} for {} completions; the endpoint's frame "
                             "counter appears to have restarted -- resynchronizing",
                    access_->description(), vector, last_frame_number_, miss_streak_);
                have_last_frame_ = false;
                miss_streak_ = 0;
                resyncs_++;
                found = find_newest_frame(block, newest_slot, rx);
            }
            if (!found) {
                continue;
            }
        }
        miss_streak_ = 0;
        delivered_frame_per_slot_[newest_slot] = rx.frame_number;

        // Extend the 16-bit wire frame number so downstream sees a monotonic counter.
        if (have_last_frame_) {
            extended_frame_number_ += frame_distance(rx.frame_number);
        } else {
            extended_frame_number_ = rx.frame_number;
        }
        last_frame_number_ = rx.frame_number;
        have_last_frame_ = true;
        delivered_++;
        if (delivered_ == 1) {
            HSB_LOG_INFO("PcieReceiver: first frame on {} vector {}: slot {} frame_number {} "
                         "valid_bytes {} flags {:#x} layout {}",
                access_->description(), vector, newest_slot, rx.frame_number, rx.valid_bytes,
                rx.flags, rx.layout == RxMetadataLayout::CURRENT ? "CURRENT" : "LEGACY");
        }

        PcieFrameInfo info;
        info.frame_number = extended_frame_number_;
        info.received_s = static_cast<uint64_t>(received_at.tv_sec);
        info.received_ns = static_cast<uint64_t>(received_at.tv_nsec);
        info.frame_memory = buffer_ + layout_.slot_offset(newest_slot);
        info.metadata_memory = info.frame_memory + layout_.metadata_offset;
        info.page_number = newest_slot;
        info.dropped = dropped_;
        info.bytes_written = rx.valid_bytes;
        info.flags = rx.flags;
        info.frames = frames_;
        info.metadata_layout = rx.layout;
        info.raw_metadata = block;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (frame_pending_) {
                // The consumer has not collected the previous frame; this one replaces it.
                // That frame is lost to the consumer exactly like one skipped by a batched
                // completion, so it counts the same way.
                dropped_++;
                info.dropped = dropped_;
            }
            pending_ = info;
            frame_pending_ = true;
        }
        frame_available_.notify_one();

        if (frame_ready_) {
            frame_ready_();
        }
    }
}

bool PcieReceiver::get_next_frame(unsigned timeout_ms, PcieFrameInfo& info)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (!frame_available_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
            [this] { return frame_pending_ || !running_; })) {
        return false;
    }
    if (!frame_pending_) {
        return false;
    }
    info = pending_;
    frame_pending_ = false;
    return true;
}

bool PcieReceiver::frames_ready()
{
    std::lock_guard<std::mutex> lock(mutex_);
    return frame_pending_;
}

void PcieReceiver::set_frame_ready(std::function<void()> frame_ready)
{
    frame_ready_ = std::move(frame_ready);
}

} // namespace hololink::module::pcie
