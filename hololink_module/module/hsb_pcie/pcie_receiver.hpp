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

#ifndef HOLOLINK_MODULE_PCIE_PCIE_RECEIVER_HPP
#define HOLOLINK_MODULE_PCIE_PCIE_RECEIVER_HPP

/**
 * @file
 * @brief Receives frames the Hololink PCIe endpoint DMAs straight into GPU memory, on the
 * module-native path.
 *
 * Module-native receiver, ported from the reference bring-up tree's PcieRxRing for the
 * hololink_module framework, with the same split the RoCE path has between
 * RoceReceiverInterfaceV1 and RoceDataChannelInterfaceV1: the receiver owns the memory
 * (dma-buf import of the caller's buffer), the MSI-X routing lookup and the completion wait through
 * its own PcieDeviceAccessInterface, and touches no register; it implements
 * hololink::module::hsb_pcie::PcieReceiverAttachmentV1 so the hsb_pcie module's
 * PcieDataChannelInterfaceV1::attach_receiver(receiver, chunk_pages) can read the
 * destination / routing values off it and program the endpoint. The per-frame surface
 * (get_next_frame / frames_ready / set_frame_ready / blocking_monitor / close) follows the
 * shape of RoceReceiverInterfaceV1 so PcieReceiverOp mirrors RoceReceiverOp. Nothing here
 * depends on src/hololink/core, and nothing here names HololinkInterfaceV1.
 */

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <cuda.h>

#include "pcie_device_access.hpp"
#include "pcie_rx_layout.hpp"
#include <hololink/module/hsb_pcie/pcie_data_channel.hpp>

namespace hololink::module::pcie {

/**
 * @brief One frame's metadata block, as written by the endpoint into the tail of its slot.
 */
struct RxMetadata {
    uint32_t slot { 0 };
    uint32_t flags { 0 };
    uint64_t valid_bytes { 0 }; ///< bytes written into the slot for this frame
    uint16_t frame_number { 0 };
    /// The block differs from the fill sentinel: the endpoint has written it.
    bool written { false };
    /// The block is an HSB metadata block and the fields above were decoded from it. A
    /// written block that is not one (non-zero padding) is never delivered as a frame.
    bool parsed { false };

    /// Payload ended early relative to the configured geometry.
    bool unsync() const { return flags & 0x1u; }
    /// Bytes-per-word-only frame; no image payload.
    bool bpw_only() const { return flags & 0x2u; }
};

/**
 * @brief Decode a 128-byte metadata block.
 *
 * The endpoint writes the 48-byte HSB end-of-frame block -- flags at 0x00, valid_bytes at
 * 0x18, frame_number at 0x22, all big-endian -- followed by zero padding to the end of the
 * block; FrameMetadataInterfaceV1 decodes the same bytes. A block that still holds the fill
 * sentinel was never written (`written` false); one that was written but whose padding is
 * not zero is not an HSB metadata block (`written` true, `parsed` false) and no field is
 * taken from it.
 *
 * @param raw The metadata block; at least RX_METADATA_BYTES.
 * @param size Bytes available at `raw`.
 * @param slot Slot the block came from, copied into the result.
 * @param fill Byte the ring was pre-filled with; an all-`fill` block means "not written".
 */
RxMetadata parse_rx_metadata(const uint8_t* raw, size_t size, uint32_t slot, uint8_t fill);

/**
 * @brief What the receiver publishes for one completed frame.
 */
struct PcieFrameInfo {
    uint64_t frame_number { 0 }; ///< 16-bit wire frame number, extended across wraparound
    uint64_t received_s { 0 }; ///< CLOCK_REALTIME when the completion was observed
    uint64_t received_ns { 0 };
    CUdeviceptr frame_memory { 0 }; ///< device address of the slot just written
    CUdeviceptr metadata_memory { 0 }; ///< device address of that slot's metadata block
    uint32_t page_number { 0 }; ///< the slot
    uint32_t dropped { 0 }; ///< frames the endpoint completed while we were not looking
    uint64_t bytes_written { 0 }; ///< valid_bytes from the metadata block
    uint32_t flags { 0 }; ///< RxMetadata flags (unsync, bpw_only)
    uint64_t frames { 0 }; ///< completions observed since start()
    /// Host copy of the slot's 128-byte metadata block (the HSB end-of-frame block, zero
    /// padded), for a caller that wants to decode it through FrameMetadataInterfaceV1.
    std::array<uint8_t, RX_METADATA_BYTES> raw_metadata {};
};

/**
 * @brief The module-native PCIe receiver: ring memory, MSI-X routing and the completion wait.
 *
 * Lifecycle, operator-driven like RoceReceiverInterfaceV1:
 *   - start(): validate the ring, make the frame buffer addressable by the endpoint (dma-buf
 *     import), read this vector's MSI-X routing (resetting its counter) and baseline the
 *     completion counter. No register is written.
 *   - the caller hands `this` (as a shared_ptr<PcieReceiverAttachmentV1>) to
 *     PcieDataChannelInterfaceV1::attach_receiver(receiver, chunk_pages), which programs the
 *     endpoint's PCIE_DP block and MSI-X entry from the accessors below and arms;
 *   - blocking_monitor() runs on the caller's thread until stop() / close();
 *   - the caller detaches the channel, then close() releases the mapping.
 */
class PcieReceiver : public hsb_pcie::PcieReceiverAttachmentV1 {
public:
    /**
     * @param access The receiver's own device handle: mapping, MSI-X info, completion waits.
     * @param config Ring configuration: frame size, page size and page count fix the
     *        layout; `config.sensor` is also the MSI-X vector.
     * @param buffer Device memory the endpoint writes into (the caller's CUDA allocation).
     * @param buffer_size Size of that allocation; must cover required_bytes().
     * @throw std::runtime_error when `access` is null or its identity cannot be read.
     */
    PcieReceiver(std::shared_ptr<PcieDeviceAccessInterface> access,
        const RxRingConfig& config,
        CUdeviceptr buffer,
        size_t buffer_size);

    ~PcieReceiver() override;

    PcieReceiver(const PcieReceiver&) = delete;
    PcieReceiver& operator=(const PcieReceiver&) = delete;

    /**
     * @brief Map the frame buffer, read this vector's MSI-X routing (zeroing
     * its counter) and baseline the completion counter. Writes no register: the channel's
     * attach_receiver() programs and arms the endpoint afterwards.
     * @return false if any of that failed; the receiver is left stopped with nothing mapped.
     */
    bool start();

    /**
     * @brief Stop the monitor; wakes get_next_frame(). Safe to call twice. Writes no register
     * -- disarming is the channel's detach_receiver().
     *
     * The mapping stays valid until close(), so a monitor thread that is mid-frame when
     * stop() runs finishes that frame on live memory.
     */
    void stop();

    /**
     * @brief stop(), then wait for blocking_monitor() to return, then release the mapping.
     *
     * Blocks until the monitor thread has left blocking_monitor() (bounded by the monitor's
     * MONITOR_WAIT_MS wait plus one metadata read), unless called from the monitor thread
     * itself. Callers detach the channel before this and join their monitor thread after.
     * Safe to call twice.
     */
    void close();

    /**
     * @brief Monitor loop body: blocks on the completion interrupt until stop() / close().
     *
     * Runs on a thread the caller owns, which must have a CUDA context current: the default
     * read_slot_metadata() copies each frame's metadata block out with cuMemcpyDtoH.
     * Subclasses that override read_slot_metadata() need none.
     */
    void blocking_monitor();

    /// Block until the next completed frame is available; false if `timeout_ms` elapses.
    bool get_next_frame(unsigned timeout_ms, PcieFrameInfo& info);

    /// @return false if get_next_frame() might block.
    bool frames_ready();

    /// Called from the monitor thread each time a frame lands. May be set or cleared from
    /// any thread at any time; the monitor picks the change up at its next frame.
    void set_frame_ready(std::function<void()> frame_ready);

    // ---- PcieReceiverAttachmentV1: what the channel programs the endpoint with -------------
    /// Device address the endpoint is to write to. 0 before start().
    uint64_t iova() const override { return mapping_.iova; }
    /// The frame size: the endpoint writes exactly this many bytes per slot (BUF_LEN).
    uint64_t window_bytes() const override { return layout_.frame_bytes; }
    uint32_t buf_inc_pages() const override { return layout_.buf_inc_pages; }
    uint32_t num_buffers() const override { return layout_.num_buffers; }
    /// From the device identity (HL_PCIE_IOC_IDENTITY, sysfs fallback).
    uint32_t requester_id() const override { return identity_.requester_id; }
    /// The sensor index: one MSI-X vector per sensor on this endpoint. start() enables it
    /// through the driver (HL_PCIE_IOC_MSIX_CTRL); close() disables it.
    uint32_t vector() const override { return config_.sensor; }
    /// Always false after a successful start(): this receiver finds frames through the
    /// metadata blocks, so start() rejects a config asking for MD_DISABLE.
    bool disable_metadata() const override { return config_.disable_metadata; }

    // ---- Ring layout ------------------------------------------------------------------------
    /// The frame buffer in use: the one the constructor was given.
    CUdeviceptr frame_buffer() const { return buffer_; }
    uint64_t metadata_offset() const { return layout_.metadata_offset; }
    uint64_t slot_stride() const { return layout_.slot_stride; }
    uint64_t required_bytes() const { return layout_.required_bytes(); }
    uint64_t slot_offset(uint32_t slot) const { return layout_.slot_offset(slot); }
    const RxRingLayout& layout() const { return layout_; }
    const RxRingConfig& config() const { return config_; }
    const PcieDeviceIdentity& identity() const { return identity_; }
    /// Number of MSI-X vectors the device reported during start(); 0 before.
    uint32_t msix_vectors() const { return msix_vectors_; }
    /// True between a successful start() and close(): this receiver holds its vector enabled.
    bool vector_enabled() const { return vector_enabled_; }
    /// Times the monitor gave up on the last delivered frame number because every completed
    /// frame looked older than it for RESYNC_AFTER_MISSES completions (frame counter restart).
    uint32_t resyncs() const { return resyncs_; }

    bool running() const { return running_; }

protected:
    /**
     * @brief Copy one slot's 128-byte metadata block into `block`.
     *
     * cuMemcpyDtoH out of the frame buffer. Virtual so tests can feed the monitor whatever
     * the endpoint "wrote" with no CUDA context and no endpoint.
     */
    virtual void read_slot_metadata(uint32_t slot, uint8_t* block);

    /**
     * @brief Zero every slot's metadata block before the channel arms.
     *
     * The monitor discovers written slots by scanning metadata, so stale blocks left in a
     * reused buffer must not look like frames of this run. cuMemsetD8 per block. Virtual so
     * tests without CUDA can make it a no-op.
     */
    virtual void clear_metadata_blocks();

private:
    /// Why the monitor could not pick a frame: every slot's decoded block and the rule that
    /// rejected it, for the warning path. Sets `saw_behind` when at least one decoded, unseen
    /// block was rejected only for being older than the last delivered frame.
    std::string describe_slots(bool& saw_behind);

    std::shared_ptr<PcieDeviceAccessInterface> access_;
    CUdeviceptr buffer_;
    size_t buffer_size_;
    RxRingConfig config_;
    RxRingLayout layout_;
    PcieDeviceIdentity identity_ {};

    DmaMapping mapping_ {};
    bool mapped_ { false };
    uint32_t msix_vectors_ { 0 };
    bool vector_enabled_ { false };

    std::atomic<bool> running_ { false };
    uint32_t last_irq_count_ { 0 };
    uint64_t frames_ { 0 };
    uint32_t dropped_ { 0 };
    uint16_t last_frame_number_ { 0 };
    bool have_last_frame_ { false };
    uint64_t extended_frame_number_ { 0 };
    uint64_t delivered_ { 0 }; ///< frames handed to get_next_frame()
    unsigned miss_streak_ { 0 }; ///< consecutive completions with no deliverable frame
    uint64_t misses_ { 0 }; ///< total such completions
    uint32_t resyncs_ { 0 };
    // Per slot, the wire frame number of the frame last delivered from it (-1: none this
    // run), so a slot whose block has not been rewritten since is not reported twice. One
    // entry per slot, never cleared mid-run: a block is the same block for as long as the
    // endpoint has not overwritten it, however many frames have gone by, and the resync
    // path (which drops the last-delivered reference) relies on that staying true.
    std::vector<int32_t> delivered_frame_per_slot_;
    bool slot_delivered(uint32_t slot, uint16_t frame_number) const
    {
        return delivered_frame_per_slot_[slot] == static_cast<int32_t>(frame_number);
    }

    /// Distance from the last delivered wire frame number, modulo 2^16.
    uint16_t frame_distance(uint16_t frame) const;
    /// Scan every slot for the newest complete, not-yet-delivered frame; false when none.
    bool find_newest_frame(std::array<uint8_t, RX_METADATA_BYTES>& block, uint32_t& slot, RxMetadata& rx);

    std::mutex mutex_;
    std::condition_variable frame_available_;
    bool frame_pending_ { false };
    PcieFrameInfo pending_ {};

    // Set for the lifetime of blocking_monitor() (under mutex_) so close() can wait for the
    // monitor to leave before the mapping it reads from is released.
    bool monitor_active_ { false };
    std::thread::id monitor_thread_id_ {};
    std::condition_variable monitor_exited_;

    std::function<void()> frame_ready_;
};

} // namespace hololink::module::pcie

#endif /* HOLOLINK_MODULE_PCIE_PCIE_RECEIVER_HPP */
