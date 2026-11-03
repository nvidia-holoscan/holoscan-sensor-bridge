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

#ifndef HOLOLINK_MODULE_HSB_PCIE_PCIE_RECEIVER_HPP
#define HOLOLINK_MODULE_HSB_PCIE_PCIE_RECEIVER_HPP

/*
 * PcieReceiverInterfaceV1: the per-channel PCIe receiver, published by the hsb_pcie
 * module and reached through get_service(metadata) -- the PCIe counterpart of
 * RoceReceiverInterfaceV1 (hololink/module/roce_receiver.hpp).
 *
 * Where the RoCE receiver fronts an ibverbs queue pair, this one owns the ring the
 * endpoint DMA-writes into (a caller-provided CUDA allocation, dma-buf exported to the
 * endpoint through the hololink_pcie kernel driver) and the completion wait on the
 * sensor's MSI-X vector. It writes no FPGA register: it implements
 * PcieReceiverAttachmentV1, and the channel (PcieDataChannelInterfaceV1::attach_receiver)
 * programs the endpoint from those values. The operator's sequence is therefore
 *
 *     receiver = PcieReceiverInterfaceV1::get_service(metadata);
 *     receiver->start(buffer, size, frame_size, page_size, pages); // map, enable the vector
 *     channel->attach_receiver(receiver, chunk_pages);             // program + arm the endpoint
 *     thread { receiver->blocking_monitor(); }
 *     ... receiver->get_next_frame(timeout, info) ...
 *     channel->detach_receiver(); receiver->close(); join;
 *
 * Like RoceReceiverInterfaceV1, the receiver knows a frame only by its size in bytes: what
 * those bytes hold (RAW10 lines, a sensor prefix, a stride) is the CSI converter's business,
 * and its length (CsiToBayerOp::get_csi_length) is what the caller passes as cu_frame_size.
 * The caller lays the ring out as RoceReceiverOp does -- `pages` pages, `cu_page_size` apart,
 * in a buffer it allocated (ReceiverMemoryDescriptor) -- and the receiver dma-buf exports it.
 *
 * The implementation lives in the module .so, so a board whose ring or metadata layout
 * differs ships a module publishing a different implementation under the same type_id,
 * with no change on the host -- exactly as the hsb_lite_2510 module does for RoCE.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

#include "hololink/module/data_channel.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hsb_pcie/pcie_data_channel.hpp"
#include "hololink/module/service.hpp"
#include "hololink/module/status.h"

namespace hololink::module::hsb_pcie {

/// Size of the metadata block the endpoint writes after each page's frame. It lands at the
/// first 128-byte boundary after the frame; the endpoint has no register to place it
/// elsewhere, which is why start() takes no metadata_offset.
constexpr size_t PCIE_RX_METADATA_BYTES = 128;

/** Per-frame info populated by PcieReceiverInterfaceV1::get_next_frame. */
class PcieReceiverFrameInfoV1 {
public:
    static constexpr const char* release_version = UNRELEASED;

    /// Device memory holding this frame, as a CUdeviceptr: the page just written.
    uint64_t frame_memory { 0 };
    /// Device memory holding that page's metadata block, as a CUdeviceptr.
    uint64_t metadata_memory { 0 };
    /// The endpoint's 16-bit frame counter, extended across wraparound to a monotonic value.
    uint64_t frame_number { 0 };
    /// CLOCK_REALTIME when the completion was observed, seconds and nanoseconds.
    uint64_t received_s { 0 };
    uint64_t received_ns { 0 };
    /// Bytes the endpoint wrote into the page for this frame (valid_bytes).
    uint64_t bytes_written { 0 };
    /// Completions observed since start(), this one included.
    uint64_t frames { 0 };
    /// The page.
    uint32_t page_number { 0 };
    /// Frames lost to the consumer since start(): completed while another was still
    /// waiting to be collected, or batched behind a newer completion.
    uint32_t dropped { 0 };
    /// Flags word from the metadata block (bit0 unsync, bit1 bytes-per-word only).
    uint32_t flags { 0 };
    /// Host copy of the page's metadata block: the 48-byte HSB end-of-frame block the
    /// endpoint writes, zero-padded to PCIE_RX_METADATA_BYTES. Every frame this receiver
    /// delivers was found through that block, so it is always present; decode it with
    /// FrameMetadataInterfaceV1 (flags / frame_number / bytes_written above are the
    /// receiver's own reading of the same block).
    std::array<uint8_t, PCIE_RX_METADATA_BYTES> raw_metadata {};
};

/**
 * The per-channel PCIe receiver service, keyed "serial=<n>;data_channel=<k>" like the
 * channel it pairs with. Also the PcieReceiverAttachmentV1 the channel programs from, so
 * the service pointer is what attach_receiver() takes.
 *
 * The metadata-form get_service runs configure(metadata) once: the receiver takes the
 * endpoint's PCI address from `peer_ip` and the sensor index (its MSI-X vector) from the
 * keys Adapter::use_sensor stamped, and registers with the per-board Hololink so a lost
 * device invalidates it. The device node is opened in start().
 *
 * Everything that describes a running ring -- the PcieReceiverAttachmentV1 accessors
 * (iova, window_bytes, buf_inc_pages, num_buffers, requester_id, disable_metadata),
 * frame_buffer(), frames_ready(), get_next_frame() and blocking_monitor() -- throws
 * std::runtime_error before start() and after close(): there is no ring to describe, and a
 * default would be programmed into the endpoint. sensor(), vector() and description() are
 * valid from configure() on; stop() and close() may be called at any time.
 */
class PcieReceiverInterfaceV1
    : public ConfigurableService<PcieReceiverInterfaceV1>,
      public PcieReceiverAttachmentV1 {
public:
    static constexpr const char* type_id = "pcie_receiver.v1";
    static constexpr const char* release_version = UNRELEASED;

    /* Per-channel instance_id derivation used by the metadata-form get_service. Same key
     * as the channel's: both services key by (serial, data_channel). */
    static std::string locator_id(const EnumerationMetadata& metadata)
    {
        return DataChannelInterfaceV1::locator_id(metadata);
    }

    ~PcieReceiverInterfaceV1() override = default;

    /**
     * @brief Bring the receiver up: open the endpoint, make the frame buffer addressable
     * by it, enable this sensor's MSI-X vector and baseline its completion counter.
     *
     * Frames are delivered into `cu_buffer`, a single contiguous region holding `pages`
     * pages `cu_page_size` bytes apart; each page holds one frame of `cu_frame_size` bytes
     * followed, at the next 128-byte boundary, by the endpoint's PCIE_RX_METADATA_BYTES
     * metadata block -- the same shape RoceReceiverInterfaceV1::start() takes, without the
     * metadata_offset (the endpoint fixes it).
     *
     * Writes no register; the channel's attach_receiver() programs and arms the endpoint
     * afterwards from this object's PcieReceiverAttachmentV1 values (`cu_frame_size` becomes
     * BUF_LEN, `pages` BUF_PTR_RANGE + 1, `cu_page_size / 128` BUF_INC). Needs the CUDA
     * context that owns `cu_buffer` current on the calling thread.
     *
     * @param cu_buffer CUDA allocation frames are delivered into, as a CUdeviceptr; at
     *      least `cu_page_size * pages` long and 128-byte aligned.
     * @param cu_buffer_size Bytes available at cu_buffer.
     * @param cu_frame_size Bytes the endpoint writes per frame: the CSI converter's length.
     *      Must be positive and fit in 32 bits.
     * @param cu_page_size Bytes from one page to the next; at least
     *      round_up(cu_frame_size, 128) + PCIE_RX_METADATA_BYTES and a multiple of 128.
     * @param pages Pages the endpoint cycles through, in [1, 4096].
     * @return HOLOLINK_MODULE_OK, or an error status when the ring or buffer is invalid or
     *      the device refused the mapping or the vector (the reason is logged).
     * @throws std::runtime_error when the endpoint cannot be reached: `peer_ip` names a
     *      PCI function no hololink_pcie device is bound to, or its node cannot be opened.
     */
    virtual hololink_module_status_t start(uint64_t cu_buffer, size_t cu_buffer_size,
        size_t cu_frame_size, size_t cu_page_size, unsigned pages)
        = 0;

    /**
     * @brief Stop the monitor and wake get_next_frame(). Writes no register (disarming is
     * the channel's detach_receiver()); the mapping stays valid until close(). Safe to
     * call twice or before start().
     */
    virtual void stop() = 0;

    /**
     * @brief stop(), wait for blocking_monitor() to return (unless called from it), disable
     * the vector and release the mapping. Callers detach the channel before this and join
     * their monitor thread after. Safe to call twice or before start().
     */
    virtual void close() = 0;

    /**
     * @brief Run the receiver until stop() / close(). Gets a thread of its own, which must
     * carry the CUDA context the frame buffer belongs to.
     * @throws std::runtime_error if start() has not been called.
     */
    virtual void blocking_monitor() = 0;

    /**
     * @brief Take the next completed frame, waiting up to `timeout_ms` for one.
     * @return True when `info` describes a frame; false when the timeout elapsed first or
     *      the receiver has stopped.
     * @throws std::runtime_error if start() has not been called.
     */
    virtual bool get_next_frame(unsigned timeout_ms, PcieReceiverFrameInfoV1& info) = 0;

    /**
     * @return False when get_next_frame() would block waiting for a frame.
     * @throws std::runtime_error if start() has not been called.
     */
    virtual bool frames_ready() = 0;

    /**
     * @brief Register a callback run on the monitor thread each time a frame becomes
     * available. Register it before blocking_monitor() starts; it stays registered across
     * calls to start(). Pass an empty std::function to stop being called.
     */
    virtual void set_frame_ready(std::function<void()> callback) = 0;

    /**
     * @brief The frame buffer in use, as a CUdeviceptr: the one start() was given.
     * @throws std::runtime_error if start() has not been called.
     */
    virtual uint64_t frame_buffer() = 0;

    /// Sensor interface index this receiver serves; also its PCIE_DP block and MSI-X vector.
    virtual uint32_t sensor() const = 0;

    /// Short description of the endpoint for diagnostics, e.g. "/dev/hololink_pcie0" once
    /// started, the PCI address before.
    virtual std::string description() const = 0;
};

} // namespace hololink::module::hsb_pcie

#endif // HOLOLINK_MODULE_HSB_PCIE_PCIE_RECEIVER_HPP
