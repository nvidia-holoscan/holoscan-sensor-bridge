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

#ifndef HOLOLINK_MODULE_PCIE_PCIE_RX_LAYOUT_HPP
#define HOLOLINK_MODULE_PCIE_PCIE_RX_LAYOUT_HPP

/**
 * @file
 * @brief Receive-ring page layout for the module-native PCIe receiver.
 *
 * The caller lays the ring out the way RoceReceiverOp does: `pages` pages of `page_bytes`
 * each, a frame of `frame_bytes` at the start of every page and the endpoint's 128-byte
 * metadata block right after it, at the next 128-byte boundary. This header records that
 * layout (RxRingConfig -> RxRingLayout), derives the one register value that follows from it
 * (BUF_INC, the page stride in 128-byte units) and checks the endpoint's limits on a ring and
 * on the buffer behind it. What the bytes of a frame mean is the CSI converter's business;
 * the frame size it computes (CsiToBayerOp::get_csi_length) is all the receiver needs.
 * Register programming is not here: the hsb_pcie module's PcieDataChannelInterfaceV1 reads
 * these values off the receiver (PcieReceiverAttachmentV1) and programs the endpoint.
 *
 * Core-free: nothing in this header depends on src/hololink/core.
 */

#include <cstddef>
#include <cstdint>

namespace hololink::module::pcie {

/// Every FPGA buffer pointer counts in 128-byte pages, and BUF_START ignores the low 7 bits.
inline constexpr uint64_t FPGA_PAGE_BYTES = 128;
/// Per-frame metadata block the endpoint writes after the frame, at the next 128-byte boundary.
inline constexpr uint64_t RX_METADATA_BYTES = 128;
/// BUF_PTR_RANGE is 12 bits wide.
inline constexpr uint32_t RX_MAX_BUFFERS = 4096;
/// DP_PKT_LEN is 8 bits wide.
inline constexpr uint32_t RX_MAX_CHUNK_PAGES = 255;

inline constexpr uint64_t align_up(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief Ring configuration: what the receiver needs to map its ring and what the channel
 * programs the endpoint from.
 *
 * The data TLP size (DP_PKT_LEN, `chunk_pages`) is not here: it is a per-endpoint word the
 * channel programs, passed to PcieDataChannelInterfaceV1::attach_receiver. The PCI requester
 * id comes from the receiver's device identity.
 */
struct RxRingConfig {
    uint32_t sensor { 0 }; ///< PCIE_DP block and MSI-X vector
    /// Bytes the endpoint writes per frame (PCIE_DP_BUF_LEN): the CSI converter's length.
    uint64_t frame_bytes { 0 };
    /// Bytes from one page to the next (the slot stride); at least
    /// align_up(frame_bytes, 128) + RX_METADATA_BYTES and a multiple of 128.
    uint64_t page_bytes { 0 };
    uint32_t num_buffers { 1 }; ///< pages in the ring (PCIE_DP_BUF_PTR_RANGE + 1)
    /// Set MD_DISABLE so the endpoint writes no metadata block. PcieReceiver::start() rejects
    /// it (the receiver finds frames through those blocks); for attachments that do not.
    bool disable_metadata { false };
};

/**
 * @brief Page layout of a receive ring: where the metadata block lands, how far apart pages
 * are, and what BUF_INC the endpoint needs. A single-page ring rewrites the same page, so
 * BUF_INC is 0.
 */
struct RxRingLayout {
    uint64_t frame_bytes { 0 }; ///< bytes the endpoint writes per page (BUF_LEN)
    uint64_t metadata_offset { 0 }; ///< offset of the metadata block within a page
    uint64_t slot_stride { 0 }; ///< distance between consecutive pages (page_bytes)
    uint32_t buf_inc_pages { 0 }; ///< PCIE_DP_BUF_INC, in 128-byte pages
    uint32_t num_buffers { 1 };

    /// Total device memory the ring needs.
    uint64_t required_bytes() const { return slot_stride * num_buffers; }

    /// Byte offset of `slot`'s frame from the start of the ring.
    uint64_t slot_offset(uint32_t slot) const { return slot * slot_stride; }

    static RxRingLayout compute(const RxRingConfig& config)
    {
        RxRingLayout layout;
        layout.num_buffers = config.num_buffers;
        layout.frame_bytes = config.frame_bytes;
        // The endpoint writes the block at the first 128-byte boundary after the frame; there
        // is no register to put it anywhere else.
        layout.metadata_offset = align_up(layout.frame_bytes, FPGA_PAGE_BYTES);
        layout.slot_stride = config.page_bytes;
        layout.buf_inc_pages = config.num_buffers > 1
            ? static_cast<uint32_t>(layout.slot_stride / FPGA_PAGE_BYTES)
            : 0;
        return layout;
    }

    /**
     * @brief The endpoint's limits on a ring of this layout.
     * @return nullptr when the ring is usable, else why it is not.
     */
    const char* validate() const
    {
        if (frame_bytes == 0) {
            return "frame_bytes must be positive";
        }
        if (frame_bytes > 0xFFFFFFFFull) {
            return "frame_bytes must fit in the 32-bit BUF_LEN register";
        }
        if (num_buffers < 1) {
            return "num_buffers must be at least 1";
        }
        if (num_buffers > RX_MAX_BUFFERS) {
            return "num_buffers must fit in the 12-bit endpoint buffer pointer";
        }
        if (slot_stride < metadata_offset + RX_METADATA_BYTES) {
            return "page_bytes must cover the frame and the 128-byte metadata block after it";
        }
        if (slot_stride % FPGA_PAGE_BYTES) {
            return "page_bytes must be 128-byte aligned for the endpoint buffer increment";
        }
        return nullptr;
    }

    /**
     * @brief The endpoint's address-generator constraints on a buffer for this ring.
     * @param iova Endpoint-visible address of page 0.
     * @param size Bytes available at `iova`.
     * @return nullptr when the buffer is usable, else why it is not.
     */
    const char* validate_buffer(uint64_t iova, uint64_t size) const
    {
        if (size < required_bytes()) {
            return "buffer is smaller than the ring requires";
        }
        if (iova % FPGA_PAGE_BYTES) {
            return "buffer address must be 128-byte aligned; BUF_START ignores the low 7 bits";
        }
        // BUF_BASE_HI supplies one fixed upper 32 bits for the whole ring, so the buffer may
        // sit anywhere in the 64-bit space but may not straddle a 4 GiB boundary.
        const uint64_t last = iova + required_bytes() - 1;
        if ((iova >> 32) != (last >> 32)) {
            return "buffer crosses a 4 GiB boundary, which the fixed-MSB address generator "
                   "cannot represent";
        }
        return nullptr;
    }
};

} // namespace hololink::module::pcie

#endif /* HOLOLINK_MODULE_PCIE_PCIE_RX_LAYOUT_HPP */
