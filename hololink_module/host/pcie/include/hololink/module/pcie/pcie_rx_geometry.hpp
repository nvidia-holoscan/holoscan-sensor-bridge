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

#ifndef HOLOLINK_MODULE_PCIE_PCIE_RX_GEOMETRY_HPP
#define HOLOLINK_MODULE_PCIE_PCIE_RX_GEOMETRY_HPP

/**
 * @file
 * @brief Frame geometry and receive-ring slot layout for the module-native PCIe receiver.
 *
 * Header-only port of the frame-geometry and ring-sizing arithmetic of the reference
 * bring-up receiver (ported from
 * hololink_pcie_bringup/lib/hololink_pcie/pcie_rx_player.py and the hsdk5_pcie_demo
 * Raw10Geometry), plus the ring configuration (RxRingConfig) and the ring / buffer limit
 * checks the receive ring used to carry. The numbers here place DMA writes: an error is
 * silent frame corruption on hardware, so the arithmetic is kept byte-identical to the
 * legacy path and covered by the parity cases in tests/hololink_module_pcie_receiver_test.cpp.
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
/// Per-frame metadata block the endpoint appends after the payload window.
inline constexpr uint64_t RX_METADATA_BYTES = 128;
/// MemRd fetches cannot cross a 4 KiB boundary, so slots start page-aligned to keep a
/// captured ring replayable through the TX path.
inline constexpr uint64_t RX_SLOT_ALIGNMENT = 4096;
/// BUF_PTR_RANGE is 12 bits wide.
inline constexpr uint32_t RX_MAX_BUFFERS = 4096;
/// DP_PKT_LEN is 8 bits wide.
inline constexpr uint32_t RX_MAX_CHUNK_PAGES = 255;

inline constexpr uint64_t align_up(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief Frame geometry of a packed RAW10 CSI frame (4 pixels per 5 bytes), and the DMA
 * sizes derived from it. RAW10 is the only format the Sensor RX path delivers.
 */
struct RxFrameGeometry {
    uint32_t width { 0 };
    uint32_t height { 0 };
    uint32_t stride_pixels { 0 };
    uint32_t crop_x { 0 };
    /// Lines are padded to this; 1920 RAW10 pixels are exactly 2400 bytes, so the default
    /// of 8 is a no-op there. Use 64 only when FPGA packetizer padding is enabled.
    uint32_t line_alignment { 8 };
    /// Leading prefix the sensor emits before the image. Measured at 640 bytes on the
    /// IMX258 bring-up path with PDAF auxiliary output disabled.
    uint32_t raw_start_byte { 0 };

    uint64_t active_bytes_per_line() const
    {
        return static_cast<uint64_t>(width) * 5 / 4;
    }

    uint64_t valid_bytes_per_line() const
    {
        return static_cast<uint64_t>(stride_pixels) * 5 / 4;
    }

    uint64_t dma_line_bytes() const
    {
        return align_up(valid_bytes_per_line(), line_alignment);
    }

    uint64_t frame_dma_bytes() const
    {
        return dma_line_bytes() * height + raw_start_byte;
    }

    /// Where the endpoint writes the metadata block within a slot sized exactly to the frame.
    uint64_t metadata_offset() const
    {
        return align_up(frame_dma_bytes(), FPGA_PAGE_BYTES);
    }

    /// @return nullptr when the geometry is usable, else why it is not.
    const char* validate() const
    {
        if (width == 0 || height == 0 || stride_pixels == 0) {
            return "width, height and stride_pixels must be positive";
        }
        if (crop_x + width > stride_pixels) {
            return "crop window must fit inside stride_pixels";
        }
        if ((width % 4) || (stride_pixels % 4) || (crop_x % 2)) {
            return "RAW10 width and stride_pixels must be multiples of 4, crop_x even";
        }
        if (line_alignment == 0 || (line_alignment & (line_alignment - 1))) {
            return "line_alignment must be a positive power of two";
        }
        if (raw_start_byte % 8) {
            return "RAW10 start byte must be 8-byte aligned";
        }
        return nullptr;
    }
};

/**
 * @brief Slot layout of a receive ring: where the metadata block lands, how far apart slots
 * are, and what BUF_INC the endpoint needs.
 *
 * Same arithmetic as the reference ring constructor: the payload window is padded to a 128-byte
 * page for the metadata block, and each slot is padded to 4 KiB so a captured ring stays
 * replayable through MemRd. A single-buffer ring rewrites the same slot, so BUF_INC is 0.
 */
struct RxRingLayout {
    uint64_t window_bytes { 0 }; ///< payload window per slot
    uint64_t metadata_offset { 0 }; ///< offset of the metadata block within a slot
    uint64_t slot_stride { 0 }; ///< distance between consecutive slots
    uint32_t buf_inc_pages { 0 }; ///< PCIE_DP_BUF_INC, in 128-byte pages
    uint32_t num_buffers { 1 };

    /// Total device memory the ring needs.
    uint64_t required_bytes() const { return slot_stride * num_buffers; }

    /// Byte offset of `slot`'s payload from the start of the ring.
    uint64_t slot_offset(uint32_t slot) const { return slot * slot_stride; }

    /**
     * @param geometry Frame geometry fixing the minimum window.
     * @param num_buffers Slots in the ring.
     * @param window_bytes Payload window per slot; 0 means exactly frame_dma_bytes().
     */
    static RxRingLayout compute(const RxFrameGeometry& geometry, uint32_t num_buffers,
        uint64_t window_bytes = 0)
    {
        RxRingLayout layout;
        layout.num_buffers = num_buffers;
        layout.window_bytes = window_bytes ? window_bytes : geometry.frame_dma_bytes();
        layout.metadata_offset = align_up(layout.window_bytes, FPGA_PAGE_BYTES);
        layout.slot_stride
            = align_up(layout.metadata_offset + RX_METADATA_BYTES, RX_SLOT_ALIGNMENT);
        layout.buf_inc_pages = num_buffers > 1
            ? static_cast<uint32_t>(layout.slot_stride / FPGA_PAGE_BYTES)
            : 0;
        return layout;
    }

    /**
     * @brief The endpoint's limits on a ring of this layout.
     * @param geometry The geometry the layout was computed from (its window is the minimum).
     * @return nullptr when the ring is usable, else why it is not.
     */
    const char* validate(const RxFrameGeometry& geometry) const
    {
        if (const char* error = geometry.validate()) {
            return error;
        }
        if (num_buffers < 1) {
            return "num_buffers must be at least 1";
        }
        if (num_buffers > RX_MAX_BUFFERS) {
            return "num_buffers must fit in the 12-bit endpoint buffer pointer";
        }
        if (window_bytes < geometry.frame_dma_bytes()) {
            return "window_bytes must be at least the frame DMA byte count";
        }
        if (window_bytes > 0xFFFFFFFFull) {
            return "window_bytes must fit in the 32-bit BUF_LEN register";
        }
        if (slot_stride % FPGA_PAGE_BYTES) {
            return "slot stride must be 128-byte aligned for the endpoint buffer increment";
        }
        return nullptr;
    }

    /**
     * @brief The endpoint's address-generator constraints on a buffer for this ring.
     * @param iova Endpoint-visible address of slot 0.
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

/**
 * @brief Ring configuration that is not geometry.
 *
 * What the receiver needs to size and map its ring. The data TLP size (DP_PKT_LEN,
 * `chunk_pages`) is not here: it is a per-endpoint word the channel programs, passed to
 * PcieDataChannelInterfaceV1::attach_receiver. The PCI requester id comes from the
 * receiver's device identity.
 */
struct RxRingConfig {
    uint32_t sensor { 0 }; ///< PCIE_DP block and MSI-X vector
    uint32_t num_buffers { 1 };
    /// Payload window per slot; 0 means exactly frame_dma_bytes.
    uint64_t window_bytes { 0 };
    /// Set MD_DISABLE so the endpoint writes no metadata block. PcieReceiver::start() rejects
    /// it (the receiver finds frames through those blocks); for attachments that do not.
    bool disable_metadata { false };
};

} // namespace hololink::module::pcie

#endif /* HOLOLINK_MODULE_PCIE_PCIE_RX_GEOMETRY_HPP */
