/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CSI_CONVERTER_HPP
#define HOLOLINK_MODULE_CSI_CONVERTER_HPP

#include <cstdint>

namespace hololink::module::csi {

/* CSI pixel- and Bayer-format enums for module-side sensor drivers.
 *
 * Adapter-owned (no dependency on the legacy hololink::csi enums). The
 * integer values are deliberately identical to the legacy
 * hololink::csi::PixelFormat / BayerFormat so an application bridging to a
 * legacy operator (CsiToBayerOp, BayerDemosaicOp, ImageProcessorOp) can
 * static_cast<int>(format) across the boundary without a conversion table.
 * The Python sibling hololink_module.sensors.csi carries the same values. */
enum class PixelFormat : uint32_t {
    RAW_8 = 0,
    RAW_10 = 1,
    RAW_12 = 2,
};

/* Matches NPP's NppiBayerGridPosition ordering (same values as the legacy
 * hololink::csi::BayerFormat). */
enum class BayerFormat : uint32_t {
    BGGR = 0,
    RGGB = 1,
    GBRG = 2,
    GRBG = 3,
};

/* Interface for an object that interprets received CSI image data. A sensor
 * driver calls these methods to train the converter on how to read the block
 * of memory the device delivers. Adapter-owned (a plain abstract base passed
 * by shared_ptr, not a locator service); it mirrors the legacy
 * hololink::csi::CsiConverter contract using the module PixelFormat, so the
 * sensors name no legacy type. The implementation (today the legacy
 * CsiToBayerOp operator, bridged at the application layer) supplies the
 * receiver-side geometry the helpers below report. */
class CsiConverterV1 {
public:
    virtual ~CsiConverterV1() = default;

    /* Train the converter on how to interpret the received buffer.
     *   start_byte                - offset of the first displayable pixel
     *                               (skips framing/metadata).
     *   received_bytes_per_line   - stride between received lines.
     *   pixel_width, pixel_height - received image dimensions.
     *   pixel_format              - how pixels are encoded in the buffer.
     *   trailing_bytes            - non-image bytes after the visual data. */
    virtual void configure(uint32_t start_byte, uint32_t received_bytes_per_line,
        uint32_t pixel_width, uint32_t pixel_height, PixelFormat pixel_format,
        uint32_t trailing_bytes = 0)
        = 0;

    /* Buffer offset of the first received pixel byte (after any metadata,
     * framing, or cache alignment the receiver inserts). */
    virtual uint32_t receiver_start_byte() = 0;

    /* Given the bytes in a transmitted horizontal line, the distance to the
     * first byte of the next received line (accounts for receiver padding /
     * alignment). */
    virtual uint32_t received_line_bytes(uint32_t line_bytes) = 0;

    /* Bytes used to encode one horizontal line of pixel_width pixels in the
     * given format (e.g. RAW_10 packs 4 pixels into 5 bytes). */
    virtual uint32_t transmitted_line_bytes(PixelFormat pixel_format, uint32_t pixel_width) = 0;
};

} // namespace hololink::module::csi

#endif // HOLOLINK_MODULE_CSI_CONVERTER_HPP
