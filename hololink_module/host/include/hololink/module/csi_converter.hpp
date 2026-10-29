/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CSI_CONVERTER_HPP
#define HOLOLINK_MODULE_CSI_CONVERTER_HPP

#include <cstdint>

namespace hololink::module::csi {

/*
 * The enums below are adapter-owned (no dependency on the legacy
 * hololink::csi enums). The integer values are deliberately identical to
 * the legacy hololink::csi::PixelFormat / BayerFormat so an application
 * bridging to a legacy operator (CsiToBayerOp, BayerDemosaicOp,
 * ImageProcessorOp) can static_cast<int>(format) across the boundary
 * without a conversion table. The Python sibling
 * hololink_module.sensors.csi carries the same values.
 */

/**
 * @brief How the pixels of a CSI-2 raw image are encoded.
 *
 * The name gives the number of bits of data per pixel. This describes the
 * pixel values themselves; how many bytes a line of pixels occupies on the
 * wire depends also on how the converter packs them, which is reported
 * by CsiConverterV1::transmitted_line_bytes().
 */
enum class PixelFormat : uint32_t {
    RAW_8 = 0, ///< 8 bits per pixel.
    RAW_10 = 1, ///< 10 bits per pixel.
    RAW_12 = 2, ///< 12 bits per pixel.
};

/**
 * @brief The arrangement of color filters over a raw image sensor, which
 * determines the color of each pixel.
 *
 * A sensor's pixels are covered with a repeating 2x2 pattern of red, green
 * and blue filters. The name lists the colors of the pattern's four
 * pixels in reading order: the first two letters are the first two
 * pixels of the top line, and the last two are the first two pixels of
 * the line below. So `BGGR` is blue then green on the top line, green then
 * red on the next. The pattern is read from the top-left pixel of the
 * image as received, so it changes if the image is cropped by an odd
 * number of pixels or lines.
 *
 * The values are the same as those of NPP's `NppiBayerGridPosition`.
 */
enum class BayerFormat : uint32_t {
    BGGR = 0, ///< Blue, green / green, red.
    RGGB = 1, ///< Red, green / green, blue.
    GBRG = 2, ///< Green, blue / red, green.
    GRBG = 3, ///< Green, red / blue, green.
};

/**
 * @brief Tells the component that interprets received image data how that
 * data is laid out in the receive buffer.
 *
 * The device delivers each frame into a buffer as CSI-2 data: possibly
 * some leading bytes (metadata, framing), then the image lines, each
 * padded to whatever alignment the receiving hardware or software
 * requires, then possibly some trailing bytes. The component that
 * converts that buffer into an image (a converter operator, or a
 * capture operator that does so itself) implements this interface. A
 * sensor driver uses it to find out how the receiver arranges the data,
 * and then tells the converter where within that arrangement the image
 * is.
 *
 * Typical use, by a sensor driver:
 *  -# Ask for the receiver's geometry with receiver_start_byte(),
 *     transmitted_line_bytes() and received_line_bytes().
 *  -# Add the sensor's own non-image content to the start offset (for
 *     example metadata lines and optical-black lines that precede the
 *     image).
 *  -# Call configure() with the resulting layout.
 *
 * The geometry depends on the converter in use: for example, how many
 * bytes a RAW_10 line occupies, and the alignment of received lines,
 * differ between converters. Applications and drivers must therefore ask
 * the converter and not assume.
 *
 * This is not a service obtained from a Module; the application hands the
 * converter to the sensor driver as a `std::shared_ptr`.
 */
class CsiConverterV1 {
public:
    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    static constexpr const char* release_version = "2.7.0";
    virtual ~CsiConverterV1() = default;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Tell the converter where the image is in the receive buffer
     * and how it is encoded.
     *
     * Call this before frames are processed, once the geometry above is
     * known.
     *
     * @param start_byte Offset, from the start of the receive buffer, of
     *      the first byte of the first image line; everything before it
     *      (leading framing, metadata, and any sensor lines that are not
     *      image) is skipped.
     * @param received_bytes_per_line Distance in bytes from the start of
     *      one image line to the start of the next, including padding
     *      (see received_line_bytes()).
     * @param pixel_width Image width, in pixels.
     * @param pixel_height Image height, in lines.
     * @param pixel_format How pixels are encoded in the received data.
     * @param trailing_bytes Bytes in the buffer, after the last image
     *      line, that are not image data.
     */
    virtual void configure(uint32_t start_byte, uint32_t received_bytes_per_line,
        uint32_t pixel_width, uint32_t pixel_height, PixelFormat pixel_format,
        uint32_t trailing_bytes = 0)
        = 0;

    /**
     * @brief Offset in the receive buffer at which the receiver puts the
     * first byte of the data the sensor sent.
     *
     * Accounts for anything the receiver places before the sensor's data.
     * Add the sensor's own leading non-image bytes to this to obtain the
     * `start_byte` for configure().
     *
     * @return The offset, in bytes.
     */
    virtual uint32_t receiver_start_byte() = 0;

    /**
     * @brief The distance, in the receive buffer, from the start of one
     * line to the start of the next.
     *
     * The receiver pads each line it stores to a multiple of an
     * alignment it requires, so this is `line_bytes` rounded up
     * accordingly.
     *
     * @param line_bytes Bytes the sensor transmits for a line, as given
     *      by transmitted_line_bytes() (or the size of any other
     *      sensor line, such as metadata).
     * @return The spacing between received lines, in bytes.
     */
    virtual uint32_t received_line_bytes(uint32_t line_bytes) = 0;

    /**
     * @brief The number of bytes the sensor transmits for one image line.
     *
     * Depends on how many bits the pixel format uses and on how the
     * converter packs pixels (for example RAW_8 uses one byte per
     * pixel, while RAW_10 and RAW_12 pack several pixels across a
     * smaller number of bytes). The count excludes any padding the
     * receiver adds; see received_line_bytes().
     *
     * @param pixel_format How pixels are encoded.
     * @param pixel_width Image width, in pixels.
     * @return Bytes per transmitted line.
     * @throws std::runtime_error if the converter does not support
     *      `pixel_format`, or if it needs to be told which format to
     *      expect before it can answer and has not been.
     */
    virtual uint32_t transmitted_line_bytes(PixelFormat pixel_format, uint32_t pixel_width) = 0;
};

} // namespace hololink::module::csi

#endif // HOLOLINK_MODULE_CSI_CONVERTER_HPP
