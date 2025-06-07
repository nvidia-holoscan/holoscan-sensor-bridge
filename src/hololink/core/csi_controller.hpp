/**
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 *
 * See README.md for detailed information.
 */

#ifndef SRC_HOLOLINK_CSI_CONTROLLER
#define SRC_HOLOLINK_CSI_CONTROLLER

#include <cstdint>

#include "csi_formats.hpp"

namespace hololink::csi {

/**
 * Interface for objects that understand CSI image data; the sensor
 * driver calls the methods here to configure the object that interprets
 * the received data.
 */
class CsiConverter {
public:
    CsiConverter() = default;

    /**
     * Camera drivers use this call to train this object on
     * how to interpret the block of GPU memory that the sensor
     * delivers.
     * @param start_byte is the location within the received buffer
     *      of the first displayable pixel.  This is used to skip
     *      over framing data and other stuff transmitted by the device
     *      that isn't a part of the displayed image.
     * @param recieved_bytes_per_line is where to find the next horizontal
     *      line of image data, relative to the start of the previous line.
     *      The first line always starts at (start_byte).
     * @param pixel_width, pixel_height are the dimensions of the
     *      received image; and (except for RAW_8) aren't the same as bytes.
     * @param pixel_format is how the pixel data is encoded within
     *      the received buffer.
     * @param trailer is how many bytes are transmitted after visual data
     *      that are not actual visual data.
     *
     * For example, IMX274, RAW10, sends a video frame like this:
     *      - frame start
     *      - line start, 175 bytes of metadata, line end
     *      - 8 lines of (line start, width pixels of black, line end)
     *      - (height) lines of (line start, width pixels of image, line end)
     *      - frame end
     * Additionally, HSB hardware can adjust any of those based on
     *      optional functionality included in HSB FPGAs.  Examples include
     *      - insertion of other data before the frame-start message,
     *          e.g. metadata or cache alignment
     *      - removal frame start, line start, line end, and frame end messages
     *      - padding of line data
     * To accommodate this, the IMX274 driver "configure" method does this;
     *      configure accepts "converter" as a parameter:
     * @code
     *      start_byte = converter.receiver_start_byte()
     *      # How many bytes, per line, does the camera send?
     *      transmitted_line_bytes = converter.transmitted_line_bytes(pixel_format, pixel_width)
     *      # How many bytes, per line, appear in GPU memory?
     *      received_line_bytes = converter.received_line_bytes(transmitted_line_bytes)
     *      # We get a horizontal line of 175 bytes of metadata preceding the image data.
     *      start_byte += converter.received_line_bytes(175)
     *      # Skip 8 lines of optical black before the real image data starts
     *      start_byte += received_line_bytes * 8
     *      converter.configure(start_byte, received_line_bytes,
     *          pixel_width, pixel_height, pixel_format)
     * @endcode
     */
    virtual void configure(uint32_t start_byte, uint32_t recieved_bytes_per_line, uint32_t pixel_width, uint32_t pixel_height, PixelFormat pixel_format, uint32_t trailing_bytes = 0) = 0;

    /**
     * Returns the buffer location of the first byte of received pixel data.  This
     * may be affected by metadata chunks, framing bytes, and cache alignment.
     */
    virtual uint32_t receiver_start_byte() = 0;

    /**
     * Given an expected number of received bytes in a horizontal line,
     * how many bytes away is the first pixel of the next received line?
     * This is because framing or padding or cache alignment requirements
     * may leave extra space between each horizontal line.
     */
    virtual uint32_t received_line_bytes(uint32_t line_bytes) = 0;

    /**
     * Given an image width in pixels, how many actual bytes are used
     * to encode that horizontal line?  For example, RAW_8 is always
     * one byte per pixel--so this routine will return that same number
     * for the bytes; but RAW_10 uses 5 bytes to encode 4 pixels--e.g.
     * given a pixel format of RAW_10 and a width of 16, this routine returns 20.
     */
    virtual uint32_t transmitted_line_bytes(PixelFormat pixel_format, uint32_t pixel_width) = 0;
};

} // namespace hololink

#endif /* SRC_HOLOLINK_CSI_CONTROLLER */
