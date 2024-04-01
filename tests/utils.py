# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# See README.md for detailed information.

import logging

import numpy as np

import hololink as hololink_module


def encode_raw_8_bayer_image(bayer_image):
    return bayer_image.astype(np.uint8)


def encode_raw_12_bayer_image(bayer_image):
    """
    Given a uint16 bayer image, encode it
    to a stream of bytes where the format
    matches the RAW12 format published in IMX274 spec;
    given the first two 12-bit color values (c0 and c1),
    the output is
        c0 bits 4..11,
        c1 bits 4..11,
        c0 bits 0..3 | ((c1 bits 0..3) << 4)
    for each uint16 in bayer_image.
    """
    bayer_height, bayer_width = bayer_image.shape
    ri = bayer_image.ravel()
    upper_byte = (ri >> 4).astype(np.uint8)
    lower_byte = (ri & 0xF).astype(np.uint8)
    combined_lower_bytes = lower_byte[::2] | (lower_byte[1::2] << 4)
    raw_12 = np.stack(
        [
            upper_byte[::2],
            upper_byte[1::2],
            combined_lower_bytes,
        ],
        axis=1,
    )
    # we're now 3 bytes per pixel wide instead of 2 half-words
    raw_12 = raw_12.reshape(bayer_height, bayer_width * 3 // 2)
    return raw_12


def encode_raw_10_bayer_image(bayer_image):
    """
    Given a uint16 bayer image, encode it
    to a stream of bytes where the format
    matches the RAW10 format published in IMX274 spec;
    given the first four 10-bit color values (c0, c1, c2, and c3),
    the output is
        c0 bits 2..9,
        c1 bits 2..9,
        c2 bits 2..9,
        c3 bits 2..9,
        (c0 bits 0..1 | ((c1 bits 0..1) << 2)
            | (c2 bits 0..1) << 4 | ((c3 bits 0..1) << 6))
    for each uint16 in bayer_image.
    """
    bayer_height, bayer_width = bayer_image.shape
    ri = bayer_image.ravel()
    upper_byte = (ri >> 2).astype(np.uint8)
    lower_byte = (ri & 0x3).astype(np.uint8)
    combined_lower_bytes = (
        lower_byte[::4]
        | (lower_byte[1::4] << 2)
        | (lower_byte[2::4] << 4)
        | (lower_byte[3::4] << 6)
    )
    raw_10 = np.stack(
        [
            upper_byte[::4],
            upper_byte[1::4],
            upper_byte[2::4],
            upper_byte[3::4],
            combined_lower_bytes,
        ],
        axis=1,
    )
    # we're now 5 bytes per four pixels wide
    raw_10 = raw_10.reshape(bayer_height, bayer_width * 5 // 4)
    return raw_10


def encode_8_bit_image(image):
    return image


def encode_12_bit_image(image):
    return (image >> 4).astype(np.uint8)


def encode_10_bit_image(image):
    return (image >> 2).astype(np.uint8)


def make_image(
    bayer_height,
    bayer_width,
    bayer_format: hololink_module.sensors.csi.BayerFormat,
    pixel_format: hololink_module.sensors.csi.PixelFormat,
):
    """Make a demo video frame."""
    logging.info(
        "Generating a demo image bayer_height=%s bayer_width=%s bayer_format=%s pixel_format=%s"  # noqa: E501
        % (bayer_height, bayer_width, bayer_format, pixel_format)
    )
    if pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_8:
        dtype = np.uint8
        limit = 255
        bayer_encoder = encode_raw_8_bayer_image
        image_encoder = encode_8_bit_image
    elif pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_12:
        dtype = np.uint16
        limit = 4095
        bayer_encoder = encode_raw_12_bayer_image
        image_encoder = encode_12_bit_image
    elif pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_10:
        dtype = np.uint16
        limit = 1023
        bayer_encoder = encode_raw_10_bayer_image
        image_encoder = encode_10_bit_image
    # the colors vary from min to max
    width = bayer_width // 2
    height = bayer_height // 2
    r = np.linspace(0, limit, num=width, dtype=dtype)
    g = np.linspace(0, limit, num=height, dtype=dtype)
    b = np.linspace(limit, 0, num=width, dtype=dtype)
    a = np.full((width,), limit, dtype=dtype)
    # make square arrays, these are one element
    # per pixel for the corresponding color.
    sr = np.tile(r, height).reshape(height, width)
    sg = np.tile(g, width).reshape(width, height).transpose()
    sb = np.tile(b, height).reshape(height, width)
    sa = np.tile(a, height).reshape(height, width)
    # merge into an RGBA frame.
    elements_per_pixel = 4
    image = np.stack([sr, sg, sb, sa], axis=2).reshape(
        height, width, elements_per_pixel
    )
    # Now make the bayer frame.
    if bayer_format == hololink_module.sensors.csi.BayerFormat.RGGB:
        # upper_line is red0, green0, red1, green1, ...
        a, b = sr.ravel(), sg.ravel()
        c = np.empty((a.size + b.size,), dtype=dtype)
        c[0::2] = a
        c[1::2] = b
        upper_line = c.reshape(height, bayer_width)
        # lower_line is green0, blue0, green1, blue1, ...
        a, b = sg.ravel(), sb.ravel()
        c = np.empty((a.size + b.size,), dtype=dtype)
        c[0::2] = a
        c[1::2] = b
        lower_line = c.reshape(height, bayer_width)
        bayer_image = np.stack([upper_line, lower_line], axis=1).reshape(
            bayer_height, bayer_width
        )
    elif bayer_format == hololink_module.sensors.csi.BayerFormat.GBRG:
        # upper_line is green0, blue0, green1, blue1, ...
        a, b = sg.ravel(), sb.ravel()
        c = np.empty((a.size + b.size,), dtype=dtype)
        c[0::2] = a
        c[1::2] = b
        upper_line = c.reshape(height, bayer_width)
        # lower_line is red0, green0, red1, green1, ...
        a, b = sr.ravel(), sg.ravel()
        c = np.empty((a.size + b.size,), dtype=dtype)
        c[0::2] = a
        c[1::2] = b
        lower_line = c.reshape(height, bayer_width)
        bayer_image = np.stack([upper_line, lower_line], axis=1).reshape(
            bayer_height, bayer_width
        )
    else:
        assert False and 'Unexpected image format "%s".' % (bayer_format,)
    #
    image = image_encoder(image)
    bayer_image = bayer_encoder(bayer_image)
    return image, bayer_image
