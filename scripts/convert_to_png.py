#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# /// script
# dependencies = [
#   "numpy",
#   "Pillow",
# ]
# ///
"""Convert fusa_coe_frame_saver .raw + .txt frame captures to viewable PNGs.

Each capture is a pair of files (see examples/fusa_coe_frame_saver.cpp):
  frame_NNNNNNNN.raw -- pixel_height rows of bytes_per_line bytes each
  frame_NNNNNNNN.txt -- sidecar: pixel_width/height, bytes_per_line,
                         pixel_format, bayer_format, packetizer_enabled

Unpacking follows the exact layouts used by the capture pipeline itself
(hololink::operators::fusa_coe_capture::FusaCoeCaptureCore::transmitted_line_bytes/
received_line_bytes, and hololink::PacketizerProgram's documented bit layouts):
  - RAW_10, packetizer enabled: 3x 10-bit pixels + 2 padding bits per 4-byte
    little-endian word, {2'b0, p3[9:0], p2[9:0], p1[9:0]} (Csi10ToPacked10),
    lines padded to a 64-byte stride.
  - RAW_10, packetizer disabled: standard MIPI CSI-2 packing, 4 pixels per 5
    bytes (LSBs packed into the 5th byte), lines padded to an 8-byte stride.
  - RAW_12, packetizer enabled: 2x 12-bit pixels per 3 bytes,
    {p2[11:0], p1[11:0]} (Csi12ToPacked12), lines padded to a 64-byte stride.
  - RAW_8: one byte per pixel, no packing.

Demosaicing is a simple 2x2 box filter (one RGB sample per Bayer tile, not a
full-resolution interpolation) -- enough to visually confirm captured content
without extra dependencies beyond numpy/Pillow. Output PNGs are therefore
half the linear resolution of the sensor capture.

Usage:
  uv run convert_to_png.py [input_dir] [--output-dir DIR]

  (or: pip install numpy Pillow && python3 convert_to_png.py ...)

  input_dir   Directory to search for frame_*.raw/.txt pairs, recursively.
              Default: <repo root>/captured_frames.
  --output-dir  Where to write PNGs, mirroring input_dir's subdirectory
                structure. Default: alongside each .raw file.
"""

import argparse
import pathlib
import sys

import numpy as np
from PIL import Image


def parse_sidecar(txt_path):
    fields = {}
    for line in txt_path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def _rows(data, pixel_height, bytes_per_line):
    buf = np.frombuffer(data, dtype=np.uint8)
    return buf[: bytes_per_line * pixel_height].reshape(pixel_height, bytes_per_line)


def unpack_raw10_packetized(data, pixel_width, pixel_height, bytes_per_line):
    transmitted_line_bytes = ((pixel_width + 2) // 3) * 4
    groups_per_line = transmitted_line_bytes // 4
    rows = _rows(data, pixel_height, bytes_per_line)
    packed = rows[:, :transmitted_line_bytes].reshape(pixel_height, groups_per_line, 4)
    words = packed.astype(np.uint32)
    word = (
        words[..., 0]
        | (words[..., 1] << 8)
        | (words[..., 2] << 16)
        | (words[..., 3] << 24)
    )
    p1 = word & 0x3FF
    p2 = (word >> 10) & 0x3FF
    p3 = (word >> 20) & 0x3FF
    pixels = np.stack([p1, p2, p3], axis=-1).reshape(pixel_height, groups_per_line * 3)
    return pixels[:, :pixel_width].astype(np.uint16)


def unpack_raw10_unpacketized(data, pixel_width, pixel_height, bytes_per_line):
    groups_per_line = (pixel_width + 3) // 4
    transmitted_line_bytes = groups_per_line * 5
    rows = _rows(data, pixel_height, bytes_per_line)
    packed = (
        rows[:, :transmitted_line_bytes]
        .reshape(pixel_height, groups_per_line, 5)
        .astype(np.uint16)
    )
    p0 = (packed[..., 0] << 2) | (packed[..., 4] & 0x03)
    p1 = (packed[..., 1] << 2) | ((packed[..., 4] >> 2) & 0x03)
    p2 = (packed[..., 2] << 2) | ((packed[..., 4] >> 4) & 0x03)
    p3 = (packed[..., 3] << 2) | ((packed[..., 4] >> 6) & 0x03)
    pixels = np.stack([p0, p1, p2, p3], axis=-1).reshape(
        pixel_height, groups_per_line * 4
    )
    return pixels[:, :pixel_width]


def unpack_raw12_packetized(data, pixel_width, pixel_height, bytes_per_line):
    transmitted_line_bytes = ((pixel_width + 1) // 2) * 3
    groups_per_line = transmitted_line_bytes // 3
    rows = _rows(data, pixel_height, bytes_per_line)
    packed = (
        rows[:, :transmitted_line_bytes]
        .reshape(pixel_height, groups_per_line, 3)
        .astype(np.uint32)
    )
    word = packed[..., 0] | (packed[..., 1] << 8) | (packed[..., 2] << 16)
    p1 = word & 0xFFF
    p2 = (word >> 12) & 0xFFF
    pixels = np.stack([p1, p2], axis=-1).reshape(pixel_height, groups_per_line * 2)
    return pixels[:, :pixel_width]


def unpack_raw8(data, pixel_width, pixel_height, bytes_per_line):
    rows = _rows(data, pixel_height, bytes_per_line)
    return rows[:, :pixel_width].astype(np.uint16)


def unpack(data, fields):
    pixel_width = int(fields["pixel_width"])
    pixel_height = int(fields["pixel_height"])
    bytes_per_line = int(fields["bytes_per_line"])
    pixel_format = fields["pixel_format"]
    packetizer_enabled = fields.get("packetizer_enabled", "0") == "1"

    if pixel_format == "RAW_8":
        return unpack_raw8(data, pixel_width, pixel_height, bytes_per_line), 8
    if pixel_format == "RAW_10":
        if packetizer_enabled:
            return (
                unpack_raw10_packetized(
                    data, pixel_width, pixel_height, bytes_per_line
                ),
                10,
            )
        return (
            unpack_raw10_unpacketized(data, pixel_width, pixel_height, bytes_per_line),
            10,
        )
    if pixel_format == "RAW_12":
        if packetizer_enabled:
            return (
                unpack_raw12_packetized(
                    data, pixel_width, pixel_height, bytes_per_line
                ),
                12,
            )
        raise NotImplementedError("Non-packetized RAW_12 is not implemented")
    raise ValueError(f"Unsupported pixel_format {pixel_format!r}")


# (row, col) offset of the Red, two Greens, and Blue samples within a 2x2 tile,
# for each of the four Bayer/CFA arrangements.
BAYER_OFFSETS = {
    "RGGB": {"R": (0, 0), "G1": (0, 1), "G2": (1, 0), "B": (1, 1)},
    "BGGR": {"R": (1, 1), "G1": (0, 1), "G2": (1, 0), "B": (0, 0)},
    "GRBG": {"R": (0, 1), "G1": (0, 0), "G2": (1, 1), "B": (1, 0)},
    "GBRG": {"R": (1, 0), "G1": (0, 0), "G2": (1, 1), "B": (0, 1)},
}


def debayer_box(bayer, bayer_format):
    """One RGB sample per 2x2 Bayer tile (box filter, not interpolated) --
    output is half the linear resolution of the input."""
    offsets = BAYER_OFFSETS[bayer_format]
    h, w = bayer.shape
    h2, w2 = h // 2, w // 2
    tiles = bayer[: h2 * 2, : w2 * 2].reshape(h2, 2, w2, 2)

    def sample(offset):
        r, c = offset
        return tiles[:, r, :, c]

    r = sample(offsets["R"]).astype(np.float32)
    g = (
        sample(offsets["G1"]).astype(np.float32)
        + sample(offsets["G2"]).astype(np.float32)
    ) / 2
    b = sample(offsets["B"]).astype(np.float32)
    return np.stack([r, g, b], axis=-1)


def to_8bit(channel, bit_depth):
    scale = 255.0 / ((1 << bit_depth) - 1)
    return np.clip(channel * scale, 0, 255).astype(np.uint8)


def convert_one(raw_path, txt_path, out_path):
    fields = parse_sidecar(txt_path)
    data = raw_path.read_bytes()
    bayer, bit_depth = unpack(data, fields)
    bayer_format = fields.get("bayer_format", "GBRG")
    rgb = debayer_box(bayer, bayer_format)
    rgb8 = to_8bit(rgb, bit_depth)
    Image.fromarray(rgb8, mode="RGB").save(out_path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=str(pathlib.Path(__file__).resolve().parent.parent / "captured_frames"),
        help="Directory to search for frame_*.raw/.txt pairs (recursive). "
        "Default: <repo root>/captured_frames.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write PNGs, mirroring input_dir's subdirectory structure. "
        "Default: alongside each .raw file.",
    )
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    raw_files = sorted(input_dir.rglob("*.raw"))
    if not raw_files:
        print(f"No .raw files found under {input_dir}", file=sys.stderr)
        return 1

    for raw_path in raw_files:
        txt_path = raw_path.with_suffix(".txt")
        if not txt_path.exists():
            print(f"skip {raw_path}: no matching .txt sidecar", file=sys.stderr)
            continue
        if args.output_dir:
            rel = raw_path.relative_to(input_dir)
            out_path = pathlib.Path(args.output_dir) / rel.with_suffix(".png")
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = raw_path.with_suffix(".png")
        convert_one(raw_path, txt_path, out_path)
        print(f"{raw_path} -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
