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

# See README.md for detailed information.

# Checks hololink bindings that need no board: I2C requests encoded into a frame-end
# sequencer (test_i2c.py), packetizer programs, and reset callbacks.

import gc
import weakref

import pytest

import hololink as hl

# Hololink::Event::SIF_0_FRAME_END, which enumeration gives sensor 0's channel.
SIF_0_FRAME_END = 16


def _data_channel(frame_end_event=SIF_0_FRAME_END):
    # Enough enumeration metadata for DataChannel's constructor; nothing contacts the board.
    metadata = hl.Metadata()
    for name, value in {
        "serial_number": "bindings-test",
        "peer_ip": "127.0.0.1",
        "control_port": 8192,
        "vp_mask": 0,
        "data_plane": 0,
        "sensor": 0,
        "sif_address": 0,
        "vp_address": 0,
        "hif_address": 0,
        "fpga_uuid": "00000000-0000-0000-0000-000000000000",
    }.items():
        metadata[name] = value
    if frame_end_event is not None:
        metadata["frame_end_event"] = frame_end_event
    return hl.DataChannel(metadata)


def test_i2c_request_encodes_into_the_frame_end_sequencer():
    # test_i2c.py encodes its pattern updates this way, as v2.7's
    # dual_imx274.Imx274Cam did; nothing reaches the board until the sequencer's
    # enable().
    channel = _data_channel()
    sequencer = channel.frame_end_sequencer()
    i2c = channel.hololink().get_i2c(hl.CAM_I2C_BUS)
    # A 16-bit register and an 8-bit value: one data word.
    write_indexes, read_indexes, status_index = i2c.encode_i2c_request(
        sequencer, 0x1A, bytes([0x30, 0x3D, 10]), 0
    )
    assert len(write_indexes) == 1
    assert read_indexes == []
    assert status_index > write_indexes[0]
    # With keywords and a bytearray: 5 bytes written
    # (two words), 2 read (one word).
    write_indexes, read_indexes, _ = i2c.encode_i2c_request(
        sequencer,
        peripheral_i2c_address=0x1A,
        write_bytes=bytearray([1, 2, 3, 4, 5]),
        read_byte_count=2,
    )
    assert (len(write_indexes), len(read_indexes)) == (2, 1)
    # And a list of int, also of ints that need converting.
    write_indexes, _, _ = i2c.encode_i2c_request(sequencer, 0x1A, [1, 2], 0)
    assert len(write_indexes) == 1
    write_indexes, _, _ = i2c.encode_i2c_request(
        sequencer, 0x1A, [hl.PixelFormat.RAW_10, True], 0
    )
    assert len(write_indexes) == 1


def test_i2c_request_rejects_a_10_bit_address():
    channel = _data_channel()
    i2c = channel.hololink().get_i2c(hl.CAM_I2C_BUS)
    with pytest.raises(RuntimeError, match="7-bit"):
        i2c.encode_i2c_request(channel.frame_end_sequencer(), 0x80, b"\x00", 0)


def test_frame_end_sequencer_needs_the_frame_end_event():
    with pytest.raises(RuntimeError, match="frame_end_sequencer"):
        _data_channel(frame_end_event=None).frame_end_sequencer()


def test_packetizer_programs():
    # A program's output size follows the schedule it loads into the HSB's packetizer, not
    # the pixel arithmetic: a line always completes a whole loop of the schedule. RAW_10's
    # loop takes 15 beats of CSI data (120 bytes, 96 pixels) and emits 16 (128 bytes: 3
    # pixels and 2 padding bits per 4 bytes), so 5 CSI bytes (4 pixels) still cost one
    # loop, and a 1280-pixel line (1600 bytes) 14 loops.
    raw_10 = hl.get_packetizer_program(hl.PixelFormat.RAW_10)
    assert type(raw_10).__name__ == "Csi10ToPacked10"
    assert raw_10.packetizes()
    assert raw_10.get_output_size(5) == 128
    assert raw_10.get_output_size(120) == 128
    assert raw_10.get_output_size(121) == 256
    assert raw_10.get_output_size(1600) == 1792  # 1280 pixels: 13.33 loops
    assert raw_10.get_output_size(2400) == 2560  # 1920 pixels: 20 loops exactly
    assert raw_10.get_output_size(4800) == 5120  # 3840 pixels: 40 loops exactly
    # ... and the HSB then pads each line to 64 bytes.
    assert raw_10.padded_output_size(1600) == 1792
    assert raw_10.padded_output_size(175) == 256  # an embedded-data line: 2 loops
    # RAW_12 is only reordered, 3 beats (24 bytes, 16 pixels) per loop.
    raw_12 = hl.get_packetizer_program(int(hl.PixelFormat.RAW_12))
    assert type(raw_12).__name__ == "Csi12ToPacked12"
    assert raw_12.packetizes()
    assert raw_12.get_output_size(6) == 24
    assert raw_12.get_output_size(1920) == 1920  # 1280 pixels: 80 loops exactly
    assert raw_12.get_output_size(1932) == 1944  # 1288 pixels: 80.5 loops
    assert raw_12.padded_output_size(1944) == 1984
    # RAW_8's program disables the packetizer, so it neither resizes nor pads.
    raw_8 = hl.get_packetizer_program(hl.PixelFormat.RAW_8)
    assert type(raw_8).__name__ == "NullPacketizerProgram"
    assert not raw_8.packetizes()
    assert raw_8.get_output_size(1280) == 1280
    with pytest.raises(ValueError, match="RAW_8, RAW_10 or RAW_12"):
        hl.get_packetizer_program(7)
    # The channel keeps the program and enables it when a receiver configures it.
    _data_channel().set_packetizer_program(raw_10)


def test_packetizer_programs_keep_sensor_modes():
    # Every IMX274 and VB1940 mode the packetizer can pack lands on the same padded line
    # size as the pixel arithmetic the programs replaced (3 RAW_10 pixels per 4 bytes, 2
    # RAW_12 pixels per 3, rounded up to 64): their CSI-2 lines are whole loops, or
    # (VB1940 2560 RAW_10: 26.67 loops, 3456 bytes) round to the same 64-byte boundary.
    # Only the stride those modes were measured with is pinned here; new modes need not
    # match the old arithmetic.
    raw_10 = hl.get_packetizer_program(hl.PixelFormat.RAW_10)
    raw_12 = hl.get_packetizer_program(hl.PixelFormat.RAW_12)
    for program, width, csi_bytes, padded in [
        (raw_10, 3840, 4800, 5120),  # IMX274 3840x2160 RAW_10
        (raw_10, 1920, 2400, 2560),  # IMX274 1920x1080 RAW_10, VB1940 1920x1080
        (raw_12, 3840, 5760, 5760),  # IMX274 3840x2160 RAW_12
        (raw_10, 2560, 3200, 3456),  # VB1940 2560x1984 RAW_10 (was 3416 before padding)
    ]:
        assert program.padded_output_size(csi_bytes) == padded, width


def test_packed_geometry():
    # The geometry of v2.7's CoE tests: the IMX274's 1080p RAW_10 frames as the
    # HSB's packetizer packs them. The embedded-data line takes 192 bytes (175
    # rounded up to 64) and 8 optical black lines precede the image; each line is
    # 1920 pixels packed 3 to 4 bytes.
    camera = hl.NativeImx274Sensor(_data_channel())
    camera.set_mode(1)  # IMX274_MODE_1920X1080_60FPS
    geometry = hl.resolve_packed_geometry(camera)
    assert geometry == {
        "start_byte": 192 + 8 * 2560,
        "bytes_per_line": 2560,
        "pixel_width": 1920,
        "pixel_height": 1080,
        "pixel_format": "raw_10",
        "trailing_bytes": 0,
    }
    # The same as FusaCoeCaptureOp's for packed frames, in builds that have it.
    if hasattr(hl, "resolve_geometry"):
        assert hl.resolve_geometry(camera, packetizer_enabled=True) == geometry


class _ResetCounter:
    def __init__(self):
        self.count = 0

    def reset(self):
        self.count += 1


def test_on_reset_takes_a_bound_method():
    device = _data_channel().hololink()
    device.on_reset(_ResetCounter().reset)
    with pytest.raises(TypeError, match="bound method"):
        device.on_reset(lambda: None)


def test_on_reset_keeps_no_camera_alive():
    # As in v2.7, the device holds the method's object weakly, so a camera that
    # registers its own method can still be collected.
    device = _data_channel().hololink()
    counter = _ResetCounter()
    device.on_reset(counter.reset)
    collected = weakref.ref(counter)
    del counter
    gc.collect()
    assert collected() is None


def test_on_reset_keeps_no_function_alive():
    # The method's function too: the device outlives Python's exit, and a function
    # would keep its module alive past it, which nanobind reports as leaks.
    class LocalCounter(_ResetCounter):
        def reset(self):
            self.count += 1

    device = _data_channel().hololink()
    counter = LocalCounter()
    device.on_reset(counter.reset)
    collected = weakref.ref(LocalCounter.reset)
    del counter, LocalCounter
    gc.collect()
    assert collected() is None
