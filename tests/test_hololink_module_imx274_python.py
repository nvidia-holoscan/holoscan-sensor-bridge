# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Mock-based tests for the IMX274 sensor driver. The driver is pure
# Python that talks through V1 module handles (HololinkInterfaceV1 /
# I2cInterfaceV1 / I2cLockV1); these tests stand in mocks for those
# handles and verify the driver produces the expected bus traffic for
# both expander_configuration=0 and =1.

import struct
from unittest.mock import MagicMock

import hololink_module.sensors.imx274 as imx274
import pytest
from hololink_module.sensors import csi

CAM_I2C_ADDRESS = 0b00011010
EXPANDER_I2C_ADDRESS = 0b01110000
OUTPUT_1_MASK = 0b0001
OUTPUT_2_MASK = 0b0010


class _FakeBus:
    """Stand-in for a per-(bus, address) I2cInterfaceV1 handle.

    Records every i2c_transaction call so tests can inspect the byte
    stream the driver emitted. Returns a fixed reply for read paths.
    """

    def __init__(self, read_reply=None):
        self.calls = []
        self._read_reply = read_reply or b"\x00"

    def i2c_transaction(self, address, write_bytes, read_byte_count, **kwargs):
        self.calls.append(
            {
                "address": address,
                "write": bytes(write_bytes),
                "read_byte_count": read_byte_count,
                "kwargs": kwargs,
            }
        )
        return self._read_reply[:read_byte_count]


class _FakeLock:
    def __init__(self):
        self.lock_count = 0
        self.unlock_count = 0

    def lock(self):
        self.lock_count += 1

    def unlock(self):
        self.unlock_count += 1


class _FakeHololink:
    """Stand-in for HololinkInterfaceV1 — hands out per-(bus,address) buses."""

    def __init__(self):
        self.buses = {}
        self.lock_handle = _FakeLock()
        self.clock_calls = []

    def get_i2c(self, bus, address):
        key = (bus, address)
        if key not in self.buses:
            self.buses[key] = _FakeBus()
        return self.buses[key]

    def i2c_lock(self):
        return self.lock_handle

    def setup_clock(self, profile):
        self.clock_calls.append(profile)


def _make_cam(expander_configuration=0):
    hl = _FakeHololink()
    # The oscillator's enable() is only reached from configure(); the
    # tests here exercise set_register / get_register / start / stop
    # paths that bypass it, so a non-null stand-in is enough.
    osc = MagicMock()
    cam = imx274.Imx274Cam(
        hl, osc, i2c_bus=1, expander_configuration=expander_configuration
    )
    return cam, hl


def test_constructor_acquires_separate_buses_for_camera_and_expander():
    cam, hl = _make_cam()
    assert (1, CAM_I2C_ADDRESS) in hl.buses
    assert (1, EXPANDER_I2C_ADDRESS) in hl.buses


def test_set_register_emits_be_register_then_value_byte():
    cam, hl = _make_cam()
    cam.set_register(0x3000, 0x42)

    expander = hl.buses[(1, EXPANDER_I2C_ADDRESS)]
    cam_bus = hl.buses[(1, CAM_I2C_ADDRESS)]

    # Expander selects camera 0 first, camera bus then receives the write.
    assert expander.calls[-1]["address"] == EXPANDER_I2C_ADDRESS
    assert expander.calls[-1]["write"] == bytes([OUTPUT_1_MASK])

    last = cam_bus.calls[-1]
    assert last["address"] == CAM_I2C_ADDRESS
    assert last["write"] == struct.pack(">H", 0x3000) + b"\x42"
    assert last["read_byte_count"] == 0


def test_get_register_uses_be_register_and_returns_first_byte():
    hl = _FakeHololink()
    hl.buses[(1, CAM_I2C_ADDRESS)] = _FakeBus(read_reply=b"\x5a")
    cam = imx274.Imx274Cam(hl, MagicMock(), i2c_bus=1, expander_configuration=0)

    value = cam.get_register(0x300A)
    assert value == 0x5A

    cam_bus = hl.buses[(1, CAM_I2C_ADDRESS)]
    last = cam_bus.calls[-1]
    assert last["address"] == CAM_I2C_ADDRESS
    assert last["write"] == struct.pack(">H", 0x300A)
    assert last["read_byte_count"] == 1


def test_expander_configuration_one_selects_output_2():
    cam, hl = _make_cam(expander_configuration=1)
    cam.set_register(0x3000, 0x01)
    expander = hl.buses[(1, EXPANDER_I2C_ADDRESS)]
    assert expander.calls[-1]["write"] == bytes([OUTPUT_2_MASK])


def test_each_register_op_acquires_and_releases_lock():
    cam, hl = _make_cam()
    cam.set_register(0x3000, 0x01)
    cam.set_register(0x3001, 0x02)
    cam.get_register(0x3002)

    # Three ops, three lock/unlock pairs.
    assert hl.lock_handle.lock_count == 3
    assert hl.lock_handle.unlock_count == 3


def test_lock_released_even_when_transaction_raises():
    hl = _FakeHololink()
    failing_bus = MagicMock()
    failing_bus.i2c_transaction.side_effect = RuntimeError("boom")
    hl.buses[(1, CAM_I2C_ADDRESS)] = failing_bus
    cam = imx274.Imx274Cam(hl, MagicMock(), i2c_bus=1, expander_configuration=0)

    with pytest.raises(RuntimeError):
        cam.set_register(0x3000, 0x01)
    assert hl.lock_handle.lock_count == 1
    assert hl.lock_handle.unlock_count == 1


def test_set_mode_records_geometry_and_pixel_format_for_4k():
    cam, _ = _make_cam()
    cam.set_mode(imx274.Imx274_Mode.IMX274_MODE_3840X2160_60FPS)
    assert cam._width == 3840
    assert cam._height == 2160
    assert cam.pixel_format() == csi.PixelFormat.RAW_10
    assert cam.bayer_format() == csi.BayerFormat.RGGB


def test_set_mode_records_geometry_and_pixel_format_for_12bit():
    cam, _ = _make_cam()
    cam.set_mode(imx274.Imx274_Mode.IMX274_MODE_3840X2160_60FPS_12BITS)
    assert cam._width == 3840
    assert cam._height == 2160
    assert cam.pixel_format() == csi.PixelFormat.RAW_12


def test_setup_clock_requires_profile_and_forwards_to_hololink():
    cam, hl = _make_cam()
    with pytest.raises(ValueError):
        cam.setup_clock(None)

    cam.setup_clock({"steps": [(0x10, 0x20)]})
    assert hl.clock_calls == [{"steps": [(0x10, 0x20)]}]


def test_test_pattern_disable_writes_zero_to_each_register():
    cam, hl = _make_cam()
    cam.test_pattern(None)
    cam_bus = hl.buses[(1, CAM_I2C_ADDRESS)]
    # Four register writes for disable (0x303C, 0x377F, 0x3781, 0x370B).
    writes = [c["write"] for c in cam_bus.calls]
    expected = [
        struct.pack(">H", 0x303C) + b"\x00",
        struct.pack(">H", 0x377F) + b"\x00",
        struct.pack(">H", 0x3781) + b"\x00",
        struct.pack(">H", 0x370B) + b"\x00",
    ]
    assert writes == expected


def test_subpackage_exposes_top_level_aliases():
    assert imx274.Imx274Cam is imx274.imx274_cam.Imx274Cam
    assert imx274.LII2CExpander is imx274.li_i2c_expander.LII2CExpander
    assert imx274.Imx274_Mode is imx274.imx274_mode.Imx274_Mode
