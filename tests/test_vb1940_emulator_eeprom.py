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

"""
Smoke tests for the Vb1940Emulator Python binding additions
(set_eeprom_data + Vb1940Emulator.EEPROM_REGION_BYTES).

These tests verify only that the pybind11 layer accepts the various Python
buffer types we expect callers to pass. The full runtime semantics
(zero-pad, clamp, OOB bounds, 24LCxx wire protocol, etc.) are covered by
the C++ gtest suite in tests/vb1940_emulator_test.cpp -- duplicating them
in Python would require exposing i2c_transaction as a Python-friendly
function, which is outside the scope of this patch.

The tests import the pybind11 extension module (_emulation_sensors) directly
rather than going through `hololink.emulation.sensors`. This keeps the smoke
test focused on the new symbols and avoids depending on the rest of the
hololink package being importable in the local test environment.
"""

import array

import pytest

import hololink.emulation.sensors as hemusens


def test_constant_is_exported_and_correct():
    assert hemusens.Vb1940Emulator.EEPROM_REGION_BYTES == 256


def test_set_eeprom_data_accepts_bytes():
    emu = hemusens.Vb1940Emulator()
    emu.set_eeprom_data(b"\x01\x02\x03\x04")


def test_set_eeprom_data_accepts_bytearray():
    emu = hemusens.Vb1940Emulator()
    emu.set_eeprom_data(bytearray(b"\xde\xad\xbe\xef"))


def test_set_eeprom_data_accepts_memoryview():
    emu = hemusens.Vb1940Emulator()
    backing = bytearray(b"\x10" * 32)
    emu.set_eeprom_data(memoryview(backing))


def test_set_eeprom_data_handles_full_region():
    emu = hemusens.Vb1940Emulator()
    blob = bytes(i % 256 for i in range(hemusens.Vb1940Emulator.EEPROM_REGION_BYTES))
    assert len(blob) == hemusens.Vb1940Emulator.EEPROM_REGION_BYTES
    emu.set_eeprom_data(blob)


def test_set_eeprom_data_handles_short_payload():
    emu = hemusens.Vb1940Emulator()
    emu.set_eeprom_data(b"\x01\x02\x03\x04\x05\x06\x07\x08")


def test_set_eeprom_data_handles_oversized_payload():
    emu = hemusens.Vb1940Emulator()
    emu.set_eeprom_data(b"\xab" * (hemusens.Vb1940Emulator.EEPROM_REGION_BYTES * 2))


def test_set_eeprom_data_handles_empty_payload():
    emu = hemusens.Vb1940Emulator()
    emu.set_eeprom_data(b"")


def test_reset_after_set_eeprom_data_does_not_crash():
    emu = hemusens.Vb1940Emulator()
    emu.set_eeprom_data(b"\x55" * 64)
    emu.reset()


def test_set_eeprom_data_rejects_non_byte_buffer():
    """A buffer with itemsize != 1 must be rejected with a clear error."""
    emu = hemusens.Vb1940Emulator()
    # array.array('I', ...) -> unsigned int, itemsize == 4.
    with pytest.raises(ValueError, match="byte buffer"):
        emu.set_eeprom_data(array.array("I", [0] * 16))


def test_set_eeprom_data_rejects_multidimensional_buffer():
    """A 2-D buffer must be rejected with a clear error."""
    emu = hemusens.Vb1940Emulator()
    two_d = memoryview(bytearray(256)).cast("B", shape=(16, 16))
    with pytest.raises(ValueError, match="1-D buffer"):
        emu.set_eeprom_data(two_d)


def test_set_eeprom_data_rejects_non_contiguous_buffer():
    """A strided (non-contiguous) view must be rejected with a clear error."""
    emu = hemusens.Vb1940Emulator()
    # Every-other-byte slice -> strides[0] == 2.
    strided = memoryview(bytearray(128))[::2]
    with pytest.raises(ValueError, match="contiguous"):
        emu.set_eeprom_data(strided)
