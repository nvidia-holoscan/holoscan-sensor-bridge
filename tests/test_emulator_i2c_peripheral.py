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
Tests for the Python side of hemu.I2CPeripheral.

Two directions are covered:

1. C++ -> Python: a peripheral implemented by subclassing hemu.I2CPeripheral in
   Python must receive the host's complete write buffer and be able to return
   read data. This goes through the pybind11 trampoline in _emulation.cpp. The
   transaction is driven the way the host's I2c sequencer does it -- by writing
   the I2C controller registers through the emulator's control-plane memory
   (HSBEmulator.write / HSBEmulator.read) -- so no network or hardware is
   needed.

2. Python -> C++: calling i2c_transaction on a C++ sensor emulator (TestSensor)
   from Python with ordinary byte buffers.

Regression context: through 2.7.0 the trampoline forwarded the raw `uint8_t*`
pointers to PYBIND11_OVERRIDE_PURE, which cast them as single ints, so a Python
override saw only the first byte of the write buffer and could not return reads.
"""

import array
import gc
import struct

import pytest

import hololink as hololink_module

# A wheel built without the emulator (HOLOLINK_BUILD_EMULATOR=OFF) has no
# hololink.emulation: skip rather than fail at collection.
hemu = pytest.importorskip("hololink.emulation")
hemusens = pytest.importorskip("hololink.emulation.sensors")

PERIPHERAL_ADDRESS = 0x1A
# I2C controller register map (mirrors hololink.hpp / hsb_config.hpp).
I2C_CTRL = hololink_module.I2C_CTRL
I2C_REG_CONTROL = I2C_CTRL + hololink_module.I2C_REG_CONTROL
I2C_REG_BUS_EN = I2C_CTRL + hololink_module.I2C_REG_BUS_EN
I2C_REG_NUM_BYTES = I2C_CTRL + hololink_module.I2C_REG_NUM_BYTES
I2C_REG_STATUS = I2C_CTRL + hololink_module.I2C_REG_STATUS
I2C_REG_DATA_BUFFER = I2C_CTRL + hololink_module.I2C_REG_DATA_BUFFER
I2C_START = hololink_module.I2C_START
I2C_DONE = hololink_module.I2C_DONE
I2C_I2C_ERR = hololink_module.I2C_I2C_ERR


def _emulator_i2c_transaction(hsb, bus, peripheral_address, write_bytes, read_count):
    """Drive one I2C transaction through the emulator's control plane.

    Same register sequence as Hololink::I2c::encode_i2c_request on the host:
    clear control, fill the data buffer with little-endian words, program
    BUS_EN and NUM_BYTES, kick START, poll STATUS, read the data buffer back.
    Returns (status, read_bytes).
    """
    assert hsb.write(I2C_REG_CONTROL, 0) == 0
    padded = bytes(write_bytes) + b"\0" * (-len(write_bytes) % 4)
    for i in range(0, len(padded), 4):
        (word,) = struct.unpack_from("<I", padded, i)
        assert hsb.write(I2C_REG_DATA_BUFFER + i, word) == 0
    assert hsb.write(I2C_REG_BUS_EN, bus) == 0
    assert hsb.write(I2C_REG_NUM_BYTES, len(write_bytes) | (read_count << 16)) == 0
    assert hsb.write(I2C_REG_CONTROL, (peripheral_address << 16) | I2C_START) == 0
    # The emulator executes the transaction synchronously inside the register write.
    status = hsb.read(I2C_REG_STATUS)
    data = bytearray()
    for i in range(0, (read_count + 3) // 4 * 4, 4):
        data += struct.pack("<I", hsb.read(I2C_REG_DATA_BUFFER + i))
    assert hsb.write(I2C_REG_CONTROL, 0) == 0
    return status, bytes(data[:read_count])


class RecordingPeripheral(hemu.I2CPeripheral):
    """A register-file peripheral that records every call it receives."""

    def __init__(self, status=hemu.I2CStatus.I2C_STATUS_SUCCESS):
        super().__init__()
        self.status = status
        self.calls = []
        self.registers = {}
        self.stashed_read_bytes = None

    def attach_to_i2c(self, controller, bus):
        controller.attach_i2c_peripheral(bus, PERIPHERAL_ADDRESS, self)

    def i2c_transaction(self, addr, write_bytes, write_size, read_bytes, read_size):
        self.calls.append(
            {
                "addr": addr,
                "write_bytes": write_bytes,
                "write_size": write_size,
                "read_bytes_type": type(read_bytes),
                "read_size": read_size,
                "read_len": len(read_bytes),
            }
        )
        self.stashed_read_bytes = read_bytes
        # 16-bit big-endian register address, optional data bytes to write.
        register = int.from_bytes(write_bytes[0:2], "big")
        for i, value in enumerate(write_bytes[2:]):
            self.registers[register + i] = value
        if read_size:
            read_bytes[0:read_size] = bytes(
                self.registers.get(register + i, 0) for i in range(read_size)
            )
        return self.status


@pytest.fixture
def hsb():
    emulator = hemu.HSBEmulator()
    yield emulator
    emulator.stop()


def test_python_peripheral_receives_full_write_buffer(hsb):
    peripheral = RecordingPeripheral()
    peripheral.attach_to_i2c(
        hsb.get_i2c(hololink_module.I2C_CTRL), hololink_module.CAM_I2C_BUS
    )
    hsb.start()

    # 7 bytes: not word aligned, exercises the partial trailing word.
    payload = bytes([0x30, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05])
    status, data = _emulator_i2c_transaction(
        hsb, hololink_module.CAM_I2C_BUS, PERIPHERAL_ADDRESS, payload, 0
    )
    assert status & I2C_DONE
    assert not status & I2C_I2C_ERR
    assert data == b""

    assert len(peripheral.calls) == 1
    call = peripheral.calls[0]
    assert call["addr"] == PERIPHERAL_ADDRESS
    assert isinstance(call["write_bytes"], bytes)
    assert call["write_bytes"] == payload
    assert call["write_size"] == len(payload)
    assert call["read_bytes_type"] is bytearray
    assert call["read_size"] == 0
    assert call["read_len"] == 0


def test_python_peripheral_returns_read_data(hsb):
    peripheral = RecordingPeripheral()
    peripheral.attach_to_i2c(
        hsb.get_i2c(hololink_module.I2C_CTRL), hololink_module.CAM_I2C_BUS
    )
    hsb.start()

    # Write 0x30,0x00 <- 0xA5 0x5A 0xC3, then read 3 bytes back from the same register.
    status, _ = _emulator_i2c_transaction(
        hsb,
        hololink_module.CAM_I2C_BUS,
        PERIPHERAL_ADDRESS,
        bytes([0x30, 0x00, 0xA5, 0x5A, 0xC3]),
        0,
    )
    assert status & I2C_DONE and not status & I2C_I2C_ERR

    status, data = _emulator_i2c_transaction(
        hsb, hololink_module.CAM_I2C_BUS, PERIPHERAL_ADDRESS, bytes([0x30, 0x00]), 3
    )
    assert status & I2C_DONE and not status & I2C_I2C_ERR
    assert data == bytes([0xA5, 0x5A, 0xC3])

    call = peripheral.calls[-1]
    assert call["write_bytes"] == bytes([0x30, 0x00])
    assert call["read_size"] == 3
    assert call["read_len"] == 3

    # Odd read sizes exercise the partial trailing word on the host side.
    status, data = _emulator_i2c_transaction(
        hsb, hololink_module.CAM_I2C_BUS, PERIPHERAL_ADDRESS, bytes([0x30, 0x01]), 1
    )
    assert status & I2C_DONE and not status & I2C_I2C_ERR
    assert data == bytes([0x5A])


def test_python_peripheral_read_bytes_is_a_private_copy(hsb):
    """read_bytes is the override's own bytearray, copied into the controller's buffer
    when the override returns: a reference kept past the call neither sees nor changes
    the controller's data."""
    peripheral = RecordingPeripheral()
    peripheral.attach_to_i2c(
        hsb.get_i2c(hololink_module.I2C_CTRL), hololink_module.CAM_I2C_BUS
    )
    hsb.start()

    _emulator_i2c_transaction(
        hsb,
        hololink_module.CAM_I2C_BUS,
        PERIPHERAL_ADDRESS,
        bytes([0x30, 0x00, 0xA5, 0x5A]),
        0,
    )
    status, data = _emulator_i2c_transaction(
        hsb, hololink_module.CAM_I2C_BUS, PERIPHERAL_ADDRESS, bytes([0x30, 0x00]), 2
    )
    assert status & I2C_DONE
    assert data == bytes([0xA5, 0x5A])
    stashed = peripheral.stashed_read_bytes
    assert isinstance(stashed, bytearray)
    assert bytes(stashed) == bytes([0xA5, 0x5A])
    # Writing to the stashed buffer after the call does not reach the controller.
    stashed[0] = 0xFF
    assert hsb.read(I2C_REG_DATA_BUFFER) & 0xFFFF == 0x5AA5


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_python_peripheral_exception_becomes_error_status(hsb):
    """An exception escaping the override is reported as unraisable and the
    transaction fails with I2C_I2C_ERR, instead of ending the process (the override
    runs on the emulator's control thread)."""

    class Faulty(RecordingPeripheral):
        def i2c_transaction(self, addr, write_bytes, write_size, read_bytes, read_size):
            raise RuntimeError("peripheral bug")

    Faulty().attach_to_i2c(
        hsb.get_i2c(hololink_module.I2C_CTRL), hololink_module.CAM_I2C_BUS
    )
    hsb.start()
    status, _ = _emulator_i2c_transaction(
        hsb, hololink_module.CAM_I2C_BUS, PERIPHERAL_ADDRESS, bytes([0x30, 0x00]), 2
    )
    assert status & I2C_I2C_ERR
    assert not status & I2C_DONE

    class WrongType(RecordingPeripheral):
        def i2c_transaction(self, addr, write_bytes, write_size, read_bytes, read_size):
            return "not a status"

    hsb.stop()
    other = hemu.HSBEmulator()
    try:
        WrongType().attach_to_i2c(
            other.get_i2c(hololink_module.I2C_CTRL), hololink_module.CAM_I2C_BUS
        )
        other.start()
        status, _ = _emulator_i2c_transaction(
            other,
            hololink_module.CAM_I2C_BUS,
            PERIPHERAL_ADDRESS,
            bytes([0x30, 0x00]),
            2,
        )
        assert status & I2C_I2C_ERR
    finally:
        other.stop()


def test_python_peripheral_error_status_is_reported(hsb):
    peripheral = RecordingPeripheral(status=hemu.I2CStatus.I2C_STATUS_NACK)
    peripheral.attach_to_i2c(
        hsb.get_i2c(hololink_module.I2C_CTRL), hololink_module.CAM_I2C_BUS
    )
    hsb.start()

    status, _ = _emulator_i2c_transaction(
        hsb, hololink_module.CAM_I2C_BUS, PERIPHERAL_ADDRESS, bytes([0x30, 0x00]), 2
    )
    assert status & I2C_I2C_ERR
    assert not status & I2C_DONE
    assert len(peripheral.calls) == 1


I2C_DATA_BUFFER_SIZE = 0x100  # src/hololink/emulation/hsb_emulator.hpp


@pytest.mark.parametrize(
    "write_count,read_count",
    [
        (I2C_DATA_BUFFER_SIZE + 1, 0),
        (2, I2C_DATA_BUFFER_SIZE + 1),
        (0x1000, 0x1000),
        (0xFFFF, 0xFFFF),
    ],
)
def test_oversized_transfer_length_is_rejected(hsb, write_count, read_count):
    """A NUM_BYTES count larger than the data window must fail the transaction
    without ever reaching the peripheral (it would otherwise read or write past
    the controller's data buffer)."""
    peripheral = RecordingPeripheral()
    peripheral.attach_to_i2c(
        hsb.get_i2c(hololink_module.I2C_CTRL), hololink_module.CAM_I2C_BUS
    )
    hsb.start()

    assert hsb.write(I2C_REG_CONTROL, 0) == 0
    assert hsb.write(I2C_REG_DATA_BUFFER, 0x00000030) == 0
    assert hsb.write(I2C_REG_BUS_EN, hololink_module.CAM_I2C_BUS) == 0
    assert hsb.write(I2C_REG_NUM_BYTES, write_count | (read_count << 16)) == 0
    assert hsb.write(I2C_REG_CONTROL, (PERIPHERAL_ADDRESS << 16) | I2C_START) == 0
    status = hsb.read(I2C_REG_STATUS)
    assert status & I2C_I2C_ERR
    assert not status & I2C_DONE
    assert peripheral.calls == []

    # The controller is still usable afterwards.
    status, data = _emulator_i2c_transaction(
        hsb,
        hololink_module.CAM_I2C_BUS,
        PERIPHERAL_ADDRESS,
        bytes([0x30, 0x00, 0x5A]),
        0,
    )
    assert status & I2C_DONE
    status, data = _emulator_i2c_transaction(
        hsb, hololink_module.CAM_I2C_BUS, PERIPHERAL_ADDRESS, bytes([0x30, 0x00]), 1
    )
    assert status & I2C_DONE
    assert data == b"\x5a"
    assert len(peripheral.calls) == 2


def test_full_window_transfer_length_is_accepted(hsb):
    """Exactly I2C_DATA_BUFFER_SIZE bytes in each direction is the largest legal
    transfer and must still reach the peripheral intact."""
    peripheral = RecordingPeripheral()
    peripheral.attach_to_i2c(
        hsb.get_i2c(hololink_module.I2C_CTRL), hololink_module.CAM_I2C_BUS
    )
    hsb.start()

    payload = bytes(range(I2C_DATA_BUFFER_SIZE - 2))
    status, _ = _emulator_i2c_transaction(
        hsb,
        hololink_module.CAM_I2C_BUS,
        PERIPHERAL_ADDRESS,
        bytes([0x30, 0x00]) + payload,
        0,
    )
    assert status & I2C_DONE
    assert peripheral.calls[-1]["write_bytes"] == bytes([0x30, 0x00]) + payload

    status, data = _emulator_i2c_transaction(
        hsb,
        hololink_module.CAM_I2C_BUS,
        PERIPHERAL_ADDRESS,
        bytes([0x30, 0x00]),
        I2C_DATA_BUFFER_SIZE,
    )
    assert status & I2C_DONE
    assert peripheral.calls[-1]["read_size"] == I2C_DATA_BUFFER_SIZE
    assert data[: len(payload)] == payload
    assert data[len(payload) :] == b"\x00\x00"


def test_python_peripheral_only_sees_its_own_address(hsb):
    peripheral = RecordingPeripheral()
    peripheral.attach_to_i2c(
        hsb.get_i2c(hololink_module.I2C_CTRL), hololink_module.CAM_I2C_BUS
    )
    hsb.start()

    # Nothing attached at 0x1B: the controller treats it as a stateless no-op.
    status, _ = _emulator_i2c_transaction(
        hsb, hololink_module.CAM_I2C_BUS, PERIPHERAL_ADDRESS + 1, bytes([0x30, 0x00]), 2
    )
    assert status & I2C_DONE
    assert peripheral.calls == []


def test_attached_python_peripheral_outlives_callers_reference(hsb):
    """The emulator must keep an attached peripheral alive even if the application
    drops its own reference (the controller only holds a raw pointer in C++)."""

    class Probe(RecordingPeripheral):
        instances = 0

        def __init__(self):
            super().__init__()
            Probe.instances += 1

        def __del__(self):
            Probe.instances -= 1

    Probe().attach_to_i2c(
        hsb.get_i2c(hololink_module.I2C_CTRL), hololink_module.CAM_I2C_BUS
    )
    gc.collect()
    assert Probe.instances == 1  # still referenced via the emulator

    hsb.start()
    status, data = _emulator_i2c_transaction(
        hsb,
        hololink_module.CAM_I2C_BUS,
        PERIPHERAL_ADDRESS,
        bytes([0x30, 0x00, 0x42]),
        0,
    )
    assert status & I2C_DONE and not status & I2C_I2C_ERR
    status, data = _emulator_i2c_transaction(
        hsb, hololink_module.CAM_I2C_BUS, PERIPHERAL_ADDRESS, bytes([0x30, 0x00]), 1
    )
    assert status & I2C_DONE and not status & I2C_I2C_ERR
    assert data == bytes([0x42])
    hsb.stop()  # calls Probe.stop(); must not touch a freed object


def test_attached_peripheral_is_released_with_the_emulator():
    """The emulator keeps the peripheral alive, and no longer: once the last reference
    to the emulator (and to its controller) is gone, so is the peripheral. The emulator
    stops first, so the control thread never runs against a freed peripheral."""

    class Probe(RecordingPeripheral):
        instances = 0

        def __init__(self):
            super().__init__()
            Probe.instances += 1

        def __del__(self):
            Probe.instances -= 1

    hsb = hemu.HSBEmulator()
    controller = hsb.get_i2c(hololink_module.I2C_CTRL)
    Probe().attach_to_i2c(controller, hololink_module.CAM_I2C_BUS)
    hsb.start()
    gc.collect()
    assert Probe.instances == 1
    # The controller handle keeps the emulator alive; dropping both frees the peripheral.
    del hsb
    gc.collect()
    assert Probe.instances == 1
    del controller
    gc.collect()
    assert Probe.instances == 0


def test_get_i2c_other_address_keeps_attached_peripherals(hsb):
    """On Linux every controller address maps to the one controller; asking for
    another address must not drop the peripherals attached through the first."""

    class Probe(RecordingPeripheral):
        instances = 0

        def __init__(self):
            super().__init__()
            Probe.instances += 1

        def __del__(self):
            Probe.instances -= 1

    Probe().attach_to_i2c(
        hsb.get_i2c(hololink_module.I2C_CTRL), hololink_module.CAM_I2C_BUS
    )
    other = hsb.get_i2c(hololink_module.I2C_CTRL + 0x100)
    gc.collect()
    assert Probe.instances == 1
    assert other is hsb.get_i2c(hololink_module.I2C_CTRL)


def test_attached_cpp_sensor_outlives_callers_reference(hsb):
    """Same guarantee for a C++ sensor attached through its own attach_to_i2c()."""
    hemusens.TestSensor().attach_to_i2c(
        hsb.get_i2c(hololink_module.I2C_CTRL), hololink_module.CAM_I2C_BUS
    )
    gc.collect()
    hsb.start()
    status, data = _emulator_i2c_transaction(
        hsb,
        hololink_module.CAM_I2C_BUS,
        hemusens.TEST_I2C_ADDRESS,
        bytes([0x00, 0x00]),
        4,
    )
    assert status & I2C_DONE and not status & I2C_I2C_ERR
    assert data == b"HLTK"
    hsb.stop()


def test_get_i2c_returns_same_controller_object(hsb):
    a = hsb.get_i2c(hololink_module.I2C_CTRL)
    b = hsb.get_i2c()  # defaults to I2C_CTRL
    assert a is b


# ---------------------------------------------------------------------------
# Python -> C++: drive a C++ sensor emulator directly.
# ---------------------------------------------------------------------------


def test_cpp_sensor_i2c_transaction_from_python():
    sensor = hemusens.TestSensor()
    device_id = bytearray(4)
    status = sensor.i2c_transaction(
        hemusens.TEST_I2C_ADDRESS, bytes([0x00, 0x00]), device_id
    )
    assert status == hemu.I2CStatus.I2C_STATUS_SUCCESS
    assert bytes(device_id) == b"HLTK"

    # Write the constant pattern byte (reg 0x0015) then read it back, using a
    # bytearray for the write side and a memoryview for the read side.
    status = sensor.i2c_transaction(
        hemusens.TEST_I2C_ADDRESS, bytearray([0x00, 0x15, 0x7E])
    )
    assert status == hemu.I2CStatus.I2C_STATUS_SUCCESS
    assert sensor.get_constant_byte() == 0x7E
    backing = bytearray(8)
    status = sensor.i2c_transaction(
        hemusens.TEST_I2C_ADDRESS,
        bytes([0x00, 0x15]),
        memoryview(backing),
        read_size=1,
    )
    assert status == hemu.I2CStatus.I2C_STATUS_SUCCESS
    assert backing[0] == 0x7E
    assert backing[1:] == bytearray(7)

    # Error paths are reported as a status, not an exception.
    status = sensor.i2c_transaction(
        hemusens.TEST_I2C_ADDRESS + 1, bytes([0x00, 0x00]), bytearray(4)
    )
    assert status == hemu.I2CStatus.I2C_STATUS_INVALID_PERIPHERAL_ADDRESS


def test_cpp_sensor_i2c_transaction_rejects_bad_buffers():
    sensor = hemusens.TestSensor()
    # Immutable read buffer.
    with pytest.raises((TypeError, BufferError, ValueError)):
        sensor.i2c_transaction(hemusens.TEST_I2C_ADDRESS, bytes([0x00, 0x00]), b"\0\0")
    # Non-byte element size.
    with pytest.raises(ValueError, match="byte buffer"):
        sensor.i2c_transaction(
            hemusens.TEST_I2C_ADDRESS, array.array("I", [0]), bytearray(4)
        )
    # Size larger than the buffer passed.
    with pytest.raises(ValueError, match="exceeds"):
        sensor.i2c_transaction(
            hemusens.TEST_I2C_ADDRESS, bytes([0x00, 0x00]), bytearray(2), read_size=4
        )


def test_python_peripheral_callable_from_python():
    """Calling i2c_transaction on a Python subclass dispatches straight to the override."""
    peripheral = RecordingPeripheral()
    out = bytearray(2)
    status = peripheral.i2c_transaction(
        PERIPHERAL_ADDRESS, bytes([0x30, 0x00, 0x11, 0x22]), 4, out, 0
    )
    assert status == hemu.I2CStatus.I2C_STATUS_SUCCESS
    assert peripheral.registers == {0x3000: 0x11, 0x3001: 0x22}


# ---------------------------------------------------------------------------
# HSBEmulator.read returns the register value (it used to return the rc).
# ---------------------------------------------------------------------------


def test_hsb_emulator_read_returns_value(hsb):
    assert hsb.write(I2C_REG_DATA_BUFFER, 0x11223344) == 0
    assert hsb.read(I2C_REG_DATA_BUFFER) == 0x11223344
    with pytest.raises(RuntimeError, match="no register"):
        hsb.read(0xFFFF_FFF0)
