# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# UART FTDI test: Hololink UART TX -> FTDI USB serial RX. Parameters come from
# pytest (hololink_address, uart_ftdi_serial_port) instead of command-line.

import logging
import os
import time

import pytest

import hololink as hololink_module

# pyserial optional; skip test if not installed
try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

FTDI_USB_VID = 0x0403


def _find_ftdi_serial_port():
    """First FTDI USB serial port (VID 0x0403), or None if none found."""
    if serial is None:
        return None
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    for port in ports:
        if getattr(port, "vid", None) == FTDI_USB_VID:
            return port.device
    logging.warning(
        "No FTDI serial port (VID 0x0403) found; %d non-FTDI port(s) present. "
        "Connect an FTDI device or set UART_FTDI_SERIAL_PORT.",
        len(ports),
    )
    return None


def _serial_params_from_cli(data_bits=8, parity=0, stop_bits=1):
    """Return (bytesize, parity, stopbits) pyserial constants (8N1 default)."""
    bytesize = {
        5: serial.FIVEBITS,
        6: serial.SIXBITS,
        7: serial.SEVENBITS,
        8: serial.EIGHTBITS,
    }.get(data_bits, serial.EIGHTBITS)
    parity_val = {
        0: serial.PARITY_NONE,
        1: serial.PARITY_ODD,
        2: serial.PARITY_EVEN,
    }.get(parity, serial.PARITY_NONE)
    stopbits = {
        1: serial.STOPBITS_ONE,
        15: serial.STOPBITS_ONE_POINT_FIVE,
        2: serial.STOPBITS_TWO,
    }.get(stop_bits, serial.STOPBITS_ONE)
    return bytesize, parity_val, stopbits


def _send_from_hololink(
    hololink_ip,
    message,
    port_number=0,
    baud_rate=115200,
    data_bits=8,
    parity=0,
    stop_bits=1,
    flow_control=0,
):
    """Send message from Hololink board via UART."""
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=hololink_ip)
    hololink_channel = hololink_module.DataChannel(channel_metadata)
    hololink = hololink_channel.hololink()
    started = False
    hololink.start()
    started = True
    try:
        hololink.reset()
        uart = hololink.get_uart(
            port_number=port_number,
            baud_rate=baud_rate,
            data_bits=data_bits,
            parity=parity,
            stop_bits=stop_bits,
            flow_control=flow_control,
        )
        uart.uart_configure(
            baud_rate=baud_rate,
            data_bits=data_bits,
            parity=parity,
            stop_bits=stop_bits,
            flow_control=flow_control,
        )
        data = message.encode("utf-8") if isinstance(message, str) else message
        chunk_size = 256
        for i in range(0, len(data), chunk_size):
            chunk = data[i : i + chunk_size]
            uart.uart_write(list(chunk))
    finally:
        if started:
            hololink.stop()


def _read_from_serial_until_timeout(
    serial_port,
    baud_rate,
    read_timeout,
    expected_bytes,
    data_bits=8,
    parity=0,
    stop_bits=1,
    flow_control=0,
):
    """Open serial port, read until expected_bytes or timeout. Returns bytes."""
    bytesize, parity_val, stopbits = _serial_params_from_cli(
        data_bits, parity, stop_bits
    )
    ser = serial.Serial(
        port=serial_port,
        baudrate=baud_rate,
        bytesize=bytesize,
        parity=parity_val,
        stopbits=stopbits,
        rtscts=(flow_control == 1),
        timeout=read_timeout,
    )
    received = bytearray()
    start_time = time.time()
    try:
        while (time.time() - start_time) <= read_timeout:
            data = ser.read(1024)
            if data:
                received.extend(data)
                start_time = time.time()
                if len(received) >= expected_bytes:
                    time.sleep(0.1)
                    if ser.in_waiting == 0:
                        break
            elif len(received) > 0:
                time.sleep(0.1)
                if ser.in_waiting == 0:
                    break
    finally:
        ser.close()
    return bytes(received)


_DEFAULT_MESSAGE = "Hello from Hololink! This is a test message."


@pytest.mark.skip_unless_hsb_nano
def test_uart_ftdi_app(hololink_address, capsys):
    """UART Hololink->FTDI: send from board, read from USB serial. Requires hardware."""
    pytest.importorskip("serial")
    serial_port = os.environ.get("UART_FTDI_SERIAL_PORT") or _find_ftdi_serial_port()
    if not serial_port:
        pytest.skip(
            "No serial port found. Connect FTDI or set UART_FTDI_SERIAL_PORT to "
            "override auto-detect"
        )
    message = _DEFAULT_MESSAGE
    expected = message.encode("utf-8")
    read_timeout = 3.0
    port_number = 0
    baud_rate = 115200
    data_bits = 8
    parity = 0
    stop_bits = 1
    flow_control = 0

    # Open serial port before sending so the host is listening when the board transmits
    bytesize, parity_val, stopbits = _serial_params_from_cli(
        data_bits, parity, stop_bits
    )
    ser = serial.Serial(
        port=serial_port,
        baudrate=baud_rate,
        bytesize=bytesize,
        parity=parity_val,
        stopbits=stopbits,
        rtscts=(flow_control == 1),
        timeout=read_timeout,
    )
    try:
        _send_from_hololink(
            hololink_ip=hololink_address,
            message=message,
            port_number=port_number,
            baud_rate=baud_rate,
            data_bits=data_bits,
            parity=parity,
            stop_bits=stop_bits,
            flow_control=flow_control,
        )
        received = bytearray()
        start_time = time.time()
        while (time.time() - start_time) <= read_timeout:
            data = ser.read(1024)
            if data:
                received.extend(data)
                start_time = time.time()
                if len(received) >= len(expected):
                    time.sleep(0.1)
                    if ser.in_waiting == 0:
                        break
            elif len(received) > 0:
                time.sleep(0.1)
                if ser.in_waiting == 0:
                    break
        received = bytes(received)
    finally:
        ser.close()

    success = received == expected

    captured = capsys.readouterr()
    assert captured.err == ""
    assert (
        success
    ), f"UART hololink-to-serial failed: sent {message!r}, received {received!r}"
