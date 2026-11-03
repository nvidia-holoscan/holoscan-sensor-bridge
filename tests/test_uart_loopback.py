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
# UART loopback test: requires physical loopback (GPIO 10 TX to GPIO 11 RX on
# the Lattice Hololink board). Parameters come from pytest (e.g. frame_limit,
# hololink_address) instead of command-line.
#
# HSDK5 difference from v2.7: v2.7 ran each cycle as two Holoscan operators,
# UartInitOp (configure the UART at the next baud rate) then UartLoopbackOp (write
# the test string and read it back), under a CountCondition. They carried no data
# between them but a trigger, and HSDK5 has no CountCondition, so the test runs
# the same cycle, with the same UART calls, frame_limit times in a loop.

import logging
import time

import pytest

import hololink as hololink_module

BAUD_RATE_OPTIONS = [115200, 57600, 38400, 19200, 9600]


def configure_uart(uart, cycle, data_bits, parity, stop_bits, flow_control):
    """UartInitOp: configure the UART at the next baud rate."""
    baud_rate = BAUD_RATE_OPTIONS[cycle % len(BAUD_RATE_OPTIONS)]
    logging.info(
        f"configuring UART baud={baud_rate} data_bits={data_bits} parity={parity} "
        f"stop_bits={stop_bits} flow_control={flow_control}"
    )
    uart.uart_configure(
        baud_rate=baud_rate,
        data_bits=data_bits,
        parity=parity,
        stop_bits=stop_bits,
        flow_control=flow_control,
    )
    logging.info("Internal loopback DISABLED - using physical GPIO wiring")


def loopback(uart, test_string, sleep_time):
    """UartLoopbackOp: write test_string and read it back; return what was read."""
    logging.info(f"current UART config: baud={uart.baud_rate()}")
    data = test_string.encode("utf-8")
    chunk_size = 32
    total_bytes = len(data)
    num_chunks = (total_bytes + chunk_size - 1) // chunk_size
    logging.info(
        f"Writing {total_bytes} bytes in {num_chunks} chunks "
        f"of {chunk_size} bytes: {test_string!r}"
    )
    rx_str = ""
    for i in range(0, total_bytes, chunk_size):
        chunk = data[i : i + chunk_size]
        chunk_num = i // chunk_size + 1
        logging.debug(
            f"uart_loopback: Sending chunk {chunk_num}/{num_chunks} ({len(chunk)} bytes)"
        )
        uart.uart_write(list(chunk))
        rx, bytes_read = uart.uart_read(len(chunk))
        logging.debug(f"uart_loopback: Received {bytes_read} bytes")
        rx_str += bytes(rx).decode("utf-8", errors="replace")
    logging.info(f"Read {len(rx_str)} total bytes: {rx_str!r}")
    time.sleep(sleep_time)
    return rx_str


# Default test string (512-byte repeating pattern)
_TEST_PATTERN = "0123456789ABCDEF"
_DEFAULT_TEST_STRING = _TEST_PATTERN * 32


@pytest.mark.skip_unless_hsb_nano
def test_uart_loopback(hololink_address, frame_limit, caplog):
    """Run UART loopback (GPIO 10 TX to GPIO 11 RX). Uses frame_limit as cycle limit."""
    channel_metadata = hololink_module.find_channel(hololink_address)
    hololink_channel = hololink_module.DataChannel(channel_metadata)
    hololink = hololink_channel.hololink()
    hololink.start()
    mismatches = 0
    try:
        hololink.reset()
        uart = hololink.get_uart(0, 115200, 8, 0, 1, 0)
        with caplog.at_level(logging.ERROR):
            for cycle in range(frame_limit):
                configure_uart(
                    uart, cycle, data_bits=8, parity=0, stop_bits=1, flow_control=0
                )
                try:
                    rx_str = loopback(uart, _DEFAULT_TEST_STRING, sleep_time=0)
                except Exception as e:
                    logging.error(f"UART write/read failed: {e}")
                    continue
                if rx_str != _DEFAULT_TEST_STRING:
                    mismatches += 1
                    logging.error(
                        "UART loopback mismatch: expected %d bytes %r, got %d bytes %r",
                        len(_DEFAULT_TEST_STRING),
                        _DEFAULT_TEST_STRING,
                        len(rx_str),
                        rx_str,
                    )
    finally:
        hololink.stop()
    error_logs = [rec for rec in caplog.records if rec.levelno >= logging.ERROR]
    assert not error_logs, (
        f"UART loopback logged {len(error_logs)} error(s); "
        f"first: {error_logs[0].getMessage()}"
    )
    assert mismatches == 0
