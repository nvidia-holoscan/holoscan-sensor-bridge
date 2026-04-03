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

import logging
import os
import time

import holoscan
import holoscan.core
import pytest

import hololink as hololink_module


class UartInitOp(holoscan.core.Operator):
    """Configures the UART and emits a trigger. Used with physical loopback."""

    def __init__(
        self,
        fragment,
        hololink_channel,
        uart,
        port_number,
        baud_rate=115200,
        data_bits=8,
        parity=0,
        stop_bits=1,
        flow_control=0,
        *args,
        **kwargs,
    ):
        self._hololink = hololink_channel.hololink()
        self._uart = uart
        self._port_number = port_number
        self._baud_rate = baud_rate
        self._data_bits = data_bits
        self._parity = parity
        self._stop_bits = stop_bits
        self._flow_control = flow_control
        self._baud_rate_options = [115200, 57600, 38400, 19200, 9600]
        self._baud_rate_index = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: holoscan.core.OperatorSpec):
        spec.output("configured_out")

    def compute(self, op_input, op_output, context):
        self._baud_rate = self._baud_rate_options[self._baud_rate_index]
        self._baud_rate_index = (self._baud_rate_index + 1) % len(
            self._baud_rate_options
        )
        logging.info(
            f"UartInitOp: configuring UART port={self._port_number} baud={self._baud_rate} "
            f"data_bits={self._data_bits} parity={self._parity} stop_bits={self._stop_bits} "
            f"flow_control={self._flow_control}"
        )
        self._uart.uart_configure(
            baud_rate=self._baud_rate,
            data_bits=self._data_bits,
            parity=self._parity,
            stop_bits=self._stop_bits,
            flow_control=self._flow_control,
        )
        logging.info(
            "UartInitOp: Internal loopback DISABLED - using physical GPIO wiring"
        )
        op_output.emit(True, "configured_out")


class UartLoopbackOp(holoscan.core.Operator):
    """Writes a test string and reads it back. Requires physical loopback wiring."""

    def __init__(
        self,
        fragment,
        hololink_channel,
        uart,
        test_string,
        read_max_bytes,
        sleep_time,
        *args,
        **kwargs,
    ):
        self._hololink = hololink_channel.hololink()
        self._uart = uart
        self._test_string = test_string
        self._read_max_bytes = read_max_bytes
        self._sleep_time = sleep_time
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: holoscan.core.OperatorSpec):
        spec.input("configured_in")

    def compute(self, op_input, op_output, context):
        _ = op_input.receive("configured_in")
        logging.info(
            f"UartLoopbackOp: current UART config: baud={self._uart.baud_rate()}"
        )
        try:
            data = self._test_string.encode("utf-8")
            chunk_size = 32
            total_bytes = len(data)
            num_chunks = (total_bytes + chunk_size - 1) // chunk_size
            logging.info(
                f"UartLoopbackOp: Writing {total_bytes} bytes in {num_chunks} chunks "
                f"of {chunk_size} bytes: {self._test_string!r}"
            )
            rx_str = ""
            for i in range(0, total_bytes, chunk_size):
                chunk = data[i : i + chunk_size]
                chunk_num = i // chunk_size + 1
                logging.debug(
                    f"uart_loopback: Sending chunk {chunk_num}/{num_chunks} ({len(chunk)} bytes)"
                )
                self._uart.uart_write(chunk)
                rx, bytes_read = self._uart.uart_read(len(chunk))
                logging.debug(f"uart_loopback: Received {bytes_read} bytes")
                rx_str += bytes(rx).decode("utf-8", errors="replace")
                logging.info(
                    f"UartLoopbackOp: Actual bytes Read this round:{bytes_read} , "
                    f"String collected so far: {rx_str!r}"
                )
            logging.info(f"UartLoopbackOp: Read {len(rx_str)} total bytes: {rx_str!r}")
            if rx_str != self._test_string:
                logging.error(
                    "UART loopback mismatch: expected %d bytes %r, got %d bytes %r",
                    len(self._test_string),
                    self._test_string,
                    len(rx_str),
                    rx_str,
                )
            logging.info(
                "\n\n============================================================================================================================\n\n"
            )
            time.sleep(self._sleep_time)
        except Exception as e:
            logging.error(f"UART write/read failed: {e}")


class UartLoopbackApplication(holoscan.core.Application):
    """Holoscan app for UART loopback test."""

    def __init__(
        self,
        hololink_channel,
        channel_metadata,
        port_number,
        baud_rate,
        test_string,
        read_max_bytes,
        sleep_time,
        cycle_limit,
        data_bits=8,
        parity=0,
        stop_bits=1,
        flow_control=0,
    ):
        super().__init__()
        self._hololink_channel = hololink_channel
        self._channel_metadata = channel_metadata
        self._hololink = hololink_channel.hololink()
        self._port_number = port_number
        self._baud_rate = baud_rate
        self._data_bits = data_bits
        self._parity = parity
        self._stop_bits = stop_bits
        self._flow_control = flow_control
        self._test_string = test_string
        self._read_max_bytes = read_max_bytes
        self._sleep_time = sleep_time
        self._cycle_limit = cycle_limit

    def compose(self):
        self._uart = self._hololink.get_uart(
            self._port_number,
            self._baud_rate,
            self._data_bits,
            self._parity,
            self._stop_bits,
            self._flow_control,
        )
        if self._cycle_limit:
            self._count = holoscan.conditions.CountCondition(
                self,
                name="count",
                count=self._cycle_limit,
            )
            condition = self._count
        else:
            self._ok = holoscan.conditions.BooleanCondition(
                self, name="ok", enable_tick=True
            )
            condition = self._ok
        uart_init = UartInitOp(
            self,
            self._hololink_channel,
            self._uart,
            self._port_number,
            self._baud_rate,
            self._data_bits,
            self._parity,
            self._stop_bits,
            self._flow_control,
            condition,
            name="uart_init",
        )
        uart_loopback = UartLoopbackOp(
            self,
            self._hololink_channel,
            self._uart,
            self._test_string,
            self._read_max_bytes,
            self._sleep_time,
            name="uart_loopback",
        )
        self.add_flow(
            uart_init,
            uart_loopback,
            {
                ("configured_out", "configured_in"),
            },
        )


# Default test string (512-byte repeating pattern)
_TEST_PATTERN = "0123456789ABCDEF"
_DEFAULT_TEST_STRING = _TEST_PATTERN * 32


@pytest.mark.skip_unless_hsb_nano
def test_uart_loopback(hololink_address, frame_limit, caplog):
    """Run UART loopback (GPIO 10 TX to GPIO 11 RX). Uses frame_limit as cycle limit."""
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "examples", "example_configuration.yaml"
    )
    channel_metadata = hololink_module.Enumerator.find_channel(
        channel_ip=hololink_address
    )
    hololink_channel = hololink_module.DataChannel(channel_metadata)
    application = UartLoopbackApplication(
        hololink_channel,
        channel_metadata,
        port_number=0,
        baud_rate=115200,
        test_string=_DEFAULT_TEST_STRING,
        read_max_bytes=256,
        sleep_time=0,
        cycle_limit=frame_limit,
        data_bits=8,
        parity=0,
        stop_bits=1,
        flow_control=0,
    )
    application.config(config_path)
    hololink = hololink_channel.hololink()
    hololink.start()
    hololink.reset()
    with caplog.at_level(logging.ERROR):
        application.run()
    hololink.stop()
    error_logs = [rec for rec in caplog.records if rec.levelno >= logging.ERROR]
    assert not error_logs, (
        f"UART loopback logged {len(error_logs)} error(s); "
        f"first: {error_logs[0].getMessage()}"
    )
