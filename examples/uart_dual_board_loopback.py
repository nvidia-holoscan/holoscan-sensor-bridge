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

"""
Dual-board UART test using Holoscan operators.

Physical wiring:
  Board 1 GPIO 10 (TX) → Board 2 GPIO 11 (RX)
  Board 2 GPIO 10 (TX) → Board 1 GPIO 11 (RX)

This app works like uart_loopback_app.py but with separate TX and RX boards.
Uses chunked writes/reads to handle the 256-byte FIFO limitation.

Usage:
  # Option 1: Separate TX/RX on different boards
  # Terminal 1 (Receiver):
  python3 uart_dual_board_loopback.py --mode rx --hololink 192.168.2.2

  # Terminal 2 (Transmitter):
  python3 uart_dual_board_loopback.py --mode tx --hololink 192.168.0.2

  # Option 2: Dual mode (both TX and RX on same board)
  # Terminal 1 (Board 1):
  python3 uart_dual_board_loopback.py --mode dual --hololink 192.168.0.2 --test-string "HELLO" --expected-rx-string "WORLD"

  # Terminal 2 (Board 2):
  python3 uart_dual_board_loopback.py --mode dual --hololink 192.168.2.2 --test-string "WORLD" --expected-rx-string "HELLO"
"""

import argparse
import logging
import os
import time

import holoscan

import hololink as hololink_module


class UartInitOp(holoscan.core.Operator):
    """
    Initializes and configures the UART with the provided parameters.
    Emits a simple trigger to downstream operators when done.
    """

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
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: holoscan.core.OperatorSpec):
        spec.output("configured_out")

    def compute(self, op_input, op_output, context):
        logging.info(
            f"UartInitOp: configuring UART port={self._port_number} baud={self._baud_rate} data_bits={self._data_bits} parity={self._parity} stop_bits={self._stop_bits} flow_control={self._flow_control}"
        )

        # Configure UART
        self._uart.uart_configure(
            baud_rate=self._baud_rate,
            data_bits=self._data_bits,
            parity=self._parity,
            stop_bits=self._stop_bits,
            flow_control=self._flow_control,
        )

        logging.info("UartInitOp: UART configured - ready for communication")

        # Give both boards time to configure
        time.sleep(2.0)

        # Emit to let downstream operators proceed
        op_output.emit(True, "configured_out")


class UartTxOp(holoscan.core.Operator):
    """
    Transmitter operator - writes test data to UART in chunks.
    Similar to UartLoopbackOp but only transmits (no readback).
    """

    def __init__(
        self,
        fragment,
        hololink_channel,
        uart,
        test_string,
        sleep_time,
        *args,
        **kwargs,
    ):
        self._hololink = hololink_channel.hololink()
        self._uart = uart
        self._test_string = test_string
        self._sleep_time = sleep_time
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: holoscan.core.OperatorSpec):
        spec.input("configured_in")
        spec.output("tx_done")  # Signal that transmission is complete

    def compute(self, op_input, op_output, context):
        _ = op_input.receive("configured_in")

        logging.info(f"UartTxOp: current UART config: baud={self._uart.baud_rate()}")

        # Write test string in chunks
        try:
            data = self._test_string.encode("utf-8")
            chunk_size = 16  # Using smaller chunks for flow control (256-byte FIFO)
            total_bytes = len(data)
            num_chunks = (
                total_bytes + chunk_size - 1
            ) // chunk_size  # Ceiling division

            logging.info(
                f"UartTxOp: Writing {total_bytes} bytes in {num_chunks} chunks of {chunk_size} bytes: {self._test_string!r}"
            )

            for i in range(0, total_bytes, chunk_size):
                chunk = data[i : i + chunk_size]
                chunk_num = i // chunk_size + 1
                chunk_str = chunk.decode("utf-8", errors="replace")

                logging.debug(
                    f"UartTxOp: Sending chunk {chunk_num}/{num_chunks} ({len(chunk)} bytes): '{chunk_str}'"
                )

                self._uart.uart_write(chunk)

            logging.info(f"UartTxOp: Transmission complete - sent {total_bytes} bytes")
            logging.info(
                "\n\n============================================================================================================================\n\n"
            )

            # Emit signal that transmission is done (for dual mode)
            op_output.emit(True, "tx_done")

            # Allow time before next cycle
            time.sleep(self._sleep_time)

        except Exception as e:
            logging.error(f"UART write failed: {e}")


class UartRxOp(holoscan.core.Operator):
    """
    Receiver operator - reads data from UART in chunks.
    Similar to UartLoopbackOp but only receives (no write).
    """

    def __init__(
        self,
        fragment,
        hololink_channel,
        uart,
        expected_string,
        read_max_bytes,
        sleep_time,
        *args,
        **kwargs,
    ):
        self._hololink = hololink_channel.hololink()
        self._uart = uart
        self._expected_string = expected_string
        self._read_max_bytes = read_max_bytes
        self._sleep_time = sleep_time
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: holoscan.core.OperatorSpec):
        spec.input("configured_in")

    def compute(self, op_input, op_output, context):
        _ = op_input.receive("configured_in")

        logging.info(f"UartRxOp: current UART config: baud={self._uart.baud_rate()}")

        # Read test string from FIFO
        try:
            expected_data = self._expected_string.encode("utf-8")
            total_bytes = len(expected_data)

            logging.info(f"UartRxOp: Expecting {total_bytes} bytes")

            # Initialize accumulators
            received = bytearray()
            start_time = time.time()
            timeout = 30.0

            # Read from FIFO until we have all expected bytes
            while len(received) < total_bytes:
                if time.time() - start_time > timeout:
                    logging.error(
                        f"UartRxOp: TIMEOUT! Got {len(received)}/{total_bytes} bytes"
                    )
                    break

                # Read whatever is available in FIFO (up to remaining bytes needed)
                bytes_to_read = min(
                    self._read_max_bytes, 256, total_bytes - len(received)
                )  # Respect user-specified read_max_bytes, FIFO limit (256), and remaining bytes
                rx, bytes_read = self._uart.uart_read(bytes_to_read)

                if bytes_read > 0:
                    # Filter out null bytes (empty FIFO reads)
                    real_data = bytes([b for b in rx if b != 0])
                    if len(real_data) > 0:
                        received.extend(real_data)
                        logging.debug(
                            f"UartRxOp: Read {len(real_data)} bytes (total: {len(received)}/{total_bytes})"
                        )

            # Decode received data
            rx_str = received.decode("utf-8", errors="replace")
            logging.info(f"UartRxOp: Read {len(received)} total bytes: {rx_str!r}")

            # Compare received string to expected one
            if rx_str == self._expected_string:
                logging.info("UartRxOp: SUCCESS! Data matches perfectly!")
            else:
                logging.error("UartRxOp: MISMATCH!")
                logging.error(f"  Expected: {self._expected_string!r}")
                logging.error(f"  Received: {rx_str!r}")

            logging.info(
                "\n\n============================================================================================================================\n\n"
            )

            # Allow time before next cycle
            time.sleep(self._sleep_time)

        except Exception as e:
            logging.error(f"UART read failed: {e}")


class HoloscanApplication(holoscan.core.Application):
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
        mode,  # 'tx', 'rx', or 'dual'
        expected_rx_string=None,  # For dual mode
        data_bits=8,
        parity=0,
        stop_bits=1,
        flow_control=0,
    ):
        logging.info("__init__")
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
        self._mode = mode
        self._expected_rx_string = expected_rx_string or test_string

    def compose(self):
        logging.info("compose")

        # Create the UART instance to be shared with the operators
        self._uart = self._hololink.get_uart(
            self._port_number,
            self._baud_rate,
            self._data_bits,
            self._parity,
            self._stop_bits,
            self._flow_control,
        )

        # Conditions support cycle-limit
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

        if self._mode == "tx":
            # Transmitter mode
            uart_tx = UartTxOp(
                self,
                self._hololink_channel,
                self._uart,
                self._test_string,
                self._sleep_time,
                name="uart_tx",
            )

            self.add_flow(
                uart_init,
                uart_tx,
                {
                    ("configured_out", "configured_in"),
                },
            )

        elif self._mode == "rx":
            # Receiver mode
            uart_rx = UartRxOp(
                self,
                self._hololink_channel,
                self._uart,
                self._test_string,  # Expected string
                self._read_max_bytes,
                self._sleep_time,
                name="uart_rx",
            )

            self.add_flow(
                uart_init,
                uart_rx,
                {
                    ("configured_out", "configured_in"),
                },
            )

        elif self._mode == "dual":
            # Dual mode: both TX and RX
            uart_tx = UartTxOp(
                self,
                self._hololink_channel,
                self._uart,
                self._test_string,  # String to transmit
                self._sleep_time,
                name="uart_tx",
            )

            uart_rx = UartRxOp(
                self,
                self._hololink_channel,
                self._uart,
                self._expected_rx_string,  # Expected string from other board
                self._read_max_bytes,
                self._sleep_time,
                name="uart_rx",
            )

            # Flow: init -> tx -> rx
            self.add_flow(
                uart_init,
                uart_tx,
                {
                    ("configured_out", "configured_in"),
                },
            )
            self.add_flow(
                uart_tx,
                uart_rx,
                {
                    ("tx_done", "configured_in"),  # TX completes, then RX starts
                },
            )

        else:
            raise ValueError(
                f"Invalid mode: {self._mode}. Must be 'tx', 'rx', or 'dual'"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Dual-board UART test using Holoscan operators"
    )

    default_configuration = os.path.join(
        os.path.dirname(__file__), "example_configuration.yaml"
    )
    parser.add_argument(
        "--configuration",
        default=default_configuration,
        help="Configuration file",
    )
    parser.add_argument(
        "--hololink",
        default="192.168.0.2",
        help="IP address of Hololink board",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=("tx", "rx", "dual"),
        help="Mode: 'tx' for transmitter, 'rx' for receiver, 'dual' for both TX and RX",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level to display",
    )

    # UART parameters
    parser.add_argument(
        "--port-number",
        type=int,
        default=0,
        help="UART port number - only one port is supported at the moment [0]",
    )
    parser.add_argument(
        "--baud-rate",
        type=int,
        default=115200,
        choices=(9600, 19200, 38400, 57600, 115200),
        help="UART baud rate [9600,19200,38400,57600,115200]",
    )
    parser.add_argument(
        "--data-bits",
        type=int,
        default=8,
        choices=(5, 6, 7, 8),
        help="UART data bits [5,6,7,8]",
    )
    parser.add_argument(
        "--parity",
        type=int,
        default=0,
        choices=(0, 1, 2),
        help="UART parity [0=None, 1=Odd, 2=Even]",
    )
    parser.add_argument(
        "--stop-bits",
        type=int,
        default=1,
        choices=(1, 15, 2),
        help="UART stop bits [1, 15 (1.5), 2]",
    )
    parser.add_argument(
        "--flow-control",
        type=int,
        default=0,
        choices=(0, 1),
        help="UART flow control [0=None, 1=RTS/CTS]",
    )

    # Create a 512-byte test string with repeating pattern for easy verification
    test_pattern = "0123456789ABCDEF"  # 16 bytes
    default_test_string = test_pattern * 32  # 16 * 32 = 512 bytes

    parser.add_argument(
        "--test-string",
        default=default_test_string,
        help="String to send over UART (default: 512-byte repeating pattern)",
    )
    parser.add_argument(
        "--expected-rx-string",
        default=None,
        help="Expected string to receive (for dual mode, defaults to test-string if not specified)",
    )
    parser.add_argument(
        "--read-max-bytes",
        type=int,
        default=256,
        help="Maximum bytes to read back from UART [256 max]",
    )
    parser.add_argument(
        "--sleep-time",
        type=float,
        default=2,
        help="Time to allow between cycles [seconds]",
    )
    parser.add_argument(
        "--cycle-limit",
        type=int,
        help="Limit the number of cycles for the application; by default this runs forever.",
    )

    args = parser.parse_args()
    hololink_module.logging_level(args.log_level)

    logging.info("=" * 80)
    logging.info(
        f"UART Dual-Board Test (Holoscan Operators) - Mode: {args.mode.upper()}"
    )
    logging.info("=" * 80)
    logging.info(f"Board IP: {args.hololink}")
    if args.mode == "tx":
        logging.info("Mode: TRANSMITTER")
        logging.info(f"TX String: {args.test_string!r} ({len(args.test_string)} bytes)")
    elif args.mode == "rx":
        logging.info("Mode: RECEIVER")
        logging.info(
            f"Expected RX: {args.test_string!r} ({len(args.test_string)} bytes)"
        )
    else:  # dual
        logging.info("Mode: DUAL (TX + RX)")
        logging.info(f"TX String: {args.test_string!r} ({len(args.test_string)} bytes)")
        expected_rx = args.expected_rx_string or args.test_string
        logging.info(f"Expected RX: {expected_rx!r} ({len(expected_rx)} bytes)")
    logging.info("=" * 80)

    logging.info("Initializing...")

    # Get a handle to the Hololink device
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
    hololink_channel = hololink_module.DataChannel(channel_metadata)

    # Set up the application
    expected_rx_string = args.expected_rx_string or args.test_string
    application = HoloscanApplication(
        hololink_channel,
        channel_metadata,
        args.port_number,
        args.baud_rate,
        args.test_string,
        args.read_max_bytes,
        args.sleep_time,
        args.cycle_limit,
        args.mode,
        expected_rx_string,
        args.data_bits,
        args.parity,
        args.stop_bits,
        args.flow_control,
    )
    application.config(args.configuration)

    # Run it.
    hololink = hololink_channel.hololink()
    hololink.start()
    hololink.reset()
    application.run()
    hololink.stop()


if __name__ == "__main__":
    main()
