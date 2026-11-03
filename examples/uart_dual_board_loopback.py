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

"""Test UART between two cross-wired boards using HSDK5.

Wire board 1 GPIO 10 (TX) to board 2 GPIO 11 (RX), board 2 GPIO 10 to
board 1 GPIO 11, and connect a common ground. Run one process per board.
Use matching UART settings. RTS/CTS flow control requires additional wiring.

Start the receiver before the transmitter for one-way tests. The transmitter is not
synchronised with the receiver and the boards' RX FIFOs hold 256 bytes, so a 512-byte
burst that arrives while nobody is draining loses its second half: bytes 0-255 queue
up, 256-510 are dropped (RX_OVERFLOW), and the UART core parks the last byte in its
holding register until a slot frees. A receiver that starts reading between bursts
then sees "half burst, parked last byte, next burst", which with the repeating default
pattern looks like one burst with byte 255 duplicated at offset 256. A transmitter
started after the receiver is a problem too: while it resets and reconfigures its
board the TX line is not idle, and the receiver gets a 0x00 break frame ahead of the
real burst. The receiver therefore discards whatever is queued, waits for the line to
go quiet, and only accepts a capture whose first bytes match the expected prefix,
resynchronising otherwise. For dual tests,
start both processes within --startup-delay (increase it when using two terminals),
and swap --test-string/--expected-rx-string between the boards. Dual mode writes
and reads one 16-byte chunk at a time, exchanging an ACK byte after each chunk,
so the default 512-byte messages do not require a 512-byte receive FIFO. Both
boards must run this version of the chunk-and-ACK protocol.

UART is configured once by get_uart(). --cycle-limit counts completed transfers;
receive timeouts, mismatches and runtime failures return a nonzero exit status.
"""

import argparse
import logging
import sys
import time
from datetime import timedelta

import holoscan as hs

import hololink as hl


class _UartShared:
    """One live UART handle and completion count for this process's graph."""

    uart = None
    completed_cycles = 0


def _read_exact(uart, size, read_max_bytes, deadline):
    """Read real bytes only; uart_read pads its result to the requested size."""
    received = bytearray()
    while len(received) < size:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"UART receive timeout: got {len(received)}/{size} bytes"
            )
        count = min(read_max_bytes, 256, size - len(received))
        data, bytes_read = uart.uart_read(count)
        if not 0 <= bytes_read <= min(count, len(data)):
            raise RuntimeError(f"Invalid UART read count: {bytes_read}")
        received.extend(data[:bytes_read])
        if not bytes_read:
            time.sleep(0.001)
    return bytes(received)


# A burst has no internal gaps longer than a few milliseconds at any supported baud
# rate, while bursts are --sleep-time (2 s by default) apart, so half a second of
# silence marks a burst boundary.
_SYNC_IDLE_S = 0.5


def _discard_until_idle(uart, read_max_bytes, deadline, idle_s=_SYNC_IDLE_S):
    """Drop queued bytes until the line has been quiet for idle_s, so the next read
    starts at a burst boundary (see the module docstring). Returns the bytes dropped."""
    discarded = 0
    quiet_since = time.monotonic()
    while time.monotonic() - quiet_since < idle_s:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"UART line never went idle for {idle_s}s; discarded {discarded} bytes"
            )
        data, bytes_read = uart.uart_read(min(read_max_bytes, 256))
        if bytes_read:
            discarded += bytes_read
            quiet_since = time.monotonic()
        else:
            time.sleep(0.001)
    if discarded:
        logging.warning(
            "UART RX: discarded %d stale byte(s) queued before the first transfer "
            "(the transmitter was already running)",
            discarded,
        )
    return discarded


# Bytes of the expected message that a capture must begin with to be accepted.
_SYNC_PREFIX = 16


def _capture_first_burst(uart, expected_data, read_max_bytes, deadline):
    """Synchronise to the transmitter, then capture one message (module docstring).

    Discard queued bytes, wait for the line to be quiet, then compare what arrives with
    the first _SYNC_PREFIX expected bytes as it arrives; on the first mismatch (a 0x00
    break frame from a peer that reset after we synced, or a burst we joined late),
    drop it and start over. The real burst follows the glitch after a quiet gap and
    is captured whole."""
    probe = expected_data[:_SYNC_PREFIX]
    while True:
        _discard_until_idle(uart, read_max_bytes, deadline)
        head = bytearray()
        while len(head) < len(probe):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"UART receive timeout: got {len(head)}/{len(expected_data)} bytes"
                )
            count = min(read_max_bytes, 256, len(probe) - len(head))
            data, bytes_read = uart.uart_read(count)
            if not bytes_read:
                time.sleep(0.001)
                continue
            head.extend(data[:bytes_read])
            if not probe.startswith(bytes(head)):
                break
        if bytes(head) == probe:
            rest = _read_exact(
                uart, len(expected_data) - len(probe), read_max_bytes, deadline
            )
            return bytes(head) + rest
        logging.warning(
            "UART RX: leading bytes %r do not match the expected %r; "
            "discarding and resynchronising",
            bytes(head),
            probe,
        )


def _transfer(
    uart, mode, tx_data, expected_data, read_max_bytes, read_timeout, first=False
):
    """One transfer; dual peers drain each other's FIFO after each small write.

    first: in rx mode, resynchronise to a burst boundary before capturing."""
    chunk_size = 16
    deadline = time.monotonic() + read_timeout
    received = bytearray()
    size = max(len(tx_data), len(expected_data)) if mode == "dual" else len(tx_data)
    if mode == "rx":
        if first:
            received.extend(
                _capture_first_burst(uart, expected_data, read_max_bytes, deadline)
            )
        else:
            received.extend(
                _read_exact(uart, len(expected_data), read_max_bytes, deadline)
            )
    else:
        for offset in range(0, size, chunk_size):
            chunk = tx_data[offset : offset + chunk_size]
            if chunk:
                # Explicit list is accepted by the native std::vector<uint8_t> binding.
                uart.uart_write(list(chunk))
            if mode == "dual":
                expected_chunk = expected_data[offset : offset + chunk_size]
                actual = _read_exact(
                    uart, len(expected_chunk), read_max_bytes, deadline
                )
                if actual != expected_chunk:
                    raise RuntimeError(
                        f"UART mismatch at byte {offset}: "
                        f"expected {expected_chunk!r}, received {actual!r}"
                    )
                received.extend(actual)
                # Even an exhausted sender participates: without this barrier,
                # unequal message lengths let the remaining sender overflow RX.
                uart.uart_write([0x06])
                ack = _read_exact(uart, 1, read_max_bytes, deadline)
                if ack != b"\x06":
                    raise RuntimeError(f"UART chunk acknowledgment mismatch: {ack!r}")
    if mode != "tx":
        if received != expected_data:
            raise RuntimeError(
                f"UART mismatch: expected {expected_data!r}, received {bytes(received)!r}"
            )
        logging.info("UART SUCCESS: verified %d bytes", len(received))
    else:
        logging.info("UART TX: wrote %d bytes (no receive verification)", len(tx_data))


class UartTransferOp(hs.Operator):
    """Keep configuration, transfer completion and cycle pacing serial."""

    mode = hs.Param("dual")
    test_string = hs.Param("")
    expected_string = hs.Param("")
    read_max_bytes = hs.Param(256)
    sleep_time = hs.Param(2.0)
    startup_delay = hs.Param(2.0)
    read_timeout = hs.Param(30.0)
    cycle_limit = hs.Param(0)

    def setup(self, spec):
        pass

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=timedelta(milliseconds=10)))

    def start(self):
        self._uart = _UartShared.uart
        self._count = 0
        self._next_cycle = time.monotonic() + self.startup_delay

    def compute(self, context):
        if self.cycle_limit and self._count >= self.cycle_limit:
            context.request_stop()
            return
        if time.monotonic() < self._next_cycle:
            return
        _transfer(
            self._uart,
            self.mode,
            self.test_string.encode("utf-8"),
            self.expected_string.encode("utf-8"),
            self.read_max_bytes,
            self.read_timeout,
            first=(self._count == 0),
        )
        self._count += 1
        _UartShared.completed_cycles = self._count
        logging.info("UART completed cycles: %d", self._count)
        self._next_cycle = time.monotonic() + self.sleep_time
        if self.cycle_limit and self._count >= self.cycle_limit:
            context.request_stop()


def run(
    hololink_ip,
    mode,
    port_number,
    baud_rate,
    data_bits,
    parity,
    stop_bits,
    flow_control,
    test_string,
    expected_rx_string,
    read_max_bytes,
    sleep_time,
    cycle_limit,
    startup_delay=2.0,
    read_timeout=30.0,
):
    metadata = hl.find_channel(hololink_ip)
    channel = hl.DataChannel(metadata)
    hololink_dev = channel.hololink()
    _UartShared.completed_cycles = 0
    try:
        hololink_dev.start()
        hololink_dev.reset()
        # get_uart configures and resets the FIFOs once; never reconfigure mid-transfer.
        _UartShared.uart = hololink_dev.get_uart(
            port_number, baud_rate, data_bits, parity, stop_bits, flow_control
        )
        logging.info(
            "UART configured; waiting %.1f seconds for peer startup", startup_delay
        )
        graph = hs.Graph("uart_dual_board_loopback")
        graph.op(
            UartTransferOp,
            "uart_transfer",
            mode=mode,
            test_string=test_string,
            expected_string=expected_rx_string,
            read_max_bytes=read_max_bytes,
            sleep_time=sleep_time,
            startup_delay=startup_delay,
            read_timeout=read_timeout,
            cycle_limit=cycle_limit or 0,
        )
        clock_key = graph.add_clock(hs.RealtimeClock, "clock")
        graph.set_default_clock(clock_key)
        plan = hs.compile(graph, hs.CompileOptions())
        if not plan.ok:
            print("Graph compilation failed:", file=sys.stderr)
            for diag in plan.diagnostics:
                print(f"  {diag}", file=sys.stderr)
            return 1
        interrupted = False
        with hs.run_async(plan) as session:
            try:
                session.wait()
            except KeyboardInterrupt:
                interrupted = True
                session.request_stop()
                session.wait()
            failed = any(
                diag.severity == hs.diagnostics.DiagnosticSeverity.ERROR
                for diag in session.diagnostics
            )
            if failed:
                for diag in session.diagnostics:
                    if diag.severity == hs.diagnostics.DiagnosticSeverity.ERROR:
                        print(f"  {diag}", file=sys.stderr)
        if failed:
            return 1
        if interrupted:
            return 130
        if cycle_limit and _UartShared.completed_cycles < cycle_limit:
            logging.error("UART stopped before the requested cycle limit")
            return 1
        return 0
    finally:
        try:
            hololink_dev.stop()
        finally:
            _UartShared.uart = None


def main():
    parser = argparse.ArgumentParser(
        description="Dual-board UART test using HSDK5 operators"
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

    # Create a 512-byte test string with repeating pattern for easy verification.
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
        help="Expected string to receive (rx and dual modes; defaults to test-string if not specified)",
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
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=2.0,
        help="Seconds before the first transfer; start both dual peers within this interval",
    )
    parser.add_argument(
        "--read-timeout",
        type=float,
        default=30.0,
        help="Receive deadline per transfer [seconds]",
    )
    args = parser.parse_args()
    if args.port_number != 0:
        parser.error("Only UART port 0 is supported")
    if not 1 <= args.read_max_bytes <= 256:
        parser.error("--read-max-bytes must be between 1 and 256")
    if args.cycle_limit is not None and args.cycle_limit <= 0:
        parser.error("--cycle-limit must be positive")
    if args.sleep_time < 0 or args.startup_delay < 0 or args.read_timeout <= 0:
        parser.error("Delays must be nonnegative and --read-timeout must be positive")
    expected_rx = (
        args.test_string if args.expected_rx_string is None else args.expected_rx_string
    )
    for message in (args.test_string, expected_rx):
        if not message:
            parser.error("Test and expected strings must be nonempty")
        if any(byte >= (1 << args.data_bits) for byte in message.encode("utf-8")):
            parser.error("Message bytes do not fit the selected --data-bits")

    logging.basicConfig(level=args.log_level)

    logging.info("=" * 80)
    logging.info(f"UART Dual-Board Test (HSDK5 Operators) - Mode: {args.mode.upper()}")
    logging.info("=" * 80)
    logging.info(f"Board IP: {args.hololink}")
    if args.mode == "tx":
        logging.info("Mode: TRANSMITTER")
        logging.info(f"TX String: {args.test_string!r} ({len(args.test_string)} bytes)")
    elif args.mode == "rx":
        logging.info("Mode: RECEIVER")
        logging.info(f"Expected RX: {expected_rx!r} ({len(expected_rx)} bytes)")
    else:  # dual
        logging.info("Mode: DUAL (TX + RX)")
        logging.info(f"TX String: {args.test_string!r} ({len(args.test_string)} bytes)")
        logging.info(f"Expected RX: {expected_rx!r} ({len(expected_rx)} bytes)")
    logging.info("=" * 80)

    sys.exit(
        run(
            args.hololink,
            args.mode,
            args.port_number,
            args.baud_rate,
            args.data_bits,
            args.parity,
            args.stop_bits,
            args.flow_control,
            args.test_string,
            expected_rx,
            args.read_max_bytes,
            args.sleep_time,
            args.cycle_limit,
            args.startup_delay,
            args.read_timeout,
        )
    )


if __name__ == "__main__":
    main()
