"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import logging
import os
import socket
import subprocess
import sys
import threading
import time

logger = None

# This address is used by QEMU for the host address.
QEMU_GATEWAY_IP = "10.0.2.2"
DEFAULT_HOLOLINK_PORT = 8400


class Device:
    def __init__(self, id, name, filename):
        self.id = id
        self.name = name
        self.filename = filename
        self.file = None
        self.thread = None
        self.pipe_r, self.pipe_w = os.pipe()


class HsbSpiDaemon:
    def __init__(self, port, devices, drivers, overrides):
        self.devices = devices
        self.drivers = drivers
        self.overrides = overrides

        if port == 0:
            logger.warning("Port of 0 provided -- no communication will occur")
            self.sock = None
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            logger.info(f"Connecting to Hololink, port {port}...")
            while True:
                try:
                    self.sock.connect((QEMU_GATEWAY_IP, port))
                    logger.info("  --> Connected.")
                    break
                except ConnectionRefusedError:
                    time.sleep(1)

        for d in self.devices:
            try:
                d.file = os.open(d.filename, os.O_RDWR)
            except Exception as e:
                logger.error(f"Failed to open file ({e})")
                raise
            logger.debug(f"Opened {d.filename} for device {d.name}")

        for d in self.drivers:
            try:
                logger.info(f"Loading kernel module '{d}'...")
                subprocess.Popen(["modprobe", d])
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to load module '{d}':", e)

    def __del__(self):
        for d in self.devices:
            if d.file:
                os.close(d.file)

    def dispatch(self, data):
        read_data = None
        with self.lock:
            if self.sock:
                self.sock.sendall(data)
                logger.debug(f"    --> sent:     {data.hex(' ')}")
                response = self.sock.recv(300)
                logger.debug(f"    --> received: {response.hex(' ')}")
                assert len(response) >= 2
                read_data_size = int.from_bytes(response[0:2], "big")
                assert read_data_size == len(response) - 2
                if read_data_size > 0:
                    read_data = response[2:]
        return read_data

    def device_task(self, device, client):
        while True:
            # Read the command from the kernel driver
            read_bytes = os.read(device.file, 256)
            send_data = bytearray(read_bytes)

            msg_type = read_bytes[0]
            if msg_type == 0:  # HSB_SPI_MSG_TYPE_SPI
                cs = read_bytes[1]
                cmd = read_bytes[2]
                wr = read_bytes[3]
                rd = read_bytes[4]
                data = read_bytes[5:]
                logger.debug(
                    f"{device.name}: {cs=} {cmd=} {wr=} {rd=} data=[{data.hex(' ')}]"
                )

                # Append the device ID to the CS byte.
                send_data[1] = send_data[1] | (device.id << 4)

                # Dispatch command
                read_data = client.dispatch(send_data)

                # Fill expected read data in case communication is being skipped
                if not read_data and cmd + rd > 0:
                    read_data = bytes(cmd + rd)

                # Apply overrides
                for override in self.overrides:
                    read_data = override.apply(data, read_data)

            elif msg_type == 1:  # HSB_SPI_MSG_TYPE_JESD
                jesd_id = read_bytes[1]
                jesd_state_names = [
                    "JESD204_OP_DEVICE_INIT",
                    "JESD204_OP_LINK_INIT",
                    "JESD204_OP_LINK_SUPPORTED",
                    "JESD204_OP_LINK_PRE_SETUP",
                    "JESD204_OP_CLK_SYNC_STAGE1",
                    "JESD204_OP_CLK_SYNC_STAGE2",
                    "JESD204_OP_CLK_SYNC_STAGE3",
                    "JESD204_OP_LINK_SETUP",
                    "JESD204_OP_OPT_SETUP_STAGE1",
                    "JESD204_OP_OPT_SETUP_STAGE2",
                    "JESD204_OP_OPT_SETUP_STAGE3",
                    "JESD204_OP_OPT_SETUP_STAGE4",
                    "JESD204_OP_OPT_SETUP_STAGE5",
                    "JESD204_OP_CLOCKS_ENABLE",
                    "JESD204_OP_LINK_ENABLE",
                    "JESD204_OP_LINK_RUNNING",
                    "JESD204_OP_OPT_POST_RUNNING_STAGE",
                ]
                logger.debug(
                    f"{device.name}: JESD transition to state {jesd_id}, {jesd_state_names[jesd_id]}"
                )

                # Dispatch command
                read_data = client.dispatch(send_data)

                # Return JESD204_STATE_CHANGE_DONE (1) if communication is being skipped
                if not read_data:
                    read_data = bytes([1])

            else:
                logger.error(f"Invalid message type: {msg_type}")

            # Write the response for the kernel driver
            if read_data:
                logger.debug(f"{device.name}:  --> [{read_data.hex(' ')}]")
                os.write(device.file, read_data)

    def run(self):
        self.lock = threading.Lock()

        for d in self.devices:
            d.thread = threading.Thread(target=self.device_task, args=(d, self))
            d.thread.start()

        for d in self.devices:
            d.thread.join()

        return


# Overrides the initialization read/write test for HMC7044
class OverrideHMC7044Init:
    def __init__(self):
        self.test_value = 0

    def apply(self, data, read_data):
        if data[0] & 0x80 == 0:
            if data[0:2] == bytes.fromhex("00 08"):
                self.test_value = data[2]
        else:
            read_data = bytearray(read_data)
            if data[0:2] == bytes.fromhex("80 08"):
                read_data[0] = self.test_value
        return read_data


# Overrides the initialization read/write test and the chip ID for AD9081
class OverrideAD9081Init:
    def __init__(self):
        self.test_value = bytearray(4)

    def apply(self, data, read_data):
        if data[0] & 0x80 == 0:
            # 8 bit r/w test
            if data[0:2] == bytes.fromhex("00 1C"):
                self.test_value[0] = data[2]
            # 32 bit r/w test
            if data[0:2] == bytes.fromhex("53 00"):
                self.test_value = bytearray(data[2:])
        else:
            read_data = bytearray(read_data)
            # 8 bit r/w test
            if data[0:2] == bytes.fromhex("80 1C"):
                read_data[2] = self.test_value[0]
            # 32 bit r/w test
            elif data[0:2] == bytes.fromhex("D3 00"):
                read_data = read_data[0:2] + self.test_value
            # Board version/revision
            elif data[0:2] == bytes.fromhex("80 04"):
                read_data[2] = 0x81
            elif data[0:2] == bytes.fromhex("80 05"):
                read_data[2] = 0x90
            elif data[0:2] == bytes.fromhex("80 06"):
                read_data[2] = 0x01
            # Boot/clock switch done
            elif data[0:2] == bytes.fromhex("B7 40"):
                read_data[2] = 0xFF
            # Status
            elif data[0:2] == bytes.fromhex("B7 42"):
                read_data[2] = 0xFF
            # JESD Channel interpolation
            elif data[0:2] == bytes.fromhex("81 FF"):
                read_data[2] = 0xFF
            # JESD Bitrate
            elif data[0:2] == bytes.fromhex("84 AC"):
                read_data[2] = 0xFE
            elif data[0:2] == bytes.fromhex("84 AE"):
                read_data[2] = 0xFE
            # Chip decimation
            elif data[0:2] == bytes.fromhex("82 89"):
                read_data[2] = 0x01
            # Oneshot sync
            elif data[0:2] == bytes.fromhex("80 B8"):
                read_data[2] = 0xFF
            # Link status
            elif data[0:2] == bytes.fromhex("85 5E"):
                read_data[2] = 0x60
        return read_data


# Overrides the PLL Lock for AD9081
class OverrideAD9081PLLLock:
    def apply(self, data, read_data):
        if data[0] & 0x80 != 0:
            read_data = bytearray(read_data)
            if data[0:2] == bytes.fromhex("87 22"):
                read_data[2] = 1 << 3
            if data[0:2] == bytes.fromhex("A0 08"):
                read_data[2] = 0x3
        return read_data


def main():
    global logger
    logger = logging.getLogger("HsbSpiDaemon")

    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--port", help="Hololink Server Port", type=int, default=DEFAULT_HOLOLINK_PORT
    )
    parser.add_argument(
        "--devices", help="Device(s) to open", type=int, nargs="+", required=True
    )
    parser.add_argument(
        "--names",
        help="Descriptive device names for logging (e.g. 'ad9081')",
        nargs="+",
    )
    parser.add_argument("--drivers", help="Drivers to load once connected", nargs="+")
    parser.add_argument(
        "--log-level",
        help="Log level (1=Debug, 2=Info, 3=Warning, 4=Error, 5=Critical)",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=2,
    )
    parser.add_argument("--log-file", help="Log output file")
    parser.add_argument(
        "--overrides",
        help="Overrides specific SPI responses for a device",
        nargs="+",
        choices=["hmc7044-init", "ad9081-init", "ad9081-pll-lock"],
        default=[],
    )
    args = parser.parse_args()

    # Create device list
    if args.names and len(args.devices) != len(args.names):
        logger.error("Number of devices and device names doesn't match")
        exit(-1)
    devices = []
    for idx, device in enumerate(args.devices):
        name = args.names[idx] if args.names else f"hsbspi{device}"
        devices.append(Device(id=idx, name=name, filename=f"/dev/hsbspi{device}"))

    # Setup logger
    logger.setLevel(args.log_level * 10)
    log_formatter = logging.Formatter("%(name)s:%(levelname)s: %(message)s")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setFormatter(log_formatter)
    logger.addHandler(stdout_handler)
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

    # Print application info
    logger.info("Starting HSB SPI Daemon")
    logger.info("Devices:")
    for d in devices:
        logger.info(f"  {d.id}: {d.filename}, {d.name}")
    if len(args.drivers):
        logger.info(f"Drivers: {args.drivers}")
    if len(args.overrides):
        logger.info(f"Overrides: {args.overrides}")

    # Set overrides
    overrides = []
    for o in args.overrides:
        if o == "hmc7044-init":
            overrides.append(OverrideHMC7044Init())
        elif o == "ad9081-init":
            overrides.append(OverrideAD9081Init())
        elif o == "ad9081-pll-lock":
            overrides.append(OverrideAD9081PLLLock())

    # Start the daemon
    client = HsbSpiDaemon(args.port, devices, args.drivers, overrides)
    client.run()


if __name__ == "__main__":
    main()
