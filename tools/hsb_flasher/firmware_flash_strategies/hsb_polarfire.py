#!/usr/bin/env python3
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
HSB PolarFire ESB Flash Strategy Module

This module contains flasher classes for the Microchip PolarFire HSB ESB
variant. Each class handles exactly one version. The base class carries the
shared fpga_uuid. `get_flasher()` returns the appropriate class instance for
a given (fpga_uuid, version) pair.

The flash itself is done in pure Python by talking to the FPGA's SPI flash via
`hololink.get_spi()` and then triggering PolarFire's In-Application
Programming (IAP) load via the System Services mailbox. This mirrors the logic
in python/tools/polarfire_esb.py but expects a pre-extracted .spi file on
disk (hsb_flasher handles manifest lookup, download, and MD5 verification).
"""

import time
from abc import ABC, abstractmethod
from typing import Optional

# SPI flash command set
BLOCK_SIZE = 128
ERASE_SIZE = 64 * 1024
WRITE_ENABLE = 0x06
ENABLE_RESET = 0x66
RESET = 0x99
BYTE_ADDRESS_3 = 0xB7
BLOCK_ERASE = 0xD8
PAGE_PROGRAM = 0x02
STATUS = 0x05
START_ADDR = 0xA00000

# PolarFire System Services mailbox registers for IAP program trigger
PFSRV_ADDR = 0x11000000
PFSRV_CMD_OFFSET = 0x04
PFSRV_SS_REQ = 0x0C
PFSRV_MBX_WRCNT = 0x14
PFSRV_MBX_WADDR = 0x1C
PFSRV_MBX_WDATA = 0x28
PFSRV_IAP_PROGRAM_OP = 0x43
IAP_IMG_START_ADDR = 0xA00000

SPI_READY_TIMEOUT_S = 3.0

# Traditional SPI (used when hsb_ip_version is in [0x2407, 0x2506]) comes from
# the traditional_peripherals_py pybind11 module that sits next to this script in
# the hsb_flasher output directory. Single source of truth: traditional_spi.cpp.


def _spi_command(in_spi, command, w_data=None, read_count=0):
    if w_data is None:
        w_data = []
    return in_spi.spi_transaction(command, w_data, read_count)


def _wait_for_spi_ready(in_spi):
    deadline = time.monotonic() + SPI_READY_TIMEOUT_S
    while True:
        r = in_spi.spi_transaction([STATUS], [], read_byte_count=1)[1:]
        if (r[0] & 1) == 0:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("SPI STATUS read timed out waiting for flash ready")


def _flash_polarfire_esb(ip_address: str, spi_path: str, traditional: bool) -> bool:
    # Lazy import so `supports()` / module load works in environments where
    # the hololink package isn't installed (e.g. lint / unit smoke tests).
    import hololink as hololink_module

    if traditional:

        def manual_enumeration(ip_address):
            m = {
                "control_port": 8192,
                "hsb_ip_version": 0x2502,
                "peer_ip": ip_address,
                "sequence_number_checking": 0,
                "serial_number": "100",
                "fpga_uuid": "ed6a9292-debf-40ac-b603-a24e025309c1",
                "ptp_enable": 0,
                "block_enable": 0,
            }
            metadata = hololink_module.Metadata(m)
            hololink_module.DataChannel.use_data_plane_configuration(metadata, 0)
            hololink_module.DataChannel.use_sensor(metadata, 0)
            return metadata

        channel_metadata = manual_enumeration(ip_address)
    else:
        channel_metadata = hololink_module.Enumerator.find_channel(
            channel_ip=ip_address
        )
    hololink_channel = hololink_module.DataChannel(channel_metadata)
    hololink = hololink_channel.hololink()
    hololink.start()
    try:
        with open(spi_path, "rb") as f:
            content = list(f.read())
        if traditional:
            import traditional_peripherals_py

            in_spi = traditional_peripherals_py.get_traditional_spi(
                peer_ip=channel_metadata["peer_ip"],
                control_port=channel_metadata["control_port"],
                serial_number=channel_metadata["serial_number"],
                spi_address=0x03000000,
                chip_select=0,
                clock_divisor=0x4,
                cpol=1,
                cpha=1,
                width=1,
            )
        else:
            in_spi = hololink.get_spi(
                bus_number=0,
                chip_select=0,
                cpol=1,
                cpha=1,
                width=1,
                prescaler=0x4,
            )

        con_len = len(content) + START_ADDR
        tot_len = len(content)
        _spi_command(in_spi, [0x01, 0])
        for erase_add in range(START_ADDR, con_len, ERASE_SIZE):
            _wait_for_spi_ready(in_spi)
            _spi_command(in_spi, [WRITE_ENABLE])
            _spi_command(in_spi, [BYTE_ADDRESS_3])
            page_erase = [
                BLOCK_ERASE,
                (erase_add >> 24) & 0xFF,
                (erase_add >> 16) & 0xFF,
                (erase_add >> 8) & 0xFF,
                (erase_add >> 0) & 0xFF,
            ]
            _spi_command(in_spi, page_erase)
            _wait_for_spi_ready(in_spi)
            for addr in range(
                erase_add, min(con_len, erase_add + ERASE_SIZE), BLOCK_SIZE
            ):
                _wait_for_spi_ready(in_spi)
                _spi_command(in_spi, [WRITE_ENABLE])
                _spi_command(in_spi, [BYTE_ADDRESS_3])
                command_bytes = [
                    PAGE_PROGRAM,
                    (addr >> 24) & 0xFF,
                    (addr >> 16) & 0xFF,
                    (addr >> 8) & 0xFF,
                    (addr >> 0) & 0xFF,
                ]
                offset = addr - START_ADDR
                _spi_command(
                    in_spi, command_bytes, content[offset : offset + BLOCK_SIZE]
                )
                print(f"writing to spi: {offset}/{tot_len}", end="\r")
                _wait_for_spi_ready(in_spi)
        _wait_for_spi_ready(in_spi)
        _spi_command(in_spi, [ENABLE_RESET])
        _spi_command(in_spi, [RESET])
        _wait_for_spi_ready(in_spi)

        hololink.write_uint32(PFSRV_ADDR + PFSRV_CMD_OFFSET, PFSRV_IAP_PROGRAM_OP)
        hololink.write_uint32(PFSRV_ADDR + PFSRV_MBX_WRCNT, 0x1)
        hololink.write_uint32(PFSRV_ADDR + PFSRV_MBX_WADDR, 0x0)
        hololink.write_uint32(PFSRV_ADDR + PFSRV_SS_REQ, 0x01)
        hololink.write_uint32(PFSRV_ADDR + PFSRV_MBX_WDATA, IAP_IMG_START_ADDR)

        print("Please power cycle the board to finish programming process")
        return True
    finally:
        hololink.stop()


class HSBPolarFireEsbFlasherBase(ABC):
    """Base class for HSB PolarFire ESB flashers."""

    FPGA_UUID = "ed6a9292-debf-40ac-b603-a24e025309c1"

    VERSION: int

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstractmethods__", None) and not isinstance(
            cls.__dict__.get("VERSION"), int
        ):
            raise TypeError(f"{cls.__name__} must define a VERSION class attribute")

    def __init__(self, ip_address: str, mac_address: str):
        self.ip_address = ip_address
        self.mac_address = mac_address

    @classmethod
    def supports(cls, fpga_uuid: str, version: int) -> bool:
        if 0x2407 <= version <= 0x2506:
            version = 0x2412
        elif 0x2507 <= version:
            version = 0x2510
        return fpga_uuid == cls.FPGA_UUID and version == cls.VERSION

    @abstractmethod
    def flash(self, cpnx_path: str) -> bool:
        pass


# ===============================================
# UDP Flasher Implementation
# ===============================================


class HSBPolarFireEsbFlasherTraditional(HSBPolarFireEsbFlasherBase):
    """Flasher for HSB PolarFire ESB traditional versions."""

    VERSION = 0x2412

    def flash(self, cpnx_path: str) -> bool:
        return _flash_polarfire_esb(self.ip_address, cpnx_path, True)


# ===============================================
# Base Flasher Implementations
# ===============================================


class HSBPolarFireEsbFlasherCurrent(HSBPolarFireEsbFlasherBase):
    """Flasher for HSB PolarFire ESB current versions."""

    VERSION = 0x2510

    def flash(self, cpnx_path: str) -> bool:
        return _flash_polarfire_esb(self.ip_address, cpnx_path, False)


FLASH_STRATEGIES = [
    HSBPolarFireEsbFlasherTraditional,
    HSBPolarFireEsbFlasherCurrent,
]


def get_flasher(
    fpga_uuid: str, version: int, ip_address: str, mac_address: str
) -> Optional[HSBPolarFireEsbFlasherBase]:
    for flasher_cls in FLASH_STRATEGIES:
        if flasher_cls.supports(fpga_uuid, version):
            return flasher_cls(ip_address=ip_address, mac_address=mac_address)
    return None


def supports(fpga_uuid: str, version: int) -> bool:
    return any(cls.supports(fpga_uuid, version) for cls in FLASH_STRATEGIES)


def do_flash(
    fpga_uuid: str,
    version: int,
    mac_address: str,
    ip_address: str,
    clnx_path: str,
    cpnx_path: str,
) -> bool:
    flasher = get_flasher(fpga_uuid, version, ip_address, mac_address)
    if flasher is None:
        print(
            f"[hsb_polarfire_esb] No flasher for uuid={fpga_uuid} "
            f"version=0x{version:04x}"
        )
        return False

    print(f"[hsb_polarfire_esb] Using {flasher.__class__.__name__}")
    print(f"  FPGA UUID: {fpga_uuid}")
    print(f"  Version: 0x{version:04x}")
    print(f"  MAC: {mac_address}")
    print(f"  IP: {ip_address}")
    print(f"  SPI: {cpnx_path}")

    try:
        success = flasher.flash(cpnx_path)
    except (FileNotFoundError, OSError, ValueError, RuntimeError, ImportError) as e:
        print(f"[hsb_polarfire_esb] ERROR: {e}")
        return False
    else:
        if success:
            print("[hsb_polarfire_esb] Flash completed successfully")
        else:
            print("[hsb_polarfire_esb] Flash failed")
        return success
