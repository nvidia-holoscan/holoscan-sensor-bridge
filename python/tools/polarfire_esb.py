# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import hashlib
import logging
import os
import time
import zipfile

import requests

import hololink as hololink_module

BLOCK_SIZE = 128
ERASE_SIZE = 64 * 1024
WRITE_ENABLE = 0x06
ENABLE_RESET = 0x66
RESET = 0x99
BYTE_ADDRESS_3 = 0xB7
BLOCK_ERASE = 0xD8
PAGE_PROGRAM = 0x2
START_ADDR = 0xA00000

PFSRV_CMD_OFFSET = 0x4
PFSRV_IAP_PROGRAM_OP = 0x43
IAP_IMG_START_ADDR = 0xA00000
PFSRV_ADDR = 0x11000000
PFSRV_MBX_WRCNT = 0x014
PFSRV_MBX_WADDR = 0x1C
PFSRV_SS_REQ = 0x0C
PFSRV_MBX_WDATA = 0x28

SPI_CONN_ADDR = 0x03000000

filename_2412 = "PF_HL_2412.spi"
fname_2412 = "Holoscan_PF_HL_2412.zip"
expected_md5_2412 = "9be0f5566b6368620392a4ffa0548d7e"
url_2412 = (
    "https://ww1.microchip.com/downloads/aemDocuments/documents/FPGA/SOCDesignFiles/"
    + fname_2412
)

filename_2407 = "PF_HL_2407_bid4.spi"
fname_2407 = "Holoscan_PF_HL_2407.zip"
expected_md5_2407 = "b248817710c09f3461b09dde229b89e3"
url_2407 = (
    "https://ww1.microchip.com/downloads/aemDocuments/documents/FPGA/SOCDesignFiles/"
    + fname_2407
)

filename_2506 = "PF_ESB_HSB2507_v2025p1/PF_ESB_HSB2507_v2025p1.spi"
fname_2506 = "PF_ESB_HSB2507_v2025p1.zip"
expected_md5_2506 = "e27289503290a3a0aa709830a50cf111"
url_2506 = (
    "https://ww1.microchip.com/downloads/aemDocuments/documents/FPGA/SOCDesignFiles/"
    + fname_2506
)


def download_extract(args):
    if args == "2412":
        r = requests.get(url_2412)
        open(fname_2412, "wb").write(r.content)
        with zipfile.ZipFile(fname_2412) as zip_ref:
            zip_ref.extractall(".")
        with open(filename_2412, "rb") as file_to_check:
            data = file_to_check.read()
            md5_returned = hashlib.md5(data).hexdigest()
        if md5_returned != expected_md5_2412:
            raise Exception("md5 Hash mismatch")

    elif args == "2407":
        r = requests.get(url_2407)
        open(fname_2407, "wb").write(r.content)
        with zipfile.ZipFile(fname_2407) as zip_ref:
            zip_ref.extractall(".")
        with open(filename_2407, "rb") as file_to_check:
            data = file_to_check.read()
            md5_returned = hashlib.md5(data).hexdigest()
        if md5_returned != expected_md5_2407:
            raise Exception("md5 Hash mismatch")

    elif args == "2506":
        r = requests.get(url_2506)
        open(fname_2506, "wb").write(r.content)
        with zipfile.ZipFile(fname_2506) as zip_ref:
            zip_ref.extractall(".")
        with open(filename_2506, "rb") as file_to_check:
            data = file_to_check.read()
            md5_returned = hashlib.md5(data).hexdigest()
        if md5_returned != expected_md5_2506:
            raise Exception("md5 Hash mismatch")


def _spi_command(in_spi, command, w_data=[], read_count=0):
    in_spi.spi_transaction(command, w_data, read_count)


def _wait_for_spi_ready(in_spi):
    STATUS = 0x05
    timeout = 3
    now = time.monotonic()
    while True:
        r = in_spi.spi_transaction([STATUS], [], read_byte_count=1)[1:]
        if (r[0] & 1) == 0:
            break
        elif time.monotonic() >= (now + timeout):
            raise Exception("SPI STATUS read fail")


def _spi_program(hololink):
    hololink.write_uint32(PFSRV_ADDR + PFSRV_CMD_OFFSET, PFSRV_IAP_PROGRAM_OP)
    hololink.write_uint32(PFSRV_ADDR + PFSRV_MBX_WRCNT, 0x1)
    hololink.write_uint32(PFSRV_ADDR + PFSRV_MBX_WADDR, 0x0)
    hololink.write_uint32(PFSRV_ADDR + PFSRV_SS_REQ, 0x01)
    hololink.write_uint32(PFSRV_ADDR + PFSRV_MBX_WDATA, IAP_IMG_START_ADDR)


def _spi_flash(spi_con_addr, hololink, fpga_bit_version):
    if fpga_bit_version == "2412":
        lfilename = filename_2412
        lfname = fname_2412
    elif fpga_bit_version == "2407":
        lfilename = filename_2407
        lfname = fname_2407
    elif fpga_bit_version == "2506":
        lfilename = filename_2506
        lfname = fname_2506
    else:
        raise Exception("In correct FPGA bit version")
    download_extract(fpga_bit_version)
    in_spi = hololink_module.get_traditional_spi(
        hololink,
        spi_con_addr,
        chip_select=0,
        cpol=1,
        cpha=1,
        width=1,
        clock_divisor=0x4,
    )
    f = open(lfilename, "rb")
    content = list(f.read())
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
        for addr in range(erase_add, min(con_len, erase_add + ERASE_SIZE), BLOCK_SIZE):
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
            _spi_command(in_spi, command_bytes, content[offset : offset + BLOCK_SIZE])
            print(f"writing to spi: {offset}/{tot_len}", end="\r")
            _wait_for_spi_ready(in_spi)
    _wait_for_spi_ready(in_spi)
    f.close()
    _spi_command(in_spi, [ENABLE_RESET])
    _spi_command(in_spi, [RESET])
    _wait_for_spi_ready(in_spi)
    if os.path.exists(lfilename):
        os.remove(lfilename)
    else:
        print("file" + lfilename + "doesn't exists")

    if os.path.exists(lfname):
        os.remove(lfname)
    else:
        print("file" + lfname + "doesn't exists")
    hololink.stop()


def manual_enumeration(args):
    m = {
        "control_port": 8192,
        "hsb_ip_version": 0x2502,
        "peer_ip": args.hololink,
        "sequence_number_checking": 0,
        "serial_number": "100",
        "fpga_uuid": "ed6a9292-debf-40ac-b603-a24e025309c1",
        "vsync_enable": 0,
        "ptp_enable": 0,
    }
    metadata = hololink_module.Metadata(m)
    hololink_module.DataChannel.use_data_plane_configuration(metadata, 0)
    hololink_module.DataChannel.use_sensor(metadata, 0)
    return metadata


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--program",
        action="store_true",
        help="Program the board with BIT file present in SPI flash",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Don't rely on enumeration data for device connection, Use this option when FPGA runs old bit file versions like 2407",
    )

    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level to display",
    )
    parser.add_argument(
        "--hololink",
        default="192.168.0.2",
        help="IP address of Hololink board",
    )

    subparsers = parser.add_subparsers(help="Subcommands", dest="flash", required=False)

    flash = subparsers.add_parser(
        "flash",
        help="Transfer the BIT file to SPI flash.",
    )

    flash.add_argument(
        "--fpga-bit-version",
        type=str,
        help="FPAG bit file version to be flashed. Currently supported versions 2506, 2412 and 2407",
        default="2506",
        choices=("2412", "2407", "2506"),
        required=True,
    )

    args = parser.parse_args()
    hololink_module.logging_level(args.log_level)

    # Get a handle to the Hololink device

    if not args.flash and not args.program:
        raise Exception(
            "Choose option 'flash' to transfer bit file to SPI flash memory or choose option '--program' to program FPGA with bit file in SPI flash memory"
        )
    elif args.flash and args.program:
        raise Exception("Choose either 'flash' option or '--program' option")
    else:
        logging.info("Initializing.")

    if args.force:
        channel_metadata = manual_enumeration(args)
    else:
        channel_metadata = hololink_module.Enumerator.find_channel(
            channel_ip="192.168.0.2"
        )

    hololink_channel = hololink_module.DataChannel(channel_metadata)
    hololink = hololink_channel.hololink()
    hololink.start()

    if args.flash:
        _spi_flash(SPI_CONN_ADDR, hololink, args.fpga_bit_version)
    elif args.program:
        _spi_program(hololink)


if __name__ == "__main__":
    main()
