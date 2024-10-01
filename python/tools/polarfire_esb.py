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

filename = "PF_HL_2407_bid4.spi"
fname = "Holoscan_PF_HL_2407.zip"
expected_md5 = "b248817710c09f3461b09dde229b89e3"
url = (
    "https://ww1.microchip.com/downloads/aemDocuments/documents/FPGA/SOCDesignFiles/"
    + fname
)


def download_extract():
    r = requests.get(url)
    open(fname, "wb").write(r.content)

    with zipfile.ZipFile(fname) as zip_ref:
        zip_ref.extractall(".")

    with open(filename, "rb") as file_to_check:
        data = file_to_check.read()
        md5_returned = hashlib.md5(data).hexdigest()
    if md5_returned != expected_md5:
        raise Exception("md5 Hash mismatch")


def _spi_command(in_spi, command, w_data=[], read_count=0):
    in_spi.spi_transaction(command, w_data, read_count)


def _wait_for_spi_ready(in_spi):
    STATUS = 0x05
    timeout = 3
    now = time.monotonic()
    while True:
        r = in_spi.spi_transaction([STATUS], [], read_byte_count=1)
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


def _spi_flash(spi_con_addr, hololink):
    download_extract()
    in_spi = hololink.get_spi(
        spi_con_addr,
        chip_select=0,
        cpol=1,
        cpha=1,
        width=1,
        clock_divisor=0x4,
    )
    f = open(filename, "rb")
    content = list(f.read())
    con_len = len(content) + START_ADDR
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
    _wait_for_spi_ready(in_spi)
    f.close()
    _spi_command(in_spi, [ENABLE_RESET])
    _spi_command(in_spi, [RESET])
    _wait_for_spi_ready(in_spi)
    if os.path.exists(filename):
        os.remove(filename)
    else:
        print("file" + filename + "doesn't exists")

    if os.path.exists(fname):
        os.remove(fname)
    else:
        print("file" + fname + "doesn't exists")
    hololink.stop()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--program",
        action="store_true",
        help="Program the board with BIT file present in SPI flash",
    )

    parser.add_argument(
        "--flash",
        action="store_true",
        help="Transfer the BIT file to SPI flash",
    )

    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level to display",
    )

    args = parser.parse_args()
    hololink_module.logging_level(args.log_level)
    logging.info("Initializing.")
    # Get a handle to the Hololink device
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip="192.168.0.2")
    hololink_channel = hololink_module.DataChannel(channel_metadata)
    hololink = hololink_channel.hololink()
    hololink.start()
    hololink.reset()

    if args.flash:
        spi_con_addr = hololink_module.CLNX_SPI_CTRL
        _spi_flash(spi_con_addr, hololink)
    elif args.program:
        _spi_program(hololink)
    else:
        print(
            'choose option "--flash" to transfer bit file to SPI flash memory or choose option "--program" to program FPGA with bit file in SPI flash memory '
        )


if __name__ == "__main__":
    main()
