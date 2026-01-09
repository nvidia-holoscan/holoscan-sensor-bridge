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
import yaml

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


def download_extract(current_file_cfg):
    current_url = current_file_cfg.get("url", None)
    filename = current_file_cfg.get("filename", None)
    expected_md5 = current_file_cfg.get("md5", None)

    if current_url:
        r = requests.get(current_url, timeout=300)
        # parse URL to get last set of / before zip
        zipname = current_url.split("/")[-1]
        with open(zipname, "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(zipname) as zip_ref:
            zip_ref.extractall(".")
    with open(filename, "rb") as file_to_check:
        data = file_to_check.read()
        md5_returned = hashlib.md5(data).hexdigest()
    if md5_returned != expected_md5:
        raise ValueError(
            f"MD5 hash mismatch: expected {expected_md5}, got {md5_returned}"
        )


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


def _spi_flash(spi_con_addr, hololink, cfg_file, channel_metadata):

    yaml_path = os.path.join(os.getcwd(), cfg_file)
    try:
        with open(yaml_path, "r") as file:
            default_config = yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"YAML configuration file not found: {yaml_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Failed to parse YAML file: {e}")
        raise

    try:
        images = default_config["hololink"]["images"]
        if not images:
            raise ValueError("No images defined in YAML manifest")
        fname = images[0]["content"]
        current_file_cfg = default_config["hololink"]["content"][fname]
        filename = current_file_cfg.get("filename", None)
        current_url = current_file_cfg.get("url", None)
        if current_url:
            zipname = current_url.split("/")[-1]
            archive_filename = zipname
        else:
            archive_filename = None
        spi_filename = filename
    except (KeyError, IndexError) as e:
        logging.error(f"Invalid YAML structure: missing required key {e}")
        raise ValueError(f"YAML manifest is missing required fields: {e}")
    download_extract(current_file_cfg)
    hsb_ip_version = channel_metadata["hsb_ip_version"]
    if hsb_ip_version < 0x2506:
        in_spi = hololink_module.get_traditional_spi(
            hololink,
            spi_con_addr,
            chip_select=0,
            cpol=1,
            cpha=1,
            width=1,
            clock_divisor=0x4,
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
    with open(spi_filename, "rb") as f:
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
    _spi_command(in_spi, [ENABLE_RESET])
    _spi_command(in_spi, [RESET])
    _wait_for_spi_ready(in_spi)
    if os.path.exists(spi_filename):
        os.remove(spi_filename)
    else:
        logging.warning(f"SPI file {spi_filename} doesn't exist for cleanup")

    if archive_filename and os.path.exists(archive_filename):
        os.remove(archive_filename)
    else:
        logging.warning(f"Archive file {archive_filename} doesn't exist for cleanup")


def manual_enumeration(args):
    m = {
        "control_port": 8192,
        "hsb_ip_version": 0x2502,
        "peer_ip": args.hololink,
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


def main():
    parser = argparse.ArgumentParser()

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

    parser.add_argument(
        "--flash",
        type=str,
        help="Yaml configuration file to load.",
    )

    args = parser.parse_args()
    hololink_module.logging_level(args.log_level)

    # Get a handle to the Hololink device

    if not args.flash and not args.program:
        raise Exception(
            "Choose option '--flash' to transfer bit file to SPI flash memory"
        )
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
        _spi_flash(SPI_CONN_ADDR, hololink, args.flash, channel_metadata)
        _spi_program(hololink)
        # Wait for FPGA programming to complete and stabilize
        time.sleep(45)
        logging.info("Please power cycle the board to finish programming process")
    hololink.stop()


if __name__ == "__main__":
    main()
