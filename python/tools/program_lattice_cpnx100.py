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

import argparse
import hashlib
import logging
import pydoc

import requests
import yaml

import hololink as hololink_module


def hexes(values):
    return " ".join(["%02X" % x for x in values])


def reverse(b):
    """Reverse the bits within a byte.  For example,
    if bit 0 is on in the input, then bit 7 will be
    on in the output.
    """
    r = (
        ((b & 0x01) << 7)
        | ((b & 0x02) << 5)
        | ((b & 0x04) << 3)
        | ((b & 0x08) << 1)
        | ((b & 0x10) >> 1)
        | ((b & 0x20) >> 3)
        | ((b & 0x40) >> 5)
        | ((b & 0x80) >> 7)
    )
    return r


reverse_map = bytearray([reverse(b) for b in range(256)])


# Programming
class WindbondW25q128jw:
    # SPI commands
    PAGE_PROGRAM = 0x2
    READ = 0x3
    STATUS = 0x5
    WRITE_ENABLE = 0x6
    WRITE_STATUS_2 = 0x31
    ENABLE_RESET = 0x66
    RESET = 0x99
    JEDEC_ID = 0x9F
    BLOCK_ERASE = 0xD8
    #
    WINBOND = 0xEF
    W25Q128JW_IQ = 0x6018
    #
    QE = 0x2
    #
    BLOCK_SIZE = 128  # bytes
    ERASE_SIZE = 64 * 1024  # bytes

    #
    def __init__(self, context, hololink, spi_address):
        self._context = context
        self._hololink = hololink
        self._hsb_ip_version = hololink.get_hsb_ip_version()
        self._datecode = hololink.get_fpga_date()
        self._spi_address = spi_address

    def _get_spi(self, chip_select, cpol, cpha, width, clock_divisor):
        traditional_spi = (self._hsb_ip_version < 0x2503) or (
            (self._hsb_ip_version == 0x2503) and (self._datecode < 0xE57E47B7)
        )
        if traditional_spi:
            CLNX_SPI_CTRL = 0x03000000
            CPNX_SPI_CTRL = 0x03000200
            if self._spi_address == hololink_module.CLNX_SPI_BUS:
                spi_address = CLNX_SPI_CTRL
            elif self._spi_address == hololink_module.CPNX_SPI_BUS:
                spi_address = CPNX_SPI_CTRL
            else:
                raise Exception("Unexpected {spi_address=:x}")
            return hololink_module.get_traditional_spi(
                self._hololink,
                spi_address,
                chip_select,
                clock_divisor,
                cpol,
                cpha,
                width,
            )
        else:
            return self._hololink.get_spi(
                self._spi_address,
                chip_select,
                clock_divisor,
                cpol,
                cpha,
                width,
            )

    def _wait_for_spi_ready(self):
        while True:
            r = self._spi_command([self.STATUS], read_byte_count=1)
            if (r[0] & 1) == 0:
                return
            logging.debug(
                "%s: BUSY, got r=%s"
                % (self._context, " ".join(["%02X" % x for x in r]))
            )

    def _check_id(self):
        manufacturer_device_id = self._spi_command([self.JEDEC_ID], read_byte_count=3)
        logging.info(
            "%s: manufacturer_device_id=%s"
            % (
                self._context,
                hexes(manufacturer_device_id),
            )
        )
        assert manufacturer_device_id == bytearray(
            [
                self.WINBOND,
                self.W25Q128JW_IQ >> 8,
                self.W25Q128JW_IQ & 0xFF,
            ]
        )

    def _status(self, message):
        logging.info(f"{self._context}: {message}")

    def program(self, content):
        self._check_id()
        content_size = len(content)
        for erase_address in range(0, content_size, self.ERASE_SIZE):
            logging.debug(
                "%s: erase address=0x%X"
                % (
                    self._context,
                    erase_address,
                )
            )
            self._wait_for_spi_ready()
            self._spi_command([self.WRITE_ENABLE])
            page_erase = [
                self.BLOCK_ERASE,
                (erase_address >> 16) & 0xFF,
                (erase_address >> 8) & 0xFF,
                (erase_address >> 0) & 0xFF,
            ]
            self._spi_command(page_erase)
            for address in range(
                erase_address,
                min(content_size, erase_address + self.ERASE_SIZE),
                self.BLOCK_SIZE,
            ):
                # Provide some status.
                if (address & 0xFFFF) == 0:
                    self._status("address=0x%X" % (address,))
                # Write this page.
                self._wait_for_spi_ready()
                self._spi_command([self.WRITE_ENABLE])
                command_bytes = [
                    self.PAGE_PROGRAM,
                    (address >> 16) & 0xFF,
                    (address >> 8) & 0xFF,
                    (address >> 0) & 0xFF,
                ]
                self._spi_command(
                    command_bytes, content[address : address + self.BLOCK_SIZE]
                )
        self._wait_for_spi_ready()
        self._spi_command([self.ENABLE_RESET])
        self._spi_command([self.RESET])
        self._wait_for_spi_ready()

    def verify(self, content):
        self._check_id()
        ok = True
        for address in range(0, len(content), self.BLOCK_SIZE):
            # Provide some status.
            if (address & 0xFFFF) == 0:
                self._status("verify address=0x%X" % (address,))
            # original_content, on the last page, will be shorter than BLOCK_SIZE
            original_content = content[address : address + self.BLOCK_SIZE]
            # Fetch this page from flash
            self._wait_for_spi_ready()
            command_bytes = [
                self.READ,
                (address >> 16) & 0xFF,
                (address >> 8) & 0xFF,
                (address >> 0) & 0xFF,
            ]
            flash_content = self._spi_command(
                command_bytes, read_byte_count=len(original_content)
            )
            # Check it
            if flash_content != original_content:
                logging.info(
                    "%s: verify failed, address=0x%X"
                    % (
                        self._context,
                        address,
                    )
                )
                ok = False
        return ok


class ClnxFlash(WindbondW25q128jw):
    # CLNX registers
    CLNX_WRITE = 0x1
    FLASH_FORWARD_EN_ADDRESS = 6

    #
    def __init__(self, context, hololink, spi_controller_address):
        super().__init__(context, hololink, spi_controller_address)
        self._slow_spi = self._get_spi(
            chip_select=0,
            cpol=0,
            cpha=1,
            width=1,
            clock_divisor=0xF,
        )
        self._fast_spi = self._get_spi(
            chip_select=0,
            cpol=1,
            cpha=1,
            width=1,
            clock_divisor=0x4,
        )

    def _spi_command(self, command_bytes, write_bytes=[], read_byte_count=0):
        # enable_clnx_bridge for 1 transaction
        transactions = 1
        spi_flash_forward_value = 0x1 | (transactions << 4)
        request = [
            self.CLNX_WRITE,
            self.FLASH_FORWARD_EN_ADDRESS,
            spi_flash_forward_value,
        ]
        self._slow_spi.spi_transaction(bytearray(), bytearray(request), 0)
        # execute its spi flash command
        return self._fast_spi.spi_transaction(
            bytearray(command_bytes), bytearray(write_bytes), read_byte_count
        )[len(command_bytes) :]


class CpnxFlash(WindbondW25q128jw):
    #
    def __init__(self, context, hololink, spi_controller_address):
        super().__init__(context, hololink, spi_controller_address)
        self._spi = self._get_spi(
            chip_select=0,
            cpol=1,
            cpha=1,
            width=1,
            clock_divisor=0,
        )

    def _spi_command(self, command_bytes, write_bytes=[], read_byte_count=0):
        return self._spi.spi_transaction(
            bytearray(command_bytes), bytearray(write_bytes), read_byte_count
        )[len(command_bytes) :]


class SensorBridgeStrategy:
    def __init__(self, programmer):
        self._programmer = programmer
        self._args = programmer._args
        self._manifest = programmer._manifest

    def hololink(self, channel_metadata):
        fpga_uuid = channel_metadata["fpga_uuid"]
        if not self.check_fpga_uuid(fpga_uuid):
            return None
        hololink_channel = hololink_module.DataChannel(channel_metadata)
        hololink = hololink_channel.hololink()
        hololink.start()
        return hololink

    def check_fpga_uuid(self, fpga_uuid):
        raise Exception('Unexpected call to abstract "check_fpga_uuid"')

    def program_and_verify_images(self, hololink):
        raise Exception('Unexpected call to abstract "program_and_verify_images"')

    def power_cycle(self, args, hololink):
        print("You must now physically power cycle the sensor bridge device.")
        if args.skip_power_cycle:
            return
        input("Press <Enter> to continue: ")


class SensorBridge10Strategy(SensorBridgeStrategy):
    def __init__(self, programmer):
        super().__init__(programmer)

    def check_fpga_uuid(self, fpga_uuid):
        return fpga_uuid in self._manifest["fpga_uuid"]

    def program_and_verify_images(self, hololink):
        content = self._programmer._content
        if "clnx" in content:
            self.program_clnx(hololink, hololink_module.CLNX_SPI_BUS, content["clnx"])
            self.verify_clnx(hololink, hololink_module.CLNX_SPI_BUS, content["clnx"])
        if "cpnx" in content:
            self.program_cpnx(hololink, hololink_module.CPNX_SPI_BUS, content["cpnx"])
            self.verify_cpnx(hololink, hololink_module.CPNX_SPI_BUS, content["cpnx"])

    def program_clnx(self, hololink, spi_controller_address, content):
        if self._args.skip_program_clnx:
            logging.info("Skipping programming CLNX per command-line instructions.")
            return
        clnx_flash = ClnxFlash("CLNX", hololink, spi_controller_address)
        clnx_flash.program(content)

    def verify_clnx(self, hololink, spi_controller_address, content):
        if self._args.skip_verify_clnx:
            logging.info("Skipping verification CLNX per command-line instructions.")
            return
        clnx_flash = ClnxFlash("CLNX", hololink, spi_controller_address)
        clnx_flash.verify(content)

    def program_cpnx(self, hololink, spi_controller_address, content):
        if self._args.skip_program_cpnx:
            logging.info("Skipping programming CPNX per command-line instructions.")
            return
        cpnx_flash = CpnxFlash("CPNX", hololink, spi_controller_address)
        cpnx_flash.program(content)

    def verify_cpnx(self, hololink, spi_controller_address, content):
        if self._args.skip_verify_cpnx:
            logging.info("Skipping verification CPNX per command-line instructions.")
            return
        cpnx_flash = CpnxFlash("CPNX", hololink, spi_controller_address)
        cpnx_flash.verify(content)


strategies = {
    "sensor_bridge_10": SensorBridge10Strategy,
}
default_strategy = "sensor_bridge_10"  # if not listed in the manifest file


class Programmer:
    def __init__(self, args, manifest_filename):
        self._args = args
        self._manifest_filename = manifest_filename
        self._skip_eula = False

    def fetch_manifest(self, section):
        with open(self._manifest_filename, "rt") as f:
            manifest = yaml.safe_load(f)
        self._manifest = manifest[section]
        strategy_name = self._manifest.get("strategy", default_strategy)
        strategy_constructor = strategies.get(strategy_name)
        if strategy_constructor is None:
            raise Exception('Unsupported strategy "{strategy_name}" specified.')
        self._strategy = strategy_constructor(self)
        if "licenses" not in self._manifest:
            # Note that removing the "licenses" section from the manifest file,
            # in order to achieve this condition, constitutes agreement with it.
            self._skip_eula = True

    def fetch_content(self, content_name):
        content_metadata = self._manifest["content"].get(content_name)
        if content_metadata is None:
            raise Exception(f'No content "{content_name}" found.')
        expected_md5 = content_metadata["md5"]
        expected_size = content_metadata["size"]
        if "url" in content_metadata:
            url = content_metadata["url"]
            request = requests.get(
                url,
                headers={
                    "Content-Type": "binary/octet-stream",
                },
            )
            if request.status_code != requests.codes.ok:
                raise Exception(
                    f'Unable to fetch "{url}"; status={request.status_code}'
                )
            content = request.content
        elif "filename" in content_metadata:
            filename = content_metadata["filename"]
            with open(filename, "rb") as f:
                content = f.read()
        else:
            raise Exception(
                f"No instructions for where to find {content_name} are provided."
            )
        actual_size = len(content)
        logging.debug(f"{expected_size=} {actual_size=}")
        if actual_size != expected_size:
            raise Exception(
                "Content name=%s expected-size=%s actual-size=%s; aborted."
                % (content_name, expected_size, actual_size)
            )
        md5 = hashlib.md5(content)
        actual_md5 = md5.hexdigest()
        logging.debug(f"{expected_md5=} {actual_md5=}")
        if actual_md5 != expected_md5:
            raise Exception(
                "Content name=%s expected-md5=%s actual-md5=%s; aborted."
                % (content_name, expected_md5, actual_md5)
            )
        # We've passed initial checks.
        return content

    def check_eula(self, args):
        if self._skip_eula:
            logging.trace("Accepting EULA is not necessary.")
            return
        if args.accept_eula:
            logging.info("All EULAs are accepted via command-line switch.")
            return
        licenses = self._manifest["licenses"]
        print("You must accept EULA terms in order to continue.")
        assert len(licenses) > 0
        print("For each document, press <Space> to see the next page;")
        print("At the end of the document, enter <Q> to continue.")
        input("To continue, press <Enter>: ")
        for license in licenses:
            content = self.fetch_content(license).decode()
            pydoc.pager(content)
            answer = input(
                "Press 'y' or 'Y' to accept this end user license agreement: "
            )
            if not answer.upper().startswith("Y"):
                # Note that altering this code path to avoid this check
                # constitutes acceptance of any EULA present in the archive.
                raise Exception(
                    "Execution of this script requires an agreement with license terms."
                )

    def check_images(self):
        images = self._manifest["images"]
        self._content = {}
        for image_metadata in images:
            context, content_name = image_metadata["context"], image_metadata["content"]
            logging.info(f"{context=} {content_name=}")
            content = self.fetch_content(content_name)
            self._content[context] = content

    def program_and_verify_images(self, hololink):
        self._strategy.program_and_verify_images(hololink)

    def power_cycle(self, args, hololink):
        self._strategy.power_cycle(args, hololink)

    def hololink(self, channel_metadata):
        logging.debug(f"{channel_metadata=}")
        r = self._strategy.hololink(channel_metadata)
        if r is None:
            ip = channel_metadata["peer_ip"]
            fpga_uuid = channel_metadata["fpga_uuid"]
            raise Exception(
                f"Sensor bridge {ip=} ({fpga_uuid}) isn't supported by this manifest file."
            )
        return r


def manual_enumeration(args):
    m = {
        "control_port": 8192,
        "hsb_ip_version": 0x2502,
        "peer_ip": args.hololink,
        "sequence_number_checking": 0,
        "serial_number": "100",
        "fpga_uuid": "889b7ce3-65a5-4247-8b05-4ff1904c3359",
    }
    metadata = hololink_module.Metadata(m)
    hololink_module.DataChannel.use_data_plane_configuration(metadata, 0)
    hololink_module.DataChannel.use_sensor(metadata, 0)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hololink",
        default="192.168.0.2",
        help="IP address of Hololink board",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Don't rely on enumeration data for device connection.",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level to display",
    )
    parser.add_argument(
        "manifest",
        help="Filename of the firmware manifest file.",
    )
    parser.add_argument(
        "--archive",
        help="Use a local zip archive instead of downloading it.",
    )
    parser.add_argument(
        "--skip-program-clnx",
        action="store_true",
        help="Skip program_clnx",
    )
    parser.add_argument(
        "--skip-verify-clnx",
        action="store_true",
        help="Skip verify_clnx",
    )
    parser.add_argument(
        "--skip-program-cpnx",
        action="store_true",
        help="Skip program_cpnx",
    )
    parser.add_argument(
        "--skip-verify-cpnx",
        action="store_true",
        help="Skip verify_cpnx",
    )
    parser.add_argument(
        "--accept-eula",
        action="store_true",
        help="Provide non-interactive EULA acceptance",
    )
    parser.add_argument(
        "--skip-power-cycle",
        action="store_true",
        help="Don't wait for confirmation of power cycle.",
    )
    parser.add_argument(
        "--skip-program-stratix",
        action="store_true",
        help="Skip program_stratix",
    )
    parser.add_argument(
        "--skip-verify-stratix",
        action="store_true",
        help="Skip verify_stratix",
    )

    #
    args = parser.parse_args()
    hololink_module.logging_level(args.log_level)

    logging.info("manifest=%s" % (args.manifest,))
    programmer = Programmer(args, args.manifest)
    programmer.fetch_manifest("hololink")
    programmer.check_eula(args)
    programmer.check_images()
    if args.force:
        channel_metadata = manual_enumeration(args)
    else:
        channel_metadata = hololink_module.Enumerator.find_channel(
            channel_ip=args.hololink
        )
    hololink = programmer.hololink(channel_metadata)
    programmer.program_and_verify_images(hololink)
    programmer.power_cycle(args, hololink)


if __name__ == "__main__":
    main()
