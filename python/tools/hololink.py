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
import pydoc
import tempfile
import zipfile

import requests
import yaml

import hololink as hololink_module


def hexes(values):
    return " ".join(["%02X" % x for x in values])


def _fpga_version(args, hololink):
    fpga_version = hololink.get_fpga_version()
    print(hex(fpga_version))


def _read_uint32(args, hololink):
    address = args.address
    value = hololink.read_uint32(address)
    logging.info(f"{address=:#x} {value=:#x}")
    print(hex(value))


def _write_uint32(args, hololink):
    address = args.address
    value = args.value
    hololink.write_uint32(address, value)


def _reset(args, hololink):
    hololink.reset()


def _set_ip(args):
    mac_id_and_ip = args.mac_id_and_ip
    # given a list [0,1,2,3,...] this produces a list [(0,1), (2,3), ...]
    settings = zip(mac_id_and_ip[0::2], mac_id_and_ip[1::2])
    ips_by_mac = {k.upper(): v for k, v in settings}
    #
    enumerator = hololink_module.HololinkEnumerator()
    timeout_s = None
    if args.one_time:
        timeout_s = 45
    else:
        logging.info(
            "Running in daemon mode; run with '--one-time' to exit after configuration."
        )
    enumerator.set_ips(ips_by_mac, timeout_s=timeout_s, one_time=args.one_time)


def _i2c_proxy(args, hololink):
    i2c = hololink.get_i2c(args.address)
    # i2c_proxy runs forever.
    hololink_module.i2c_proxy(i2c, args.driver)


def _enumerate(args):
    for channel_metadata in hololink_module.HololinkEnumerator.enumerated(args.timeout):
        if args.all_metadata:
            logging.info("---")
            for name, value in channel_metadata.items():
                logging.info(f"{name}={value}")
        else:
            mac_id = channel_metadata.get("mac_id", "N/A")
            cpnx_version = channel_metadata.get("cpnx_version", "N/A")
            clnx_version = channel_metadata.get("clnx_version", "N/A")
            ip_address = channel_metadata.get("client_ip_address", "N/A")
            serial_number = channel_metadata.get("serial_number", "N/A")
            interface = channel_metadata.get("interface", "N/A")
            logging.info(
                f"{mac_id=!s} {cpnx_version=:#X} {clnx_version=:#X} {ip_address=!s} {serial_number=!s} {interface=!s}"
            )


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
        self._slow_spi = hololink.get_spi(
            spi_controller_address,
            chip_select=0,
            cpol=0,
            cpha=1,
            width=1,
            clock_divisor=0xF,
        )
        self._fast_spi = hololink.get_spi(
            spi_controller_address,
            chip_select=0,
            cpol=1,
            cpha=1,
            width=4,
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
        self._slow_spi.spi_transaction([], request, 0)
        # execute its spi flash command
        return self._fast_spi.spi_transaction(
            command_bytes, write_bytes, read_byte_count
        )


class CpnxFlash(WindbondW25q128jw):
    #
    def __init__(self, context, hololink, spi_controller_address):
        super().__init__(context, hololink, spi_controller_address)
        #       self._slow_spi = hololink.get_spi(
        #           spi_controller_address,
        #           chip_select=0,
        #           cpol=1,
        #           cpha=1,
        #           width=1,
        #           clock_divisor=0xF,
        #       )
        self._fast_spi = hololink.get_spi(
            spi_controller_address,
            chip_select=0,
            cpol=1,
            cpha=1,
            width=1,
            clock_divisor=0,
        )

    def _spi_command(self, command_bytes, write_bytes=[], read_byte_count=0):
        return self._fast_spi.spi_transaction(
            command_bytes, write_bytes, read_byte_count
        )


class Programmer:
    def __init__(self, args, manifest_filename):
        self._args = args
        self._manifest_filename = manifest_filename

    def fetch_manifest(self, section):
        with open(self._manifest_filename, "rt") as f:
            manifest = yaml.safe_load(f)
        self._manifest = manifest[section]

    def fetch_archive(self, args):
        archive = self._manifest["archive"]
        assert archive["type"] == "zip"
        if args.archive:
            f = open(args.archive, "rb")
        else:
            url = archive["url"]
            request = requests.get(
                url,
                headers={
                    "Content-Type": "application/json",
                },
            )
            if request.status_code != requests.codes.ok:
                raise Exception(
                    f'Unable to fetch "{url}"; status={request.status_code}'
                )
            f = tempfile.TemporaryFile()
            # Save the content
            f.write(request.content)
            f.seek(0)
        # Check the archive hash
        md5 = hashlib.md5()
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            md5.update(chunk)
        computed_md5 = md5.hexdigest()
        archive_md5 = archive["md5"]
        if computed_md5 != archive_md5:
            raise Exception(
                f"MD5 checksum is different, {archive_md5=} but {computed_md5=}"
            )
        #
        f.seek(0)
        self._archive = zipfile.ZipFile(f)

    def check_eula(self, args):
        if args.accept_eula:
            logging.info("All EULAs are accepted via command-line switch.")
            return
        licenses = self._manifest["licenses"]
        assert len(licenses) > 0
        print("You must accept EULA terms in order to continue;")
        print("For each document, press <Space> to see the next page;")
        print("At the end of the document, enter <Q> to continue.")
        input("To continue, press <Enter>: ")
        for license in licenses:
            content = self._archive.read(license["name"]).decode()
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
        for name, image in images.items():
            logging.info(f"{name=} {image=}")
            content_name = image["content"]
            expected_md5 = image["md5"]
            expected_size = image["size"]
            content = self._archive.read(content_name)
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
            self._content[name] = content

    def program_and_verify_images(self, hololink):
        self.program_clnx(
            hololink, hololink_module.CLNX_SPI_CTRL, self._content["clnx"]
        )
        self.verify_clnx(hololink, hololink_module.CLNX_SPI_CTRL, self._content["clnx"])
        self.program_cpnx(
            hololink, hololink_module.CPNX_SPI_CTRL, self._content["cpnx"]
        )
        self.verify_cpnx(hololink, hololink_module.CPNX_SPI_CTRL, self._content["cpnx"])

    def power_cycle(self, args, hololink):
        print("You must how physically power cycle the sensor bridge device.")
        if args.skip_power_cycle:
            return
        input("Press <Enter> to continue: ")

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


def _program(args, hololink):
    logging.info("manifest=%s" % (args.manifest,))
    programmer = Programmer(args, args.manifest)
    programmer.fetch_manifest("hololink")
    programmer.fetch_archive(args)
    programmer.check_eula(args)
    programmer.check_images()
    programmer.program_and_verify_images(hololink)
    programmer.power_cycle(args, hololink)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hololink",
        default="192.168.0.2",
        help="IP address of Hololink board",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Configure logging for debug output",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Configure logging for trace output",
    )
    parser.set_defaults(needs_hololink=True)
    subparsers = parser.add_subparsers(
        help="Subcommands", dest="command", required=True
    )

    # "fpga_version": Fetch the FPGA version register and show it.
    fpga_version = subparsers.add_parser(
        "fpga_version", help="Show the version ID of the Hololink FPGA."
    )
    fpga_version.set_defaults(go=_fpga_version)

    # "read_uint32 (address)": Display the value read from the given FPGA address.
    read_uint32 = subparsers.add_parser(
        "read_uint32",
        help="Display the value read from the given FPGA address.",
    )
    read_uint32.add_argument(
        "address",
        type=lambda x: int(x, 0),
        help="Address to read; prefix with 0x to interpret address as hex.",
    )
    read_uint32.set_defaults(go=_read_uint32)

    # "write_uint32 (address) (value)": Write a given value to the given address.
    write_uint32 = subparsers.add_parser(
        "write_uint32",
        help="Write a given value to the given address; prefix with 0x to interpret values as hex.",
    )
    write_uint32.add_argument(
        "address",
        type=lambda x: int(x, 0),
        help="Address to write; prefix with 0x to interpret address as hex.",
    )
    write_uint32.add_argument(
        "value",
        type=lambda x: int(x, 0),
        help="Value to write; prefix with 0x to interpret address as hex.",
    )
    write_uint32.set_defaults(go=_write_uint32)

    # "reset": Restart the hololink device.
    reset = subparsers.add_parser(
        "reset",
        help="Restart the Hololink device.",
    )
    reset.set_defaults(go=_reset)

    # "set-ip": Set the IP address of the device with the given MAC ID.
    set_ip = subparsers.add_parser(
        "set-ip",
        help="Configure the IP address for a given device",
    )
    set_ip.add_argument(
        "--one-time",
        action="store_true",
        help="Don't run in daemon mode.",
    )
    set_ip.add_argument(
        "mac_id_and_ip",
        help="Two arguments for each node: MAC ID and IP address of a device to configure.",
        nargs="+",
    )
    set_ip.set_defaults(go=_set_ip, needs_hololink=False)

    # "i2c-proxy": Run the daemon that provides I2C access
    # to hololink_i2c.ko.
    i2c_proxy = subparsers.add_parser(
        "i2c-proxy",
        help="Run the I2C proxy enabling hololink_i2c to access the I2C bus devices on the given hololink board.",
    )
    i2c_proxy.add_argument(
        "--address",
        type=lambda x: int(x, 0),  # allow users to say "--address=0xABC"
        default=hololink_module.CAM_I2C_CTRL,
        choices=(hololink_module.BL_I2C_CTRL, hololink_module.CAM_I2C_CTRL),
    )
    parser.add_argument(
        "--driver", default="/dev/hololink_i2c", help="Device to access."
    )
    i2c_proxy.set_defaults(go=_i2c_proxy)

    # "enumerate": Display enumeration information as it's received.
    enumeration = subparsers.add_parser(
        "enumerate",
        help="Run the I2C proxy enabling hololink_i2c to access the I2C bus devices on the given hololink board.",
    )
    enumeration.add_argument(
        "--timeout",
        type=float,
        help="Time out after the given number of seconds.",
    )
    enumeration.add_argument(
        "--all-metadata",
        action="store_true",
        help="Dump all channel metadata received.",
    )
    enumeration.set_defaults(go=_enumerate, needs_hololink=False)

    # "program": Update the flash memory(ies) on a Hololink board.
    program = subparsers.add_parser(
        "program",
        help="Program the hololink board via a manifest file.",
    )
    program.add_argument(
        "manifest",
        help="Filename of the firmware manifest file.",
    )
    program.add_argument(
        "--archive",
        help="Use a local zip archive instead of downloading it.",
    )
    program.add_argument(
        "--skip-program-clnx",
        action="store_true",
        help="Skip program_clnx",
    )
    program.add_argument(
        "--skip-verify-clnx",
        action="store_true",
        help="Skip verify_clnx",
    )
    program.add_argument(
        "--skip-program-cpnx",
        action="store_true",
        help="Skip program_cpnx",
    )
    program.add_argument(
        "--skip-verify-cpnx",
        action="store_true",
        help="Skip verify_cpnx",
    )
    program.add_argument(
        "--accept-eula",
        action="store_true",
        help="Provide non-interactive EULA acceptance",
    )
    program.add_argument(
        "--skip-power-cycle",
        action="store_true",
        help="Don't wait for confirmation of power cycle.",
    )
    program.set_defaults(go=_program)

    #
    args = parser.parse_args()

    if args.trace:
        logging.basicConfig(level=logging.TRACE)
    elif args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.needs_hololink:
        #
        channel_metadata = hololink_module.HololinkEnumerator.find_channel(
            channel_ip=args.hololink
        )
        hololink_channel = hololink_module.HololinkDataChannel(channel_metadata)
        hololink = hololink_channel.hololink()
        hololink.start()
        #
        try:
            args.go(args, hololink)
        finally:
            hololink.stop()
    else:
        args.go(args)


if __name__ == "__main__":
    main()
