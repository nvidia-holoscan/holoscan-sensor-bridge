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
import pydoc
import socket
import struct
import sys
import time

import requests
import yaml

import hololink as hololink_module


def _hsb_ip_version(args, hololink):
    hsb_ip_version = hololink.get_hsb_ip_version()
    print(hex(hsb_ip_version))


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


def _set_ips(ips_by_mac, timeout_s=None, one_time=False, interface=""):
    enumerator = hololink_module.Enumerator(interface)
    # Runs forever if "timeout_s" is None
    # Make sure the MAC IDs are all upper case.
    by_mac = {k.upper(): v for k, v in ips_by_mac.items()}
    reported = {}

    #
    def call_back(enumerator, packet, metadata):
        logging.trace(f"Enumeration {metadata=}")
        mac_id = metadata.get("mac_id")  # Note that this value is upper case
        peer_ip = metadata.get("peer_ip")
        new_peer_ip = by_mac.get(mac_id)
        if new_peer_ip is None:
            # This is true if any of (mac_id, peer_ip, new_peer_ip) aren't provided.
            # continue
            return True
        if new_peer_ip == peer_ip:
            if not reported.get(mac_id):
                logging.info(f"Found {mac_id=} found using {peer_ip=}")
                reported[mac_id] = True
            # We're good.
            if one_time:
                if len(reported) == len(by_mac):
                    return False
            # continue
            return True
        # At this point, let's update that thing.
        # Set our local ARP cache so that we don't generate
        # an ARP request to the new IP-- the client doesn't
        # know it's IP yet so it won't be able to answer.
        local_device = metadata["interface"]
        local_mac = hololink_module.local_mac(local_device)
        local_ip = metadata["interface_address"]
        logging.trace(
            f"{local_device=} {local_mac=} {local_ip=} {new_peer_ip=} {mac_id=}"
        )
        # This check is here because I somehow seem to fat-finger
        # my way into this condition way too often.
        if new_peer_ip == local_ip:
            raise Exception(
                f"Can't assign {new_peer_ip} to {mac_id} because that's the host IP address."
            )
        logging.info(f"Updating {mac_id=} from {peer_ip=} to {new_peer_ip=}")
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            socket_fd = s.fileno()
            r = hololink_module.ArpWrapper.arp_set(
                socket_fd, local_device, new_peer_ip, mac_id
            )
            if r != 0:
                raise Exception(f"Unable to set IP address, errno={r}.")
        finally:
            s.close()
        # Now send the bootp reply that reconfigures it.
        reply = _make_bootp_reply(
            metadata,
            new_peer_ip,
            local_ip,
        )
        enumerator.send_bootp_reply(new_peer_ip, reply, metadata)
        if mac_id in reported:
            del reported[mac_id]

        # continue
        return True

    enumerator.enumeration_packets(call_back)


def _make_bootp_reply(metadata, new_device_ip, local_ip):
    reply = bytearray(1000)
    serializer = hololink_module.Serializer(reply)
    BOOTREPLY = 2
    serializer.append_uint8(BOOTREPLY)  # opcode
    serializer.append_uint8(metadata["hardware_type"])
    serializer.append_uint8(metadata["hardware_address_length"])
    hops = 0
    serializer.append_uint8(hops)
    serializer.append_uint32_be(metadata["transaction_id"])
    serializer.append_uint16_be(metadata["seconds"])
    flags = 0
    serializer.append_uint16_be(flags)
    ciaddr = 0  # per bootp spec
    serializer.append_uint32_be(ciaddr)
    serializer.append_buffer(socket.inet_aton(new_device_ip))
    serializer.append_buffer(socket.inet_aton(local_ip))
    gateway_ip = 0
    serializer.append_uint32_be(gateway_ip)
    hardware_address = metadata["hardware_address"]
    assert len(hardware_address) == 16
    # Note that hardware_address comes from metadata
    # as a list but append_buffer wants it as bytes.
    serializer.append_buffer(bytes(hardware_address))
    logging.trace(f"{len(hardware_address)=}")
    host_name = bytearray(64)
    serializer.append_buffer(host_name)
    file_name = bytearray(128)
    serializer.append_buffer(file_name)
    vendor_specific = bytearray(64)
    serializer.append_buffer(vendor_specific)
    return serializer.data()


def _set_ip(args):
    mac_id_and_ip = args.mac_id_and_ip
    # given a list [0,1,2,3,...] this produces a list [(0,1), (2,3), ...]
    settings = zip(mac_id_and_ip[0::2], mac_id_and_ip[1::2])
    ips_by_mac = {k.upper(): v for k, v in settings}
    #
    timeout_s = None
    if args.one_time:
        timeout_s = 45
    else:
        logging.info(
            "Running in daemon mode; run with '--one-time' to exit after configuration."
        )
    _set_ips(
        ips_by_mac,
        timeout_s=timeout_s,
        one_time=args.one_time,
        interface=args.interface,
    )


def _i2c_proxy(args, hololink):
    i2c = hololink.get_i2c(args.i2c_bus)
    # i2c_proxy runs forever.
    hololink_module.i2c_proxy(i2c, args.driver)


def _enumerate(args):
    def call_back(channel_metadata):
        if args.all_metadata:
            logging.info("---")
            logging.info(f"{channel_metadata}")
        else:
            mac_id = channel_metadata.get("mac_id", "N/A")
            hsb_ip_version = channel_metadata.get("hsb_ip_version", "N/A")
            fpga_crc = channel_metadata.get("fpga_crc", "N/A")
            ip_address = channel_metadata.get("peer_ip", "N/A")
            serial_number = channel_metadata.get("serial_number", "N/A")
            interface = channel_metadata.get("interface", "N/A")
            logging.info(
                f"{mac_id=!s} {hsb_ip_version=:#X} {fpga_crc=:#X} {ip_address=!s} {serial_number=!s} {interface=!s}"
            )
        # continue
        return True

    hololink_module.Enumerator.enumerated(
        call_back,
        args.timeout if not args.timeout else hololink_module.Timeout(args.timeout),
    )


class Timer:
    def __init__(self, timeout_s):
        self._start = time.monotonic()
        self._deadline = self._start + timeout_s
        self._tick_s = timeout_s
        self._first = True

    def check(self):
        now = time.monotonic()
        if now > self._deadline:
            raise Exception("Timeout.")

    def tick(self):
        """Returns a single True every timeout_s period from the start; False otherwise."""
        now = time.monotonic()
        if now >= self._deadline:
            self._deadline += self._tick_s
            return True
        return False

    def first(self):
        r = self._first
        self._first = False
        return r


class StratixMailboxError(Exception):
    def __init__(self, response_id, length, error_code, data, *args):
        self.response_id = response_id
        self.length = length
        self.error_code = error_code
        self.data = data
        super().__init__(*args)

    def log_error(self):
        logging.error(
            f"{self.response_id=:#x} {self.length=:#x} {self.error_code=:#x} {self.data=}"
        )


# build exception hook for StratixMailboxError to only log if uncaught
# if exception is not caught, an error the error is logged. else behave as normal.
def build_stratix_mailbox_except_hook(original_hook):
    def stratix_mailbox_except_hook(type, value, traceback):
        if isinstance(value, StratixMailboxError):
            value.log_error()
        original_hook(type, value, traceback)

    return stratix_mailbox_except_hook


# register the exception hook in the system
sys.excepthook = build_stratix_mailbox_except_hook(sys.excepthook)


class StratixMailbox:
    # Addresses for use with self.write_uint32
    COMMAND = 0x00
    COMMAND_EOP = 0x04
    FIFO_SPACE = 0x08
    RESPONSE = 0x14
    STATUS = 0x18
    IER = 0x1C
    ISR = 0x20
    TIMER_1 = 0x24
    TIMER_2 = 0x28
    # Commands
    GET_IDCODE = 0x10
    GET_CHIPID = 0x12
    RSU_STATUS = 0x5B
    RSU_IMAGE_UPDATE = 0x5C

    #
    def __init__(self, hololink, address=0x3000_0000):
        self._hololink = hololink
        self._address = address
        self._id = os.getpid()  # start with a randomish value
        logging.debug(f"{self._id=:#x}")
        timeout = Timer(timeout_s=1.0)
        self._flush(timeout)
        self._clear_isr(hololink)

    def _write_uint32(self, address, value):
        self._hololink.write_uint32(address + self._address, value)

    def _read_uint32(self, address):
        return self._hololink.read_uint32(address + self._address)

    def _flush(self, timeout):
        flushed = 0
        while True:
            status = self._read_uint32(self.STATUS)
            length, eop, sop = (status >> 2), (status & 2) != 0, (status & 1) != 0
            logging.debug(f"{length=} {eop=} {sop=}")
            if length == 0:
                break
            flushed_response = self._read_uint32(self.RESPONSE)
            logging.debug(f"{flushed_response=:#x}")
            flushed += 1
            timeout.check()
        logging.debug(f"{flushed=}")

    def _clear_isr(self, hololink):
        """Work around a problem where abnormal termination
        of a previous run of this program can leave the
        mailbox controller in a bad state."""
        isr = self._read_uint32(self.ISR)
        if (isr & 0x3F8) != 0:
            logging.info(f"{isr=:#x}; resetting it to clear this condition.")
            hololink.reset()
        isr = self._read_uint32(self.ISR)
        assert (isr & 0x3F8) == 0

    def _execute(self, command, args=[]):
        assert len(args) <= (1024 + 2)  # our FIFO is 1024 words plus two for write
        assert command < 1024  # we only get 10 bits for the command value
        timeout = Timer(timeout_s=2.0)
        # Inspired by https://www.intel.com/content/www/us/en/docs/programmable/683290/24-1/using-the.html
        # Make sure we have space.
        fifo_space = self._read_uint32(self.FIFO_SPACE)
        logging.trace(f"{fifo_space=}")
        assert fifo_space > 0
        # Write the command FIFO
        self._id += 1
        command_value = ((self._id & 0xF) << 24) | (len(args) << 12) | command
        cx = [command_value]
        cx.extend(args)
        for c in cx[:-1]:
            self._write_uint32(self.COMMAND, c)
            fifo_space = self._read_uint32(self.FIFO_SPACE)
            logging.debug(f"{fifo_space=}")
            assert fifo_space > 0
        self._write_uint32(self.COMMAND_EOP, cx[-1])
        # Wait for ISR flag
        while True:
            isr = self._read_uint32(self.ISR)
            logging.trace(f"{isr=:#x}")
            if isr & 1:
                break
            assert (isr & 0x3F8) == 0
            timeout.check()
        # Wait for a response (which should be already available)
        while True:
            status = self._read_uint32(self.STATUS)
            length, eop, sop = (status >> 2), (status & 2) != 0, (status & 1) != 0
            logging.trace(f"{length=} {eop=} {sop=}")
            if sop:  # and (length > 0):
                assert length > 0  # we always get a response header
                break
            timeout.check()
        # Fish out the response
        r = []
        while True:
            assert length > 0
            response = self._read_uint32(self.RESPONSE)
            logging.debug(f"{response=:#x}")
            r.append(response)
            # NOTE that we borrow "eop" from above, first time through--
            # we always get at least 1 response header word.
            if eop:
                break
            status = self._read_uint32(self.STATUS)
            length, eop, sop = (status >> 2), (status & 2) != 0, (status & 1) != 0
            logging.debug(f"{length=} {eop=} {sop=}")
            assert not sop
            timeout.check()
        # Response header is first
        header, data = r[0], r[1:]
        response_id, length, error_code = (
            (header >> 24) & 0xF,
            (header >> 12) & 0x7FF,
            (header >> 0) & 0x7FF,
        )
        logging.debug(f"{response_id=} {length=} {error_code=:#x} {len(data)=}")
        if error_code != 0:
            raise StratixMailboxError(response_id, length, error_code, data)
        assert (length == len(data)) or ((length == 0) and (len(data) == 1024))
        assert response_id == (self._id & 0xF)
        return data

    def get_idcode(self):
        r = self._execute(self.GET_IDCODE)
        return r[0]

    def get_chipid(self):
        r = self._execute(self.GET_CHIPID)
        return r

    def qspi(self):
        return StratixQspi(self)

    def rsu_status(self):
        logging.info("rsu_status")
        r = self._execute(self.RSU_STATUS)
        status = {
            "current": (r[0] | (r[1] << 32)),
            "failing": (r[2] | (r[3] << 32)),
            "state": r[4],
            "version": r[5],
            "error_location": r[6],
            "error_details": r[7],
            "retries": r[8],
        }
        return status

    def fpga_application_address(self):
        # What flash address are we using?
        rsu_status = self.rsu_status()
        rsu_status_s = ["%s:0x%X" % (k, v) for k, v in rsu_status.items()]
        logging.info(f"{rsu_status_s=}")
        # fpga_flash_address = rsu_status["current"]
        # return fpga_flash_address
        # FOR NOW we're just hard-coding this address.
        return 0xB10000

    def rsu_image_update(self, address):
        logging.info(f"rsu_image_update {address=:#x}")
        assert (address & 0x3) == 0
        # only 32-bit addresses are supported by RSU_IMAGE_UPDATE
        assert address == (address & 0xFFFFFFFF)
        unused = 0
        r = self._execute(self.RSU_IMAGE_UPDATE, [address, unused])
        assert len(r) == 0


class StratixQspi:
    # Commands
    QSPI_OPEN = 0x32
    QSPI_CLOSE = 0x33
    QSPI_SET_CS = 0x34
    QSPI_ERASE = 0x38
    QSPI_WRITE = 0x39
    QSPI_READ = 0x3A
    # Sizes
    PAGE_WORDS = 1024
    PAGE_BYTES = PAGE_WORDS * 4
    #
    ERASE_64K_WORDS = 0x4000
    ERASE_64K_BYTES = ERASE_64K_WORDS * 4

    # Errors
    QSPI_NOT_OPENED_BY_CLIENT = 0x008

    #
    def __init__(self, mailbox):
        self._mailbox = mailbox

    def __enter__(self):
        try:
            self.close()
        except StratixMailboxError as e:
            if e.error_code == self.QSPI_NOT_OPENED_BY_CLIENT:
                logging.debug(f"Ignoring {e}({type(e)=})")
            else:
                raise
        self.open()
        self.set_cs(0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _execute(self, command, args=[]):
        return self._mailbox._execute(command, args)

    def open(self):
        logging.info("open")
        r = self._execute(self.QSPI_OPEN)
        assert len(r) == 0

    def close(self):
        logging.info("close")
        r = self._execute(self.QSPI_CLOSE)
        assert len(r) == 0

    def set_cs(self, cs):
        logging.info("set_cs")
        assert cs < 16
        r = self._execute(self.QSPI_SET_CS, [cs << 28])
        assert len(r) == 0

    def read(self, flash_address, flash_words):
        assert (flash_address & 0x3) == 0
        # only 32-bit addresses are supported by QSPI_READ.
        assert flash_address == (flash_address & 0xFFFFFFFF)
        assert flash_words <= self.PAGE_WORDS
        r = self._execute(self.QSPI_READ, [flash_address, flash_words])
        logging.debug(f"{len(r)=}")
        assert len(r) == flash_words
        return r

    def erase(self, flash_address, erase_words):
        # device supports 4k and 32k words as well
        assert erase_words in [self.ERASE_64K_WORDS]
        # only 32-bit addresses are supported by QSPI_ERASE.
        assert flash_address == (flash_address & 0xFFFFFFFF)
        assert (flash_address & 0x3) == 0
        assert (flash_address & (erase_words - 1)) == 0
        logging.info(f"erase {flash_address=:#x} {erase_words=:#x}")
        r = self._execute(self.QSPI_ERASE, [flash_address, erase_words])
        assert len(r) == 0

    def write(self, flash_address, content_words):
        logging.debug(f"write {flash_address=:#x} {len(content_words)=:#x}")
        assert (flash_address & 0x3) == 0
        # only 32-bit addresses are supported by QSPI_WRITE
        assert flash_address == (flash_address & 0xFFFFFFFF)
        content_size = len(content_words)
        assert content_size <= self.PAGE_WORDS
        args = [flash_address, content_size]
        args.extend(content_words)
        r = self._execute(self.QSPI_WRITE, args)
        assert len(r) == 0


class SensorBridgeStrategy:
    def __init__(self, programmer):
        self._programmer = programmer
        self._args = programmer._args

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


class SensorBridge100Strategy(SensorBridgeStrategy):
    def __init__(self, programmer):
        super().__init__(programmer)

    def check_fpga_uuid(self, fpga_uuid):
        supported_fpga_uuids = {
            "7a377bf7-76cb-4756-a4c5-7dddaed8354b",  # Stratix 10 HSB
        }
        return fpga_uuid in supported_fpga_uuids

    def program_and_verify_images(self, hololink):
        content = self._programmer._content
        if "stratix" in content:
            original_stratix_content = content["stratix"]
            stratix_content = original_stratix_content.translate(reverse_map)
            self.program_stratix(hololink, stratix_content)
            self.verify_stratix(hololink, stratix_content)

    def verify_stratix(self, hololink, expected):
        if self._args.skip_verify_stratix:
            logging.info(
                "Skipping verification of stratix per command-line instructions."
            )
            return
        logging.info("Verifying stratix.")
        expected_size = len(expected)
        # Attach to the device
        mailbox = StratixMailbox(hololink)
        id_code = mailbox.get_idcode()
        logging.info(f"{id_code=:#x}")
        chip_id = ["0x%X" % x for x in mailbox.get_chipid()]
        logging.info(f"{chip_id=}")
        fpga_flash_address = mailbox.fpga_application_address()
        # Read/verify the contents
        show_status = Timer(timeout_s=10.0)
        with StratixQspi(mailbox) as qspi:
            byte_count = qspi.PAGE_BYTES
            for page in range(0, expected_size, byte_count):
                flash_address = fpga_flash_address + page
                # Let users know that we're running
                if show_status.first() or show_status.tick():
                    percent = int(100 * (page / expected_size))
                    logging.info(f"{flash_address=:#x} ({percent}%)")
                # Fetch this page
                current_bytes = min(byte_count, expected_size - page)
                assert (current_bytes & 3) == 0
                current_words = current_bytes // 4
                actual_words = qspi.read(flash_address, current_words)
                actual = struct.pack("<%dI" % len(actual_words), *actual_words)
                # Verify the content.
                assert expected[page : page + current_bytes] == actual

    def program_stratix(self, hololink, content):
        if self._args.skip_program_stratix:
            logging.info(
                "Skipping programming of stratix per command-line instructions."
            )
            return
        logging.info("Programming stratix.")
        content_size = len(content)
        # Attach to the device
        mailbox = StratixMailbox(hololink)
        id_code = mailbox.get_idcode()
        logging.info(f"{id_code=:#x}")
        chip_id = ["0x%X" % x for x in mailbox.get_chipid()]
        logging.info(f"{chip_id=}")
        fpga_flash_address = mailbox.fpga_application_address()
        # Program contents
        show_status = Timer(timeout_s=10.0)
        with StratixQspi(mailbox) as qspi:
            byte_count = (
                qspi.PAGE_BYTES
            )  # we can't write 1024 because the command adds 2 words
            for erase_address in range(0, content_size, qspi.ERASE_64K_BYTES):
                qspi.erase(fpga_flash_address + erase_address, qspi.ERASE_64K_WORDS)
                remaining = min(qspi.ERASE_64K_BYTES, content_size - erase_address)
                for write_address in range(
                    erase_address, erase_address + remaining, byte_count
                ):
                    flash_address = fpga_flash_address + write_address
                    if show_status.first() or show_status.tick():
                        percent = int(100 * (write_address / content_size))
                        logging.info(f"{flash_address=:#x} ({percent}%)")
                    content_bytes = content[write_address : write_address + byte_count]
                    assert (len(content_bytes) & 0x3) == 0
                    word_count = (
                        len(content_bytes) // 4
                    )  # which could be less than the page size on the last page
                    content_words = struct.unpack("<%dI" % word_count, content_bytes)
                    qspi.write(flash_address, content_words)

    def power_cycle(self, args, hololink):
        # Attach to the device
        mailbox = StratixMailbox(hololink)
        id_code = mailbox.get_idcode()
        logging.info(f"{id_code=:#x}")
        chip_id = ["0x%X" % x for x in mailbox.get_chipid()]
        logging.info(f"{chip_id=}")
        fpga_flash_address = mailbox.fpga_application_address()
        mailbox.rsu_image_update(fpga_flash_address)
        # Now wait for the device to come back up.
        # This guy raises an exception if we're not found;
        # this can happen if set-ip is used in one-time
        # mode.
        hololink_module.Enumerator.find_channel(
            channel_ip=args.hololink, timeout=hololink_module.Timeout(30)
        )
        # Following reset, allow for more timeout on this initial read.
        get_hsb_ip_version_timeout = hololink_module.Timeout(timeout_s=30, retry_s=0.2)
        version = hololink.get_hsb_ip_version(
            timeout=get_hsb_ip_version_timeout, check_sequence=False
        )
        logging.info(f"{version=:#x}")


strategies = {
    "sensor_bridge_100": SensorBridge100Strategy,
}


class Programmer:
    def __init__(self, args, manifest_filename):
        self._args = args
        self._manifest_filename = manifest_filename
        self._skip_eula = False

    def fetch_manifest(self, section):
        with open(self._manifest_filename, "rt") as f:
            manifest = yaml.safe_load(f)
        self._manifest = manifest[section]
        strategy_name = self._manifest["strategy"]
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
        "fpga_uuid": args.fpga_uuid,
    }
    metadata = hololink_module.Metadata(m)
    hololink_module.DataChannel.use_data_plane_configuration(metadata, 0)
    hololink_module.DataChannel.use_sensor(metadata, 0)
    return metadata


def _program(args):
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
    parser.set_defaults(needs_hololink=True)
    subparsers = parser.add_subparsers(
        help="Subcommands", dest="command", required=True
    )

    # "hsb_ip_version": Fetch the HSB IP version register and show it.
    hsb_ip_version = subparsers.add_parser(
        "hsb_ip_version", help="Show the version ID of the Hololink FPGA."
    )
    hsb_ip_version.set_defaults(go=_hsb_ip_version)

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
        "--interface",
        default="",
        help="Only listen on the given interface.",
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
        "--i2c-bus",
        type=lambda x: int(x, 0),  # allow users to say "--i2c-bus=1"
        default=hololink_module.CAM_I2C_BUS,
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
        "--accept-eula",
        action="store_true",
        help="Provide non-interactive EULA acceptance",
    )
    program.add_argument(
        "--skip-power-cycle",
        action="store_true",
        help="Don't wait for confirmation of power cycle.",
    )
    program.add_argument(
        "--skip-program-stratix",
        action="store_true",
        help="Skip program_stratix",
    )
    program.add_argument(
        "--skip-verify-stratix",
        action="store_true",
        help="Skip verify_stratix",
    )

    program.add_argument(
        "--fpga-uuid",
        default="7a377bf7-76cb-4756-a4c5-7dddaed8354b",  # Stratix 10 HSB
        help="FPGA UUID, which determines how the device is programmed",
    )
    program.set_defaults(go=_program, needs_hololink=False)

    #
    args = parser.parse_args()
    hololink_module.logging_level(args.log_level)

    if args.needs_hololink:
        #
        if args.force:
            channel_metadata = manual_enumeration(args)
        else:
            channel_metadata = hololink_module.Enumerator.find_channel(
                channel_ip=args.hololink
            )
        hololink_channel = hololink_module.DataChannel(channel_metadata)
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
