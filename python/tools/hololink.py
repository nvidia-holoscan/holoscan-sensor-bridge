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
import glob
import hashlib
import logging
import os
import pydoc
import re
import shutil
import socket
import struct
import subprocess
import sys
import time

import requests
import yaml

import hololink as hololink_module

_FPGA_PTP_SYNC_STAT = 0x188


def _hex_or_na(value, spec="#X"):
    """Format an integer as hex; pass a non-integer (e.g. the "N/A" placeholder) through.

    Enumeration metadata fields are missing on a malformed or truncated BOOTP
    vendor payload, in which case callers substitute a string placeholder. A bare
    f"{value:#X}" raises TypeError on that placeholder, which is exactly the case
    you most want to see printed.
    """
    return format(value, spec) if isinstance(value, int) else str(value)


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


def _ptp_relock(args, hololink):
    """Repeatedly reset an idle HSB and measure how long PTP takes to re-lock."""
    if args.iterations <= 0:
        raise ValueError("--iterations must be greater than zero")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero")
    if args.slow_threshold < 0:
        raise ValueError("--slow-threshold must not be negative")

    lock_times_s = []
    slow_cycles = []
    failures = 0
    diagnostic_start = time.monotonic()

    print(
        "PTP re-lock diagnostic: "
        f"iterations={args.iterations} timeout_s={args.timeout:g}",
        flush=True,
    )
    for cycle in range(1, args.iterations + 1):
        cycle_start = time.monotonic()
        try:
            hololink.reset()
        except Exception as e:
            failures += 1
            print(
                f"PTP_RELOCK cycle={cycle}/{args.iterations} result=reset-error "
                f"error={e!r}",
                flush=True,
            )
            if not args.continue_on_failure:
                break
            continue

        reset_s = time.monotonic() - cycle_start
        lock_start = time.monotonic()
        timeout = hololink_module.Timeout(args.timeout)
        synchronized = hololink.ptp_synchronize(timeout)
        lock_s = time.monotonic() - lock_start
        try:
            status = hololink.read_uint32(_FPGA_PTP_SYNC_STAT)
            status_text = f"{status:#x}"
        except Exception as e:
            status_text = f"read-error:{e!r}"

        result = "locked" if synchronized else "timeout"
        print(
            f"PTP_RELOCK cycle={cycle}/{args.iterations} result={result} "
            f"reset_s={reset_s:.3f} lock_s={lock_s:.3f} status={status_text}",
            flush=True,
        )
        if synchronized:
            lock_times_s.append(lock_s)
            if lock_s > args.slow_threshold:
                slow_cycles.append((cycle, lock_s))
        else:
            failures += 1
            if not args.continue_on_failure:
                break

    elapsed_s = time.monotonic() - diagnostic_start
    slow_cycles_text = ",".join(
        f"{cycle}:{lock_s:.3f}s" for cycle, lock_s in slow_cycles
    )
    if not slow_cycles_text:
        slow_cycles_text = "none"
    if lock_times_s:
        ordered = sorted(lock_times_s)
        p95_index = max(0, (95 * len(ordered) + 99) // 100 - 1)
        print(
            "PTP_RELOCK summary "
            f"locked={len(lock_times_s)} failures={failures} "
            f"slow_over_{args.slow_threshold:g}s={len(slow_cycles)} "
            f"slow_cycles={slow_cycles_text} "
            f"min_s={ordered[0]:.3f} mean_s={sum(ordered) / len(ordered):.3f} "
            f"p95_s={ordered[p95_index]:.3f} max_s={ordered[-1]:.3f} "
            f"elapsed_s={elapsed_s:.3f}",
            flush=True,
        )
    else:
        print(
            f"PTP_RELOCK summary locked=0 failures={failures} "
            f"slow_over_{args.slow_threshold:g}s=0 slow_cycles=none "
            f"elapsed_s={elapsed_s:.3f}",
            flush=True,
        )

    if failures:
        raise SystemExit(1)


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
    if len(hardware_address) != 16:
        raise ValueError(
            f"hardware_address is {len(hardware_address)} bytes; expected 16."
        )
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
            # hsb_ip_version and fpga_crc go through _hex_or_na rather than a
            # ":#X" format spec: when the key is absent the default is the string
            # "N/A", and formatting that as hex raises TypeError -- which, inside
            # this callback, takes down the whole enumeration rather than printing
            # a partial row.
            logging.info(
                f"{mac_id=!s} hsb_ip_version={_hex_or_na(hsb_ip_version)} "
                f"fpga_crc={_hex_or_na(fpga_crc)} {ip_address=!s} {serial_number=!s} {interface=!s}"
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
        if (isr & 0x3F8) != 0:
            raise RuntimeError(f"Mailbox ISR is still {isr:#x} after reset.")

    def _execute(self, command, args=[]):
        if len(args) > (1024 + 2):  # our FIFO is 1024 words plus two for write
            raise ValueError(f"Mailbox command has too many words: {len(args)}.")
        if command >= 1024:  # we only get 10 bits for the command value
            raise ValueError(f"Mailbox command {command:#x} doesn't fit 10 bits.")
        timeout = Timer(timeout_s=2.0)
        # Inspired by https://www.intel.com/content/www/us/en/docs/programmable/683290/24-1/using-the.html
        # Make sure we have space.
        fifo_space = self._read_uint32(self.FIFO_SPACE)
        logging.trace(f"{fifo_space=}")
        if fifo_space <= 0:
            raise RuntimeError("Mailbox command FIFO is full.")
        # Write the command FIFO
        self._id += 1
        command_value = ((self._id & 0xF) << 24) | (len(args) << 12) | command
        cx = [command_value]
        cx.extend(args)
        for c in cx[:-1]:
            self._write_uint32(self.COMMAND, c)
            fifo_space = self._read_uint32(self.FIFO_SPACE)
            logging.debug(f"{fifo_space=}")
            if fifo_space <= 0:
                raise RuntimeError("Mailbox command FIFO is full.")
        self._write_uint32(self.COMMAND_EOP, cx[-1])
        # Wait for ISR flag
        while True:
            isr = self._read_uint32(self.ISR)
            logging.trace(f"{isr=:#x}")
            if isr & 1:
                break
            if (isr & 0x3F8) != 0:
                raise RuntimeError(f"Mailbox reports an error: {isr=:#x}.")
            timeout.check()
        # Wait for a response (which should be already available)
        while True:
            status = self._read_uint32(self.STATUS)
            length, eop, sop = (status >> 2), (status & 2) != 0, (status & 1) != 0
            logging.trace(f"{length=} {eop=} {sop=}")
            if sop:  # and (length > 0):
                if length <= 0:  # we always get a response header
                    raise RuntimeError("Mailbox response header is empty.")
                break
            timeout.check()
        # Fish out the response
        r = []
        while True:
            if length <= 0:
                raise RuntimeError("Mailbox response ran out of data.")
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
            if sop:
                raise RuntimeError("Unexpected start of packet in mailbox reply.")
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
        if not ((length == len(data)) or ((length == 0) and (len(data) == 1024))):
            raise RuntimeError(f"Mailbox reply has {len(data)} words; {length=}.")
        if response_id != (self._id & 0xF):
            raise RuntimeError(f"Mailbox reply {response_id=} is out of sequence.")
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
        if (address & 0x3) != 0:
            raise ValueError(f"{address=:#x} must be a multiple of four.")
        # only 32-bit addresses are supported by RSU_IMAGE_UPDATE
        if address != (address & 0xFFFFFFFF):
            raise ValueError(f"{address=:#x} doesn't fit in 32 bits.")
        unused = 0
        r = self._execute(self.RSU_IMAGE_UPDATE, [address, unused])
        if len(r) != 0:
            raise RuntimeError(f"RSU_IMAGE_UPDATE returned {len(r)} words.")


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
        if len(r) != 0:
            raise RuntimeError(f"QSPI_OPEN returned {len(r)} words.")

    def close(self):
        logging.info("close")
        r = self._execute(self.QSPI_CLOSE)
        if len(r) != 0:
            raise RuntimeError(f"QSPI_CLOSE returned {len(r)} words.")

    def set_cs(self, cs):
        logging.info("set_cs")
        if cs < 0 or cs >= 16:
            raise ValueError(f"{cs=} must be in the range 0 to 15.")
        r = self._execute(self.QSPI_SET_CS, [cs << 28])
        if len(r) != 0:
            raise RuntimeError(f"QSPI_SET_CS returned {len(r)} words.")

    def read(self, flash_address, flash_words):
        if (flash_address & 0x3) != 0:
            raise ValueError(f"{flash_address=:#x} must be a multiple of four.")
        # only 32-bit addresses are supported by QSPI_READ.
        if flash_address != (flash_address & 0xFFFFFFFF):
            raise ValueError(f"{flash_address=:#x} doesn't fit in 32 bits.")
        if flash_words > self.PAGE_WORDS:
            raise ValueError(f"{flash_words=} exceeds {self.PAGE_WORDS=}.")
        r = self._execute(self.QSPI_READ, [flash_address, flash_words])
        logging.debug(f"{len(r)=}")
        if len(r) != flash_words:
            raise RuntimeError(f"QSPI_READ returned {len(r)} of {flash_words} words.")
        return r

    def erase(self, flash_address, erase_words):
        # device supports 4k and 32k words as well
        if erase_words not in [self.ERASE_64K_WORDS]:
            raise ValueError(f"{erase_words=} isn't a supported erase size.")
        # only 32-bit addresses are supported by QSPI_ERASE.
        if flash_address != (flash_address & 0xFFFFFFFF):
            raise ValueError(f"{flash_address=:#x} doesn't fit in 32 bits.")
        if (flash_address & 0x3) != 0:
            raise ValueError(f"{flash_address=:#x} must be a multiple of four.")
        if (flash_address & (erase_words - 1)) != 0:
            raise ValueError(f"{flash_address=:#x} isn't erase-block aligned.")
        logging.info(f"erase {flash_address=:#x} {erase_words=:#x}")
        r = self._execute(self.QSPI_ERASE, [flash_address, erase_words])
        if len(r) != 0:
            raise RuntimeError(f"QSPI_ERASE returned {len(r)} words.")

    def write(self, flash_address, content_words):
        logging.debug(f"write {flash_address=:#x} {len(content_words)=:#x}")
        if (flash_address & 0x3) != 0:
            raise ValueError(f"{flash_address=:#x} must be a multiple of four.")
        # only 32-bit addresses are supported by QSPI_WRITE
        if flash_address != (flash_address & 0xFFFFFFFF):
            raise ValueError(f"{flash_address=:#x} doesn't fit in 32 bits.")
        content_size = len(content_words)
        if content_size > self.PAGE_WORDS:
            raise ValueError(f"{content_size=} exceeds {self.PAGE_WORDS=}.")
        args = [flash_address, content_size]
        args.extend(content_words)
        r = self._execute(self.QSPI_WRITE, args)
        if len(r) != 0:
            raise RuntimeError(f"QSPI_WRITE returned {len(r)} words.")


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
                if (current_bytes & 3) != 0:
                    raise ValueError(f"{current_bytes=} isn't a multiple of four.")
                current_words = current_bytes // 4
                actual_words = qspi.read(flash_address, current_words)
                actual = struct.pack("<%dI" % len(actual_words), *actual_words)
                # Verify the content.
                if expected[page : page + current_bytes] != actual:
                    raise RuntimeError(f"Flash verify failed at {flash_address=:#x}.")

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
                    if (len(content_bytes) & 0x3) != 0:
                        raise ValueError("Content size isn't a multiple of four.")
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
            raise Exception(f'Unsupported strategy "{strategy_name}" specified.')
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
        if len(licenses) == 0:
            raise RuntimeError("The manifest lists no licenses to accept.")
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


# ---------------------------------------------------------------------------
# "health": a one-shot diagnostic for an HSB device.
#
#   1. ICMP reachability
#   2. enumeration -- uuid, board, and whether the FPGA firmware is current
#   3. controller / control-plane status registers
#   4. HSB Lite only: is the CLNX's SPI flash answering with the expected JEDEC id
#
# Every step is best-effort and independent: a failure in one is recorded and the
# rest still run, so the report is always complete. The exit code is decided at
# the end from what actually failed.
# ---------------------------------------------------------------------------

# ANSI colors. There's no colorama/rich dependency in this project, so these are
# written out and suppressed whenever the output isn't an interactive terminal.
_COLORS = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "grey": "\033[90m",
}
_COLOR_RESET = "\033[0m"


def _color(text, name, enabled):
    if not enabled or name is None:
        return text
    return f"{_COLORS[name]}{text}{_COLOR_RESET}"


def _color_enabled(args):
    # Honor the de-facto NO_COLOR standard plus an explicit opt-out, and never
    # emit escapes into a pipe or a log file.
    if args.no_color or os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _ping(ip_address, count, timeout_s):
    """ICMP-echo `ip_address`. Returns (ok, detail); ok is None if we couldn't try.

    Shells out to the system ping rather than opening an AF_INET/SOCK_RAW socket:
    the ping binary is setuid (or carries cap_net_raw), so this keeps `health`
    usable as a normal user like every other subcommand here.
    """
    # ping's -W is whole seconds; floor it at 1 so a sub-second --timeout doesn't
    # become "-W 0", which fails every request regardless of the device.
    per_reply_timeout = max(1, int(round(timeout_s)))
    command = ["ping", "-c", str(count), "-W", str(per_reply_timeout), ip_address]
    logging.debug(f"running: {command}")
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None, "'ping' not found on PATH; skipping"
    except OSError as e:
        return None, f"could not run ping: {e}"

    stdout = process.stdout
    detail = []
    # "3 packets transmitted, 3 received, 0% packet loss, time 2003ms"
    transmitted = re.search(
        r"(\d+) packets transmitted,\s*(\d+)\s*(?:packets\s*)?received", stdout
    )
    if transmitted:
        detail.append(f"{transmitted.group(2)}/{transmitted.group(1)} replies")
    # "rtt min/avg/max/mdev = 0.152/0.184/0.221/0.028 ms"
    rtt = re.search(r"= [\d.]+/([\d.]+)/", stdout)
    if rtt:
        detail.append(f"rtt avg {rtt.group(1)} ms")
    if not detail:
        detail.append(f"ping exited {process.returncode}")
    return process.returncode == 0, ", ".join(detail)


_PING_ADVICE = (
    "Could not ping {ip_address}. Check the physical connections to the board and "
    "that the network profiles are set up according to the docs with the proper "
    "routing: "
    "https://docs.nvidia.com/holoscan/sensor-bridge/getting-started/host-setup"
)

# Firmware older than this doesn't broadcast the BOOTP requests enumeration
# listens for, so a device can be perfectly reachable and still never show up.
_MINIMUM_BOOTP_HSB_IP_VERSION = 0x2412

_CONTROL_PLANE_ADVICE = "cannot read/write to control plane ip"

_FW_MISMATCH_ADVICE = "fw mismatch, try power cycling the HSB."

_CONTROL_PORT = 8192
_CONTROL_PROBE_TIMEOUT_S = 2.0


def _probe_hsb_ip_version(ip_address):
    """Read HSB_IP_VERSION over the control plane without enumerating; None if silent.

    Deliberately speaks the wire protocol instead of going through Enumerator or
    Hololink: this runs precisely when enumeration has failed, so there's no
    metadata to build those from. Mirrors probe_version_udp() in
    tools/hsb_flasher/hsb_discovery.cpp -- one RD_DWORD straight at the control port.
    """
    address = hololink_module.HSB_IP_VERSION
    sequence = 0x0100
    request = bytes(
        [
            hololink_module.RD_DWORD,
            hololink_module.REQUEST_FLAGS_ACK_REQUEST,
            (sequence >> 8) & 0xFF,
            sequence & 0xFF,
            0,
            0,
            (address >> 24) & 0xFF,
            (address >> 16) & 0xFF,
            (address >> 8) & 0xFF,
            address & 0xFF,
        ]
    )
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.settimeout(_CONTROL_PROBE_TIMEOUT_S)
        s.sendto(request, (ip_address, _CONTROL_PORT))
        response = s.recv(256)
    except OSError as e:
        logging.debug(f"control plane probe of {ip_address} got no answer: {e}")
        return None
    finally:
        s.close()
    if len(response) < 14:
        logging.debug(f"control plane probe response too short: {len(response)} bytes")
        return None
    return int.from_bytes(response[10:14], "big") & 0xFFFF


def _enumeration_advice(hsb_ip_version):
    """What to tell the user when a pingable device won't enumerate."""
    if hsb_ip_version is None:
        return (
            "ping succeeds, but device communication failed. Check network settings "
            "and if necessary reflash the device"
        )
    if hsb_ip_version < _MINIMUM_BOOTP_HSB_IP_VERSION:
        return (
            "bootp enumeration failed due to old firmware. Please reflash the device "
            "to a newer firmware"
        )
    return (
        "device responds to Control Plane IP but bootp not being received. Check "
        "network settings, manually enumerate with 'hololink enumerate', and if "
        "necessary reflash the device"
    )


def _advise(message, use_color):
    """Write remediation guidance to stderr, after any pending report output."""
    sys.stdout.flush()
    # use_color reflects stdout; check stderr separately so we never write escape
    # sequences into a redirected stderr just because stdout is a terminal.
    print(_color(message, "red", use_color and sys.stderr.isatty()), file=sys.stderr)


# Below this the RoCE data-plane memory map this SDK programs isn't supported;
# DataChannel::configure_roce() throws UnsupportedVersion. Kept in sync by hand
# with MINIMUM_HSB_IP_VERSION in src/hololink/core/data_channel.cpp -- it has no
# pybind export to read it from.
_MINIMUM_HSB_IP_VERSION = 0x2602


def _firmware_information_directory():
    """Locate the hsb_flasher firmware manifests, or None.

    These YAML files are copied next to the hsb_flasher binary at build time
    (tools/hsb_flasher/CMakeLists.txt) and are NOT shipped in the Python wheel,
    so not finding them is an ordinary outcome rather than an error.
    """
    override = os.environ.get("HOLOLINK_FIRMWARE_INFORMATION")
    if override and os.path.isdir(override):
        return override
    # Walk up from this file looking for the in-repo copy.
    directory = os.path.dirname(os.path.abspath(__file__))
    while True:
        candidate = os.path.join(
            directory, "tools", "hsb_flasher", "firmware_information"
        )
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(directory)
        if parent == directory:
            break
        directory = parent
    # Fall back to the copy installed alongside the flasher executable.
    flasher = shutil.which("hsb_flasher")
    if flasher:
        candidate = os.path.join(
            os.path.dirname(os.path.abspath(flasher)), "firmware_information"
        )
        if os.path.isdir(candidate):
            return candidate
    return None


def _latest_firmware_for_uuid(fpga_uuid):
    """Highest firmware version any manifest offers for `fpga_uuid`, or None."""
    directory = _firmware_information_directory()
    if not directory or not fpga_uuid:
        return None
    latest = None
    for filename in sorted(glob.glob(os.path.join(directory, "*.y*ml"))):
        try:
            with open(filename, "r") as f:
                manifest = yaml.safe_load(f)
        except (OSError, yaml.YAMLError) as e:
            logging.debug(f"ignoring firmware manifest {filename}: {e}")
            continue
        if not isinstance(manifest, dict):
            continue
        # "fpga_uuid" is a scalar in some manifests and a list in others.
        uuids = manifest.get("fpga_uuid") or []
        if isinstance(uuids, str):
            uuids = [uuids]
        if not any(str(u).lower() == str(fpga_uuid).lower() for u in uuids):
            continue
        for entry in manifest.get("firmware_versions") or []:
            version = entry.get("version") if isinstance(entry, dict) else None
            if isinstance(version, str):
                try:
                    version = int(version, 0)
                except ValueError:
                    continue
            if not isinstance(version, int):
                continue
            latest = version if latest is None else max(latest, version)
    return latest


def _firmware_verdict(hsb_ip_version, fpga_uuid):
    """Classify the running firmware. Returns (text, color, failed)."""
    if not isinstance(hsb_ip_version, int):
        return "unknown version", "grey", False
    if hsb_ip_version < _MINIMUM_HSB_IP_VERSION:
        return (
            f"below SDK minimum {_MINIMUM_HSB_IP_VERSION:#x} - reflash required",
            "red",
            True,
        )
    latest = _latest_firmware_for_uuid(fpga_uuid)
    if latest is None:
        return "cannot check latest (no firmware manifest found)", "grey", False
    if hsb_ip_version >= latest:
        return "up to date", "green", False
    return (
        f"latest for this board is {latest:#x} - reflash suggested",
        "yellow",
        False,
    )


def _decode_i2c_status(value):
    """Render the I2C status flag bits (see I2C_* in src/hololink/core/hololink.hpp)."""
    flags = [
        ("BUSY", hololink_module.I2C_BUSY),
        ("FSM_ERR", hololink_module.I2C_FSM_ERR),
        ("I2C_ERR", hololink_module.I2C_I2C_ERR),
        ("I2C_NAK", hololink_module.I2C_I2C_NAK),
        ("DONE", hololink_module.I2C_DONE),
    ]
    return " ".join(name for name, bit in flags if value & bit)


def _status_registers():
    """(label, address, decoder) for the registers the health report dumps.

    HSB_IP_VERSION comes first deliberately: it's the one register every device
    implements and whose expected value we already know from enumeration, so it
    doubles as the proof that the control plane works at all. If it doesn't
    answer, nothing below it is worth believing.
    """
    return [
        ("HSB_IP_VERSION", hololink_module.HSB_IP_VERSION, None),
        # SPI has no named status constant in the core header, but its controller
        # layout matches I2C's: reg_status_ = base + 0x80 (core/hololink.cpp).
        ("SPI status", hololink_module.SPI_CTRL + 0x80, None),
        (
            "I2C status",
            hololink_module.I2C_CTRL + hololink_module.I2C_REG_STATUS,
            _decode_i2c_status,
        ),
        ("CTRL_EVT_STAT", hololink_module.CTRL_EVENT + 0x80, None),
        # FPGA_PTP_SYNC_STAT; a literal because it has no pybind export.
        ("PTP_SYNC_STAT", _FPGA_PTP_SYNC_STAT, None),
        ("FPGA_DATE", hololink_module.FPGA_DATE, None),
    ]


def _report_status_registers(args, hololink, hsb_ip_version, use_color):
    """Read and print each status register.

    Returns (control_plane_failed, version_mismatch); both are hard failures for
    the caller, and they're mutually exclusive by construction.

    A missing register is normally fine -- older firmware doesn't implement all of
    them -- with one exception. HSB_IP_VERSION is read first, every device has it,
    and enumeration already told us what it should say. Failing to read it means
    we aren't really talking to the control plane; reading a value that disagrees
    with the BOOTP payload means the device and its advertised identity are out of
    step, which a power cycle usually resolves.
    """
    print("")
    print("Status registers")
    control_plane_failed = False
    version_mismatch = False
    # An unimplemented register makes the SDK log a RESPONSE_INVALID_ADDR at ERROR
    # level. Here that's an expected result which we report as "No response", so
    # quiet logging for the duration of the sweep rather than printing an alarming
    # error next to every line that already says the same thing. Pass
    # --log-level 10 to see the SDK's messages and the exception text instead.
    #
    # Note this can't go through hololink_module.logging_level(): that is
    # logging.basicConfig(), which does nothing after the first call.
    quiet = args.log_level > logging.DEBUG
    if quiet:
        logging.disable(logging.CRITICAL)
    try:
        for label, address, decoder in _status_registers():
            is_version = address == hololink_module.HSB_IP_VERSION
            try:
                value = hololink.read_uint32(address)
            except Exception as e:
                # Older FPGA firmware doesn't implement all of these; only the
                # version register is mandatory.
                logging.debug(f"{label} ({address:#010x}) read failed: {e}")
                control_plane_failed = control_plane_failed or is_version
                print(
                    f"  {label:<16} {address:#010x}  "
                    + _color("No response", "red" if is_version else "grey", use_color)
                )
                continue
            notes = []
            if decoder:
                notes.append(decoder(value))
            if (
                is_version
                and isinstance(hsb_ip_version, int)
                and (value & 0xFFFF) != hsb_ip_version
            ):
                # The register and the BOOTP payload disagree about what's running.
                version_mismatch = True
                notes.append(
                    _color(
                        f"differs from enumerated {hsb_ip_version:#x}",
                        "red",
                        use_color,
                    )
                )
            suffix = "  ".join(n for n in notes if n)
            print(f"  {label:<16} {address:#010x}  {value:#010x}  {suffix}".rstrip())
    finally:
        if quiet:
            logging.disable(logging.NOTSET)
    return control_plane_failed, version_mismatch


# HSB Lite carries a CrossLink-NX (CLNX) and a CertusPro-NX (CPNX), each with its
# own Winbond W25Q128JW SPI NOR flash. The CLNX flash isn't on an FPGA chip select
# directly -- it sits behind the CrossLink-NX, so a transaction has to be bridged
# by first arming the CLNX's flash-forward register. Mirrors ClnxFlash::spi_command
# and WinbondW25q128jw::check_id in tools/hsb_flasher/flash_tools/hsb_lite/.
_HSB_LITE_UUID = "889b7ce3-65a5-4247-8b05-4ff1904c3359"
_CLNX_WRITE = 0x1
_CLNX_FLASH_FORWARD_EN_ADDRESS = 6
_SPI_JEDEC_ID = 0x9F
# WINBOND, W25Q128JW_IQ >> 8, W25Q128JW_IQ & 0xFF
_W25Q128JW_EXPECTED_ID = bytes([0xEF, 0x60, 0x18])
# Older FPGAs need the separate get_traditional_spi path, which this doesn't implement.
_MINIMUM_MODERN_SPI_VERSION = 0x2503


def _check_clnx(hololink, hsb_ip_version):
    """Read the CLNX flash JEDEC id. Returns (text, color, failed)."""
    if isinstance(hsb_ip_version, int) and hsb_ip_version < _MINIMUM_MODERN_SPI_VERSION:
        return (
            f"skipped; firmware {hsb_ip_version:#X} predates the current SPI "
            f"controller ({_MINIMUM_MODERN_SPI_VERSION:#X})",
            "grey",
            False,
        )
    try:
        # Same bus and chip select; the slow/CPOL=0 pair drives the CLNX itself,
        # the fast/CPOL=1 pair talks to the flash behind it.
        slow_spi = hololink.get_spi(
            hololink_module.CLNX_SPI_BUS,
            chip_select=0,
            cpol=0,
            cpha=1,
            width=1,
            prescaler=0xF,
        )
        fast_spi = hololink.get_spi(
            hololink_module.CLNX_SPI_BUS,
            chip_select=0,
            cpol=1,
            cpha=1,
            width=1,
            prescaler=0x4,
        )
        # Enable the CLNX bridge for exactly one transaction.
        transactions = 1
        forward = 0x1 | (transactions << 4)
        # NOTE: spi_transaction's pybind keyword names are copied from the I2C
        # binding and don't describe these arguments; pass them positionally.
        # They are (write_command_bytes, write_data_bytes, read_byte_count).
        slow_spi.spi_transaction(
            bytearray(),
            bytearray([_CLNX_WRITE, _CLNX_FLASH_FORWARD_EN_ADDRESS, forward]),
            0,
        )
        command = bytearray([_SPI_JEDEC_ID])
        result = bytes(fast_spi.spi_transaction(command, bytearray(), 3))
    except Exception as e:
        return f"SPI read failed: {e}", "red", True

    # The response echoes the command bytes back before the read data.
    if len(result) < len(command):
        return (
            f"SPI response shorter than command ({len(result)} bytes)",
            "red",
            True,
        )
    device_id = result[len(command) :]
    printed = " ".join(f"{b:02x}" for b in device_id)
    if device_id != _W25Q128JW_EXPECTED_ID:
        expected = " ".join(f"{b:02x}" for b in _W25Q128JW_EXPECTED_ID)
        return f"id={printed} - MISMATCH, expected {expected}", "red", True
    return f"id={printed} - OK", "green", False


def _health(args):
    ip_address = args.ip_address or args.hololink
    use_color = _color_enabled(args)
    failed = False

    print("")

    # 1) Is it on the network at all?
    ok, detail = _ping(ip_address, args.ping_count, args.timeout)
    if ok is None:
        print(f"ICMP        {_color(detail, 'grey', use_color)}")
    elif ok:
        print(f"ICMP        {ip_address} reachable - {detail}")
    else:
        failed = True
        print(
            f"ICMP        {_color(f'{ip_address} unreachable', 'red', use_color)} - {detail}"
        )
        _advise(_PING_ADVICE.format(ip_address=ip_address), use_color)

    # 2) Does it enumerate?
    try:
        channel_metadata = hololink_module.Enumerator.find_channel(
            channel_ip=ip_address,
            timeout=hololink_module.Timeout(args.timeout),
        )
    except Exception as e:
        print(f"Enumeration {_color('failed', 'red', use_color)} - {e}")
        if ok is not False:
            # It answers ICMP but won't enumerate. Ask the control plane directly
            # whether anything is home, and what firmware it claims -- that's what
            # separates "too old to broadcast BOOTP" from "BOOTP isn't reaching us"
            # from "not really talking to us at all".
            probed = _probe_hsb_ip_version(ip_address)
            print(
                "  control plane  "
                + (
                    _color("no response", "red", use_color)
                    if probed is None
                    else f"responds, firmware {probed:#x}"
                )
            )
            _advise(_enumeration_advice(probed), use_color)
        raise SystemExit(1)

    board = channel_metadata.get("board_description", "N/A")
    serial_number = channel_metadata.get("serial_number", "N/A")
    mac_id = channel_metadata.get("mac_id", "N/A")
    interface = channel_metadata.get("interface", "N/A")
    fpga_uuid = channel_metadata.get("fpga_uuid", "N/A")
    hsb_ip_version = channel_metadata.get("hsb_ip_version", "N/A")
    fpga_crc = channel_metadata.get("fpga_crc", "N/A")
    print(f"Enumeration ok - serial={serial_number} mac={mac_id} iface={interface}")
    print(f"  board     {board}")
    print(f"  uuid      {fpga_uuid}")

    text, color, firmware_failed = _firmware_verdict(hsb_ip_version, fpga_uuid)
    failed = failed or firmware_failed
    print(
        f"  firmware  {_hex_or_na(hsb_ip_version, '#x')}  "
        + _color(f"({text})", color, use_color)
    )
    print(f"  fpga_crc  {_hex_or_na(fpga_crc, '#x')}")

    # 3) and 4) both need the control plane up. start() only; deliberately no
    # reset() -- a health check shouldn't disrupt a device someone is using.
    hololink = hololink_module.Hololink.from_enumeration_metadata(channel_metadata)
    hololink.start()
    try:
        control_plane_failed, version_mismatch = _report_status_registers(
            args, hololink, hsb_ip_version, use_color
        )
        if control_plane_failed:
            failed = True
            _advise(_CONTROL_PLANE_ADVICE, use_color)
        elif version_mismatch:
            # The control plane is fine here, so the CLNX check below still runs.
            failed = True
            _advise(_FW_MISMATCH_ADVICE, use_color)

        # Only HSB Lite has a CLNX; on every other board the check is meaningless,
        # so it's left out of the report entirely rather than reported as skipped.
        if str(fpga_uuid).lower() == _HSB_LITE_UUID:
            print("")
            if control_plane_failed:
                # The CLNX check rides the same control plane, so running it would
                # just restate the problem already reported above.
                print(
                    "CLNX flash  "
                    + _color(
                        "skipped; control plane is not responding", "grey", use_color
                    )
                )
            else:
                text, color, clnx_failed = _check_clnx(hololink, hsb_ip_version)
                failed = failed or clnx_failed
                print("CLNX flash  " + _color(text, color, use_color))
    finally:
        hololink.stop()

    print("")
    if failed:
        raise SystemExit(1)


def manual_enumeration(args, interface=""):
    """Build a metadata blob for a board that never answered bootp.

    `interface` names the NIC control-plane requests should leave by; it is
    what enumeration would have recorded from the bootp reply's IP_PKTINFO.
    The default of "" resolves to ifindex 0, which leaves egress selection to
    the routing table -- the behavior this path has always had. Pass a real
    NIC name when the routing table cannot pick correctly, e.g. two boards on
    overlapping data-plane subnets.
    """
    m = {
        "control_port": 8192,
        "hsb_ip_version": 0x2502,
        "peer_ip": args.hololink,
        "interface": interface,
        "sequence_number_checking": 0,
        "serial_number": "100",
        "fpga_uuid": args.fpga_uuid,
        "block_enable": 0,
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

    # "ptp-relock": Repeatedly reset an idle HSB and measure PTP re-lock time.
    ptp_relock = subparsers.add_parser(
        "ptp-relock",
        help="Stress PTP re-lock after HSB reset without starting camera traffic.",
    )
    ptp_relock.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of reset/re-lock cycles to run (default: 100).",
    )
    ptp_relock.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for PTP lock in each cycle (default: 60).",
    )
    ptp_relock.add_argument(
        "--slow-threshold",
        type=float,
        default=20.0,
        help="Track successful locks slower than this many seconds (default: 20).",
    )
    ptp_relock.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue remaining cycles after a reset error or PTP timeout.",
    )
    ptp_relock.set_defaults(go=_ptp_relock)

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

    # "health": One-shot diagnostic report for a given device.
    health = subparsers.add_parser(
        "health",
        help="Report the health of an HSB device: ICMP reachability, enumeration and "
        "firmware version, status registers, and (HSB Lite) CLNX flash ID.",
    )
    health.add_argument(
        "ip_address",
        nargs="?",
        default=None,
        help="IP address of the HSB device; defaults to --hololink.",
    )
    health.add_argument(
        "--ping-count",
        type=int,
        default=3,
        help="Number of ICMP echo requests to send.",
    )
    health.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for ICMP replies and for enumeration.",
    )
    health.add_argument(
        "--no-color",
        action="store_true",
        help="Don't colorize the report.",
    )
    health.set_defaults(go=_health, needs_hololink=False)

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
