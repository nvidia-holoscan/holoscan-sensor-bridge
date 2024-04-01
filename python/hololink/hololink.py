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

import array
import collections
import fcntl
import logging
import select
import socket
import struct
import time

from .native import ArpWrapper, Deserializer, Serializer


def u32s(s):
    i = 0
    length = len(s)
    while True:
        rem = length - i
        if rem == 0:
            return
        r = s[i] << 0
        i += 1
        if rem > 1:
            r |= s[i] << 8
            i += 1
        if rem > 2:
            r |= s[i] << 16
            i += 1
        if rem > 3:
            r |= s[i] << 24
            i += 1
        yield r


# This memory map is only supported on CPNX FPGAs that are this
# version or newer.
MINIMUM_CPNX_VERSION = 0x2402

# Hololink-lite data plane configuration is implied by the value
# passed in the bootp transaction_id field, which is coopted
# by FPGA to imply which port is publishing the request.  We use
# that port ID to figure out what the address of the port's
# configuration data is; which is the value listed here.
HololinkChannelConfiguration = collections.namedtuple(
    "HololinkChannelConfiguration", ["configuration_address", "vip_mask"]
)
BOOTP_TRANSACTION_ID_MAP = {
    0: HololinkChannelConfiguration(0x0200_0000, vip_mask=0x1),
    1: HololinkChannelConfiguration(0x0201_0000, vip_mask=0x2),
}
# SPI interfaces
CLNX_SPI_CTRL = 0x0300_0000
CPNX_SPI_CTRL = 0x0300_0200
# I2C interfaces
BL_I2C_CTRL = 0x0400_0000
CAM_I2C_CTRL = 0x0400_0200

# packet command byte
WR_DWORD = 0x04
RD_DWORD = 0x14
# request packet flag bits
REQUEST_FLAGS_ACK_REQUEST = 0b0000_0001
# response codes
RESPONSE_SUCCESS = 0
RESPONSE_INVALID_CMD = 0x04

# control flags
I2C_START = 0b0000_0000_0000_0001
I2C_CORE_EN = 0b0000_0000_0000_0010
I2C_DONE_CLEAR = 0b0000_0000_0001_0000
I2C_BUSY = 0b0000_0001_0000_0000
I2C_DONE = 0b0001_0000_0000_0000

# SPI control flags
SPI_START = 0b0000_0000_0000_0001
SPI_BUSY = 0b0000_0001_0000_0000
# SPI_CFG
SPI_CFG_CPOL = 0b0000_0000_0001_0000
SPI_CFG_CPHA = 0b0000_0000_0010_0000

#
FPGA_VERSION = 0x80
FPGA_DATE = 0x84

#
DEFAULT_TIMEOUT_S = 0.5
DEFAULT_RETRY_S = 0.1

# board IDs
HOLOLINK_LITE_BOARD_ID = 1
HOLOLINK_BOARD_ID = 2


class Timeout:
    @staticmethod
    def now():
        return time.monotonic()

    @staticmethod
    def configure(timeout):
        if timeout is None:
            now = Timeout.now()
            timeout = Timeout(now, DEFAULT_TIMEOUT_S)
        return timeout

    @staticmethod
    def configure_i2c(timeout):
        if timeout is None:
            now = Timeout.now()
            timeout = Timeout(now, DEFAULT_TIMEOUT_S)
            timeout.retry_s()
        return timeout

    @staticmethod
    def configure_spi(timeout):
        if timeout is None:
            now = Timeout.now()
            timeout = Timeout(now, DEFAULT_TIMEOUT_S)
            timeout.retry_s()
        return timeout

    def __init__(self, start, timeout_s):
        self._start = start
        self._timeout_s = timeout_s
        self._retry_s = None
        self._deadline = start + timeout_s
        self._expiry = self._deadline

    def retry_s(self, retry_s=DEFAULT_RETRY_S):
        self._retry_s = retry_s
        self._deadline = self._start + retry_s
        return self

    def expired(self):
        now = self.now()
        return now >= self._deadline

    def trigger_s(self):
        now = self.now()
        dt = self._deadline - now
        return dt

    def retry(self):
        if self._retry_s is None:
            return False
        now = self.now()
        if now >= self._expiry:
            return False
        if self._deadline <= now:
            self._deadline += self._retry_s
        return True

    def __repr__(self):
        return (
            "Timeout@0x%X(start=%s,timeout_s=%s,retry_s=%s,deadline=%s,expiry=%s,now=%s)"
            % (
                id(self),
                self._start,
                self._timeout_s,
                self._retry_s,
                self._deadline,
                self._expiry,
                Timeout.now(),
            )
        )


def timeout(timeout_s=DEFAULT_TIMEOUT_S):
    now = Timeout.now()
    return Timeout(now, timeout_s)


def retry(retry_s=DEFAULT_RETRY_S, timeout_s=DEFAULT_TIMEOUT_S):
    return timeout(timeout_s).retry_s(retry_s)


# Camera Receiver interfaces
VP_START = [0x00, 0x80]
# Note that these are offsets from VP_START.
DP_PACKET_SIZE = 0x30C
DP_HOST_MAC_LOW = 0x310
DP_HOST_MAC_HIGH = 0x314
DP_HOST_IP = 0x318
DP_HOST_UDP_PORT = 0x31C
DP_VIP_MASK = 0x324

# Fields in DP_ROCE_CFG
# "31:28 = end buf
#  27:24 = start buf
#  23: 0 = qp"
DP_ROCE_CFG = 0x1000
DP_ROCE_RKEY_0 = 0x1004
DP_ROCE_VADDR_MSB_0 = 0x1008
DP_ROCE_VADDR_LSB_0 = 0x100C
DP_ROCE_BUF_END_MSB_0 = 0x1010
DP_ROCE_BUF_END_LSB_0 = 0x1014


class HololinkDataChannel:
    def __init__(self, metadata):
        cpnx_version_s = metadata.get("cpnx_version")  # or None
        if cpnx_version_s is None:
            raise UnsupportedVersion("No 'cpnx_version' field found.")
        cpnx_version = int(cpnx_version_s)
        if cpnx_version < MINIMUM_CPNX_VERSION:
            raise UnsupportedVersion(
                "cpnx_version=0x%X; minimum supported version=0x%X."
                % (cpnx_version, MINIMUM_CPNX_VERSION)
            )
        self._hololink = Hololink.from_enumeration_metadata(metadata)
        self._address = metadata["configuration_address"]
        self._peer_ip = metadata["peer_ip"]
        self._vip_mask = metadata["vip_mask"]
        self._qp_number = 0
        self._rkey = 0

    @staticmethod
    def _enumerated(metadata):
        if "configuration_address" not in metadata:
            return False
        if "peer_ip" not in metadata:
            return False
        return Hololink._enumerated(metadata)

    def hololink(self):
        return self._hololink

    def peer_ip(self):
        return self._peer_ip

    def authenticate(self, qp_number, rkey):
        self._qp_number = qp_number
        self._rkey = rkey

    def configure(self, frame_address, frame_size, local_data_port):
        header_size = 78
        cache_size = 128
        mtu = 1472  # TCP/IP illustrated vol 1 (1994), section 11.6, page 151
        payload_size = ((mtu - header_size + cache_size - 1) // cache_size) * cache_size
        logging.info(f"{header_size=} {payload_size=}")
        peer_ip = self.peer_ip()
        local_ip, local_device, local_mac = Hololink._local_ip_and_mac(peer_ip)
        return self._configure(
            frame_size,
            payload_size,
            header_size,
            local_mac,
            local_ip,
            local_data_port,
            self._qp_number,
            self._rkey,
            frame_address,
            frame_size,
        )

    def _configure(
        self,
        frame_size,
        payload_size,
        header_size,
        local_mac,
        local_ip,
        local_data_port,
        # roce header contents
        qp_number,
        rkey,
        address,
        size,
    ):
        # This is for FPGA 0116 in classic data plane mode
        mac_high = (local_mac[0] << 8) | (local_mac[1] << 0)
        mac_low = (
            (local_mac[2] << 24)
            | (local_mac[3] << 16)
            | (local_mac[4] << 8)
            | (local_mac[5] << 0)
        )
        bip = [int(s) for s in local_ip.split(".")]
        ip = (bip[0] << 24) | (bip[1] << 16) | (bip[2] << 8) | (bip[3] << 0)
        # Clearing DP_VIP_MASK should be unnecessary-- we should only
        # be here following a reset, but be defensive and make
        # sure we're not transmitting anything while we update.
        self.write_uint32(DP_VIP_MASK, 0)
        self.write_uint32(DP_PACKET_SIZE, header_size + payload_size)
        self.write_uint32(DP_HOST_MAC_LOW, mac_low)
        self.write_uint32(DP_HOST_MAC_HIGH, mac_high)
        self.write_uint32(DP_HOST_IP, ip)
        self.write_uint32(DP_HOST_UDP_PORT, local_data_port)
        #
        # "31:28 = end buf
        #  27:24 = start buf
        #  23: 0 = qp"
        # Only use DMA descriptor ("buf") 0.
        # We write the same addressing information into both VPs for
        # this ethernet port; DP_VIP_MASK from the map above selects
        # which one of these is actually used in the hardware.
        for vp in VP_START:
            self.write_uint32(DP_ROCE_CFG + vp, qp_number & 0x00FF_FFFF)
            self.write_uint32(DP_ROCE_RKEY_0 + vp, rkey)
            self.write_uint32(DP_ROCE_VADDR_MSB_0 + vp, (address >> 32))
            self.write_uint32(DP_ROCE_VADDR_LSB_0 + vp, (address & 0xFFFF_FFFF))
            self.write_uint32(DP_ROCE_BUF_END_MSB_0 + vp, ((address + size) >> 32))
            self.write_uint32(
                DP_ROCE_BUF_END_LSB_0 + vp, ((address + size) & 0xFFFF_FFFF)
            )
        # 0x1 meaning to connect sensor 1 to the current ethernet port
        self.write_uint32(DP_VIP_MASK, self._vip_mask)

    def write_uint32(self, reg, value):
        return self._hololink.write_uint32(self._address + reg, value)


class HololinkTimeoutError(RuntimeError):
    pass


class NotFoundException(Exception):
    pass


class UnsupportedVersion(Exception):
    pass


hololink_by_serial_number = {}


class Hololink:
    SIOCGIFHWADDR = 0x8927

    @staticmethod
    def _enumerated(metadata):
        if "serial_number" not in metadata:
            return False
        if "peer_ip" not in metadata:
            return False
        if "control_port" not in metadata:
            return False
        if "hololink_class" not in metadata:
            return False
        return True

    @staticmethod
    def from_enumeration_metadata(metadata):
        serial_number = metadata["serial_number"]
        global hololink_by_serial_number
        r = hololink_by_serial_number.get(serial_number, None)
        if r is None:
            peer_ip = metadata["peer_ip"]
            control_port = metadata["control_port"]
            hololink_class = metadata["hololink_class"]
            r = hololink_class(peer_ip, control_port, serial_number)
            hololink_by_serial_number[serial_number] = r
        return r

    @staticmethod
    def _reset_framework():
        global hololink_by_serial_number
        keys = list(hololink_by_serial_number.keys())
        for key in keys:
            logging.info(f'Removing hololink "{key}"')
            del hololink_by_serial_number[key]

    def __init__(self, peer_ip, control_port, serial_number):
        self._peer_ip = peer_ip
        self._control_port = control_port
        self._serial_number = serial_number
        self._sequence = 0x100
        self._i2cs = {}

    def error(self, message):
        logging.error('Error "%s".' % (message,))
        raise RuntimeError(message)

    def timeout(self, message):
        logging.error('Timeout: "%s".' % (message,))
        raise HololinkTimeoutError(message)

    def csi_size(self):
        frame_start_size = 8
        frame_end_size = 8
        line_start_size = 8
        line_end_size = 8
        return (frame_start_size, frame_end_size, line_start_size, line_end_size)

    def start(self):
        self._control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._version = self.get_fpga_version()
        self._datecode = self.get_fpga_date()
        logging.info(f"FPGA version={self._version:#x} datecode={self._datecode:#x}")

    def stop(self):
        self._control_socket.close()

    def reset(self):
        spi = self.get_spi(
            CLNX_SPI_CTRL, chip_select=0, clock_divisor=15, cpol=0, cpha=1, width=1
        )
        #
        write_command_bytes = [0x01, 0x07]
        write_data_bytes = [0x0C]
        read_byte_count = 0
        spi.spi_transaction(write_command_bytes, write_data_bytes, read_byte_count)
        #
        self.write_uint32(0x8, 0)
        time.sleep(0.1)
        #
        write_data_bytes = [0x0F]
        spi.spi_transaction(write_command_bytes, write_data_bytes, read_byte_count)
        time.sleep(0.1)
        #
        self.write_uint32(0x8, 0x3)
        try:
            # Because this drives the unit to reset,
            # we won't get a reply.
            self.write_uint32(0x4, 0x8, retry=False)
        except RuntimeError as e:
            logging.info(f"ignoring error {e=}.")
        # Now wait for the device to come back up.
        # This guy raises an exception if we're not found;
        # this can happen if set-ip is used in one-time
        # mode.
        enumerator = HololinkEnumerator()
        for packet, metadata in enumerator.enumeration_packets(timeout_s=30):
            logging.trace(f"Enumeration {metadata=}")
            peer_ip = metadata.get("peer_ip")
            if peer_ip == self._peer_ip:
                break
            serial_number = metadata.get("serial_number")
            if serial_number == self._serial_number:
                logging.warning(f"{serial_number=} appears at {peer_ip=}; ignoring.")
        else:
            # We get here if "break" didn't get us out above
            raise NotFoundException(f"device_ip={self._peer_ip} not found.")
        version = self.get_fpga_version()
        logging.info(f"{version=:#x}")

    def get_fpga_version(self):
        version = self.read_uint32(FPGA_VERSION)
        return version

    def get_fpga_date(self):
        date = self.read_uint32(FPGA_DATE)
        return date

    def write_uint32(self, address, value, timeout=None, retry=True):
        count = 0
        timeout = Timeout.configure(timeout)
        try:
            while True:
                count += 1
                status = self._write_uint32(
                    address, value, timeout, response_expected=retry
                )
                if status:
                    return status
                if not retry:
                    break
                if not timeout.retry():
                    return self.timeout(
                        "write_uint32 address=0x%X value=0x%X" % (address, value)
                    )
        finally:
            assert count > 0
            self._add_write_retries(count - 1)

    def _write_uint32(self, address, value, timeout, response_expected=True):
        logging.debug(f"_write_uint32({address=:#x}, {value=:#x})")
        assert (address & 3) == 0
        # BLOCKING on ack or timeout
        # This routine serializes a write_uint32 request
        # and forwards it to the device.
        sequence = self._next_sequence()
        # Serialize
        request = bytearray(20)
        serializer = Serializer(request)
        if not (
            serializer.append_uint8(WR_DWORD)
            and serializer.append_uint8(REQUEST_FLAGS_ACK_REQUEST)
            and serializer.append_uint16_be(sequence)
            and serializer.append_uint8(0)  # reserved \
            and serializer.append_uint8(0)  # reserved \
            and serializer.append_uint32_be(address)
            and serializer.append_uint32_be(value)
        ):
            return self.error("Unable to serialize")
        reply = bytearray(20)
        status, response_code, _ = self._execute(
            sequence, request[: serializer.length()], reply, timeout
        )
        if status is None:
            # timed out
            return False
        if response_code != RESPONSE_SUCCESS:
            if response_code is None:
                if response_expected:
                    logging.error(
                        f"write_uint32 {address=:#X} {value=:#X} response_code=None"
                    )
                return False
            return self.error(
                f"write_uint32 {address=:#X} {value=:#X} {response_code=:#X}"
            )
        return True

    def read_uint32(self, address, timeout=None):
        """Returns the value found at the location
        or calls hololink timeout if there's a problem."""
        count = 0
        timeout = Timeout.configure(timeout)
        try:
            while True:
                count += 1
                status, value = self._read_uint32(address, timeout)
                if status:
                    return value
                if not timeout.retry():
                    return self.timeout("read_uint32 address=0x%X" % (address,))
        finally:
            assert count > 0
            self._add_read_retries(count - 1)

    def _read_uint32(self, address, timeout):
        assert (address & 3) == 0
        logging.trace(f"_read_uint32({address=:#x})")
        # BLOCKING on ack or timeout
        # This routine serializes a read_uint32 request
        # and forwards it to the device.
        sequence = self._next_sequence()
        # Serialize
        request = bytearray(20)
        serializer = Serializer(request)
        if not (
            serializer.append_uint8(RD_DWORD)
            and serializer.append_uint8(REQUEST_FLAGS_ACK_REQUEST)
            and serializer.append_uint16_be(sequence)
            and serializer.append_uint8(0)  # reserved
            and serializer.append_uint8(0)  # reserved
            and serializer.append_uint32_be(address)
        ):
            return self.error("Unable to serialize")
        logging.trace(f"read_uint32: {request[: serializer.length()]}....{sequence}")
        reply = bytearray(20)
        status, response_code, deserializer = self._execute(
            sequence, request[: serializer.length()], reply, timeout
        )
        if status is not True:
            return False, None
        if response_code != RESPONSE_SUCCESS:
            self.error("read_uint32 response_code=0x%X" % (response_code,))
            return False, None
        deserializer.next_uint8()  # reserved
        response_address = deserializer.next_uint32_be()  # address
        assert response_address == address
        value = deserializer.next_uint32_be()
        logging.debug(f"read_uint32({address=:#x})={value:#x}")
        return True, value

    def _next_sequence(self):
        r = self._sequence
        # Sequence is a 16-bit value.
        self._sequence = (self._sequence + 1) & 0xFFFF
        return r

    def _execute(self, sequence, request, reply, timeout):
        logging.trace(f"Sending {request=}")
        request_time = Timeout.now()
        self._send_control(request)
        while True:
            reply = self._receive_control(timeout)
            reply_time = Timeout.now()
            self._executed(request_time, request, reply_time, reply)
            if reply is None:
                # timed out
                return False, None, None
            deserializer = Deserializer(reply)
            deserializer.next_uint8()  # reply_cmd_code
            deserializer.next_uint8()  # reply_flags
            reply_sequence = deserializer.next_uint16_be()
            response_code = deserializer.next_uint8()
            logging.trace(f"reply {reply_sequence=} {response_code=} {sequence=}")
            if sequence == reply_sequence:
                return True, response_code, deserializer

    def _send_control(self, request):
        logging.trace(
            f"_send_control {request=} peer_ip={self._peer_ip} control_port={self._control_port}"
        )
        self._control_socket.sendto(request, (self._peer_ip, self._control_port))

    def _receive_control(self, timeout):
        while True:
            if timeout.expired():
                logging.debug("Timed out.")
                return None
            timeout_s = max(timeout.trigger_s(), 0)
            r = [self._control_socket]
            w = []
            x = []
            r, w, x = select.select(r, w, x, timeout_s)
            assert len(w) == 0
            assert len(x) == 0
            if self._control_socket in r:
                received, peer_address = self._control_socket.recvfrom(8192)
                return received

    def _executed(self, request_time, request, reply_time, reply):
        # Override this guy to record timing around ACKs etc
        logging.trace("Got reply=%s" % (reply,))

    def _add_read_retries(self, n):
        pass

    def _add_write_retries(self, n):
        pass

    @staticmethod
    def _local_ip_and_mac(destination_ip, port=1):
        # Works only on Linux.
        # @returns our IP address that can connect to /address/ and the MAC ID
        # for that interface.
        # We need a port number for the connect call to work, but because it's
        # SOCK_DGRAM, there's no actual traffic sent.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # Start with a map of IP address to interfaces.
            interface_by_ip = {}
            SIOCGIFCONF = 0x8912
            # First, find out how many interfaces there are.
            request = struct.pack("iP", 0, 0)
            reply = fcntl.ioctl(s.fileno(), SIOCGIFCONF, request)
            ifreq_buffer_size, p = struct.unpack("iP", reply)
            assert ifreq_buffer_size > 0
            #
            ifreq_buffer = array.array("B", b"\0" * ifreq_buffer_size)
            ifreq_buffer_address, _ifreq_buffer_size = ifreq_buffer.buffer_info()
            assert ifreq_buffer_size == _ifreq_buffer_size
            request = struct.pack("iP", ifreq_buffer_size, ifreq_buffer_address)
            reply = fcntl.ioctl(s.fileno(), SIOCGIFCONF, request)
            reply_size, reply_address = struct.unpack("iP", reply)
            assert reply_size == ifreq_buffer_size
            assert reply_address == ifreq_buffer_address
            # Now ifreq_buffer has our ifreqs in it.
            IFREQ_SIZE = 40
            IFNAMSIZ = 16
            IP = 20
            IPv4_LEN = 4
            ifreqs = ifreq_buffer.tobytes()
            for i in range(0, ifreq_buffer_size, IFREQ_SIZE):
                name = ifreqs[i : i + IFNAMSIZ].strip(b"\0")
                bip = ifreqs[i + IP : i + IP + IPv4_LEN]
                ip = ".".join(["%d" % x for x in bip])
                logging.trace(f"{name=} {ip=}")
                interface_by_ip[ip] = name
            # datagram sockets, when connected, will only
            # do I/O with the address they're connected to.
            # Once it's connected, getsockname() will tell
            # us the IP we're using on our side.
            s.settimeout(0)
            s.connect((destination_ip, port))
            ip = s.getsockname()[0]
            binterface = interface_by_ip[ip]
            info = fcntl.ioctl(
                s.fileno(), Hololink.SIOCGIFHWADDR, struct.pack("256s", binterface[:15])
            )
            MAC_ID = 18
            MAC_ID_SIZE = 6
            mac_id = info[MAC_ID : MAC_ID + MAC_ID_SIZE]
            mac_id_string = ":".join(["%02X" % x for x in mac_id])
            logging.debug(f"{destination_ip=} local_ip={ip} mac_id={mac_id_string}")
            return ip, binterface.decode(), mac_id

    @staticmethod
    def _local_mac(interface):
        # Works only on Linux.
        # @returns local_ip, local_mac_id for the given interface by name.
        binterface = interface.encode()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            info = fcntl.ioctl(
                s.fileno(), Hololink.SIOCGIFHWADDR, struct.pack("256s", binterface[:15])
            )
            MAC_ID = 18
            MAC_ID_SIZE = 6
            mac_id = info[MAC_ID : MAC_ID + MAC_ID_SIZE]
            mac_id_string = ":".join(["%02X" % x for x in mac_id])
            logging.debug(f"{interface=} mac_id={mac_id_string}")
            return mac_id

    def _write_renesas(self, i2c, data):
        logging.trace(
            "write_renesas(data=%s)" % (",".join(["0x%02X" % x for x in data]),)
        )
        read_byte_count = 0
        RENESAS_I2C_ADDRESS = 0x09
        reply = i2c.i2c_transaction(RENESAS_I2C_ADDRESS, data, read_byte_count)
        logging.trace("reply=%s." % (reply,))

    def setup_clock(self, clock_profile):
        # set the clock driver.
        i2c = self.get_i2c(BL_I2C_CTRL)
        i2c.set_i2c_clock()
        time.sleep(0.1)
        #
        for data in clock_profile.device_configuration():
            self._write_renesas(i2c, data)
        time.sleep(0.1)
        # enable the clock synthesizer and output
        self.write_uint32(0x8, 0x30)
        time.sleep(0.1)
        # enable camera power.
        self.write_uint32(0x8, 0x03)
        time.sleep(0.1)
        i2c.set_i2c_clock()

    def get_i2c(self, i2c_address):
        r = Hololink.I2c(self, i2c_address)
        return r

    def get_spi(
        self, spi_address, chip_select, clock_divisor=0x0F, cpol=1, cpha=1, width=1
    ):
        assert 0 <= clock_divisor < 16
        assert 0 <= chip_select < 8
        width_map = {
            1: 0,
            2: 2 << 8,
            4: 3 << 8,
        }
        # we let this next statement raise an
        # exception if the width parameter isn't
        # supported.
        spi_cfg = clock_divisor | (chip_select << 12) | width_map[width]
        if cpol:
            spi_cfg |= SPI_CFG_CPOL
        if cpha:
            spi_cfg |= SPI_CFG_CPHA
        return Hololink.Spi(self, spi_address, spi_cfg)

    class I2c:
        def __init__(self, hololink, i2c_address):
            self._hololink = hololink
            self._reg_control = i2c_address + 0
            self._reg_num_bytes = i2c_address + 4
            self._reg_clk_ctrl = i2c_address + 8
            self._reg_data_buffer = i2c_address + 16

        def set_i2c_clock(self):
            # set the clock to 400KHz (fastmode) i2c speed once at init
            clock = 0b0000_0101
            self._hololink.write_uint32(self._reg_clk_ctrl, clock, retry())

        def i2c_transaction(
            self,
            peripheral_i2c_address,
            write_bytes,
            read_byte_count,
            timeout=None,
        ):
            logging.debug(
                "i2c_transaction peripheral=0x%X len(write_bytes)=%d read_byte_count=%d"
                % (peripheral_i2c_address, len(write_bytes), read_byte_count)
            )
            assert peripheral_i2c_address < 0x80
            write_byte_count = len(write_bytes)
            assert write_byte_count < 0x100
            assert read_byte_count < 0x100
            timeout = Timeout.configure_i2c(timeout)
            # Hololink FPGA doesn't support resetting the I2C interface;
            # so the best we can do is make sure it's not busy.
            value = self._hololink.read_uint32(self._reg_control, timeout)
            assert (value & I2C_BUSY) == 0
            #
            # set the device address and enable the i2c controller
            # I2C_DONE_CLEAR -> 1
            control = (peripheral_i2c_address << 16) | I2C_CORE_EN | I2C_DONE_CLEAR
            self._hololink.write_uint32(self._reg_control, control, timeout)
            # I2C_DONE_CLEAR -> 0
            control = (peripheral_i2c_address << 16) | I2C_CORE_EN
            self._hololink.write_uint32(self._reg_control, control, timeout)
            # make sure DONE is 0.
            value = self._hololink.read_uint32(self._reg_control, timeout)
            logging.debug("control value=0x%X" % (value,))
            assert (value & I2C_DONE) == 0
            # write the num_bytes
            num_bytes = (write_byte_count << 0) | (read_byte_count << 8)
            self._hololink.write_uint32(self._reg_num_bytes, num_bytes, timeout)
            for n, v in enumerate(u32s(write_bytes)):
                # write the register and its value
                self._hololink.write_uint32(self._reg_data_buffer + (n * 4), v, timeout)
            while True:
                # start i2c transaction.
                control = (peripheral_i2c_address << 16) | I2C_CORE_EN | I2C_START
                self._hololink.write_uint32(self._reg_control, control, timeout)
                # retry if we don't see BUSY or DONE
                value = self._hololink.read_uint32(self._reg_control, timeout)
                if value & (I2C_DONE | I2C_BUSY):
                    break
                if not timeout.retry():
                    # timed out
                    logging.debug("Timed out.")
                    return self._hololink.timeout(
                        "i2c_transaction i2c_address=0x%X" % (peripheral_i2c_address,)
                    )
            # Poll until done.  Future version will have an event packet too.
            while True:
                value = self._hololink.read_uint32(self._reg_control, timeout)
                logging.trace("control=0x%X." % (value,))
                done = value & I2C_DONE
                if done != 0:
                    break
                if not timeout.retry():
                    # timed out
                    logging.debug("Timed out.")
                    return self._hololink.timeout(
                        "i2c_transaction i2c_address=0x%X" % (peripheral_i2c_address,)
                    )
            # round up to get the whole next word
            word_count = (read_byte_count + 3) // 4
            r = bytearray(word_count * 4)
            for i in range(word_count):
                value = self._hololink.read_uint32(
                    self._reg_data_buffer + (i * 4), timeout
                )
                r[i * 4 : (i + 1) * 4] = [
                    (value >> 0) & 0xFF,
                    (value >> 8) & 0xFF,
                    (value >> 16) & 0xFF,
                    (value >> 24) & 0xFF,
                ]
            return r[:read_byte_count]

    class Spi:
        def __init__(self, hololink, spi_address, spi_cfg):
            self._hololink = hololink
            self._reg_control = spi_address + 0
            self._reg_num_bytes = spi_address + 4
            self._reg_spi_cfg = spi_address + 8
            self._reg_num_bytes2 = spi_address + 12
            self._reg_data_buffer = spi_address + 16
            self._spi_cfg = spi_cfg
            self._turnaround_cycles = 0

        def spi_transaction(
            self,
            write_command_bytes,
            write_data_bytes,
            read_byte_count,
            timeout=None,
        ):
            write_bytes = bytearray(write_command_bytes)
            write_bytes.extend(write_data_bytes)
            write_command_count = len(write_command_bytes)
            assert write_command_count < 16  # available bits in num_bytes2
            write_byte_count = len(write_bytes)
            buffer_size = 288
            # Because the controller always records ingress data,
            # whether we're transmitting or receiving, we get a copy
            # of the written data in the buffer on completion--
            # which means the buffer has to have enough space for
            # both the egress and ingress data.
            buffer_count = write_byte_count + read_byte_count
            assert buffer_count < buffer_size
            timeout = Timeout.configure_spi(timeout)
            # Hololink FPGA doesn't support resetting the SPI interface;
            # so the best we can do is see that it's not busy.
            value = self._hololink.read_uint32(self._reg_control, timeout)
            assert (value & SPI_BUSY) == 0
            # Set the configuration
            self._hololink.write_uint32(self._reg_spi_cfg, self._spi_cfg, timeout)
            for n, v in enumerate(u32s(write_bytes)):
                self._hololink.write_uint32(self._reg_data_buffer + (n * 4), v, timeout)
            # write the num_bytes; note that these are 9-bit values that top
            # out at (buffer_size=288) (length checked above)
            num_bytes = (write_byte_count << 0) | (read_byte_count << 16)
            self._hololink.write_uint32(self._reg_num_bytes, num_bytes, timeout)
            assert 0 <= self._turnaround_cycles < 16
            num_bytes2 = self._turnaround_cycles | (write_command_count << 8)
            self._hololink.write_uint32(self._reg_num_bytes2, num_bytes2, timeout)
            # start the SPI transaction.  don't retry this guy; just raise
            # an error if we don't see the ack.
            control = SPI_START
            status = self._hololink.write_uint32(
                self._reg_control, control, timeout, retry=False
            )
            if not status:
                return self._hololink.error(
                    "ACK failure writing to SPI control register 0x%X."
                    % (self._reg_control,)
                )
            # wait until we don't see busy, which may be immediately
            while True:
                value = self._hololink.read_uint32(self._reg_control, timeout)
                busy = value & SPI_BUSY
                if busy == 0:
                    break
                if not timeout.retry():
                    # timed out
                    logging.debug("Timed out.")
                    return self._hololink.timeout(
                        "spi_transaction control=0x%X" % (self._reg_control,)
                    )
            # round up to get the whole next word
            r = bytearray(buffer_count + 3)
            # no need to re-read the transmitted data
            start_byte_offset = write_byte_count
            # but we can only read words; so back up to the word boundary
            start_byte_offset &= ~3
            for i in range(start_byte_offset, buffer_count, 4):
                value = self._hololink.read_uint32(self._reg_data_buffer + i, timeout)
                r[i : i + 4] = [
                    (value >> 0) & 0xFF,
                    (value >> 8) & 0xFF,
                    (value >> 16) & 0xFF,
                    (value >> 24) & 0xFF,
                ]
            # skip over the data that we wrote out.
            return r[
                write_byte_count : write_byte_count + read_byte_count
            ]  # noqa: E203


class HololinkEnumerator:
    """Handle device discovery and IP address assignment."""

    IP_PKTINFO = 8

    def __init__(
        self,
        local_interface="",  # blank for all local interfaces
        enumeration_port=10001,
        bootp_request_port=12267,
        bootp_reply_port=12268,
    ):
        self._local_interface = local_interface
        self._enumeration_port = enumeration_port
        self._bootp_request_port = bootp_request_port
        self._bootp_reply_port = bootp_reply_port
        self._ip_map = {}

    def enumeration_packets(self, timeout_s=None):
        """Yields out a pair of (packet, metadata) for each
        received enumeration or bootp request packet.  This
        function terminates after the given timeout_s; set that to None
        to run forever.
        """
        self._enumeration_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._bootp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        deadline = None
        if timeout_s is not None:
            deadline = Timeout.now() + timeout_s
        try:
            # Allow other programs to receive these broadcast packets.
            self._enumeration_socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEPORT, 1
            )
            self._enumeration_socket.bind(
                (self._local_interface, self._enumeration_port)
            )
            # Tell us what interface the request came in on.
            self._enumeration_socket.setsockopt(
                socket.SOL_IP, HololinkEnumerator.IP_PKTINFO, 1
            )
            self._bootp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            self._bootp_socket.bind((self._local_interface, self._bootp_request_port))
            # Tell us what interface the request came in on, so we reply to the same place
            self._bootp_socket.setsockopt(
                socket.SOL_IP, HololinkEnumerator.IP_PKTINFO, 1
            )
            receive_message_size = 8192
            while True:
                r = [self._enumeration_socket, self._bootp_socket]
                w = []
                x = [self._enumeration_socket, self._bootp_socket]
                timeout = None
                if deadline is not None:
                    timeout = deadline - Timeout.now()
                    if timeout < 0:
                        return
                r, w, x = select.select(r, w, x, timeout)
                assert len(w) == 0
                if len(x) != 0:
                    logging.error("Error reading enumeration sockets.")
                    return
                if len(r) == 0:
                    # Timed out.
                    return
                if self._enumeration_socket in r:
                    (
                        packet,
                        ancdata,
                        msg_flags,
                        peer_address,
                    ) = self._enumeration_socket.recvmsg(
                        receive_message_size, socket.CMSG_LEN(receive_message_size)
                    )
                    logging.trace(
                        f"enumeration {peer_address=} {len(ancdata)=} {msg_flags=:#X} {len(packet)=}"
                    )
                    metadata = {
                        "peer_ip": peer_address[0],
                        "source_port": peer_address[1],
                    }
                    self._deserialize_ancdata(ancdata, metadata)
                    self._deserialize_enumeration(packet, metadata)
                    yield packet, metadata
                if self._bootp_socket in r:
                    (
                        packet,
                        ancdata,
                        msg_flags,
                        peer_address,
                    ) = self._bootp_socket.recvmsg(
                        receive_message_size, socket.CMSG_LEN(receive_message_size)
                    )
                    logging.trace(
                        f"bootp {peer_address=} {len(ancdata)=} {msg_flags=:#X} {len(packet)=}"
                    )
                    metadata = {
                        "peer_ip": peer_address[0],
                        "source_port": peer_address[1],
                    }
                    self._deserialize_ancdata(ancdata, metadata)
                    self._deserialize_bootp_request(packet, metadata)
                    yield packet, metadata
        finally:
            self._enumeration_socket.close()
            self._bootp_socket.close()

    def _deserialize_ancdata(self, ancdata, metadata):
        for anc in ancdata:
            cmsg_level, cmsg_type, cmsg_data = anc
            logging.trace(f"{cmsg_level=} {cmsg_type=} {len(cmsg_data)=}")
            if (cmsg_level == socket.SOL_IP) and (
                cmsg_type == HololinkEnumerator.IP_PKTINFO
            ):
                ipi_ifindex, ipi_spec_dst, ipi_addr = struct.unpack("I4s4s", cmsg_data)
                interface = socket.if_indextoname(ipi_ifindex)
                logging.trace(
                    f"{ipi_ifindex=} {interface=} {ipi_spec_dst=} {ipi_addr=}"
                )
                metadata.update(
                    {
                        "interface_index": ipi_ifindex,
                        "interface": interface,
                        "interface_address": socket.inet_ntoa(ipi_spec_dst),
                        "destination_address": socket.inet_ntoa(ipi_addr),
                    }
                )

    BOARD_IDS = {
        1: "hololink-lite",
        2: "hololink",
    }

    def _deserialize_enumeration(self, packet, metadata):
        deserializer = Deserializer(packet)
        board_id = deserializer.next_uint8()
        if board_id == HOLOLINK_LITE_BOARD_ID:
            board_version = deserializer.next_buffer(20)
            _serial_number = deserializer.next_buffer(7)
            serial_number = "".join(["%02X" % x for x in _serial_number])
            cpnx_version = deserializer.next_uint16_le()
            cpnx_crc = deserializer.next_uint16_le()
            clnx_version = deserializer.next_uint16_le()
            clnx_crc = deserializer.next_uint16_le()
            default_control_port = 8192
            metadata.update(
                {
                    "type": "enumeration",
                    "board_id": board_id,
                    "board_description": HololinkEnumerator.BOARD_IDS.get(
                        board_id, "N/A"
                    ),
                    "board_version": ["%02X" % x for x in board_version],
                    "serial_number": serial_number,
                    "cpnx_version": cpnx_version,
                    "cpnx_crc": cpnx_crc,
                    "clnx_version": clnx_version,
                    "clnx_crc": clnx_crc,
                    "control_port": default_control_port,
                    "hololink_class": Hololink,
                }
            )

    def _deserialize_bootp_request(self, packet, metadata):
        deserializer = Deserializer(packet)
        op = deserializer.next_uint8()
        hardware_type = deserializer.next_uint8()
        hardware_address_length = deserializer.next_uint8()
        hops = deserializer.next_uint8()
        transaction_id = deserializer.next_uint32_be()
        seconds = deserializer.next_uint16_be()
        flags = deserializer.next_uint16_be()
        client_ip_address = deserializer.next_uint32_be()  # current IP address
        your_ip_address = (
            deserializer.next_uint32_be()
        )  # host IP that assigned the IP address
        server_ip_address = deserializer.next_uint32_be()  # expected to be 0s
        gateway_ip_address = deserializer.next_uint32_be()
        hardware_address = deserializer.next_buffer(16)
        mac_id = ":".join(
            ["%02X" % x for x in hardware_address[:hardware_address_length]]
        )
        _ = deserializer.next_buffer(64)  # server_hostname
        _ = deserializer.next_buffer(128)  # boot_filename
        vendor_information = deserializer.next_buffer(64)
        metadata.update(
            {
                "type": "bootp_request",
                "op": op,
                "hardware_type": hardware_type,
                "hardware_address_length": hardware_address_length,
                "hops": hops,
                "transaction_id": transaction_id,
                "seconds": seconds,
                "flags": flags,
                "client_ip_address": socket.inet_ntoa(
                    struct.pack("!I", client_ip_address)
                ),
                "your_ip_address": socket.inet_ntoa(struct.pack("!I", your_ip_address)),
                "server_ip_address": socket.inet_ntoa(
                    struct.pack("!I", server_ip_address)
                ),
                "gateway_ip_address": socket.inet_ntoa(
                    struct.pack("!I", gateway_ip_address)
                ),
                "hardware_address": hardware_address[:hardware_address_length],
                "mac_id": mac_id,
                "vendor_information": vendor_information,
            }
        )

    def set_ips(self, ips_by_mac, timeout_s=None, one_time=False):
        # Runs forever if "timeout_s" is None
        # Make sure the MAC IDs are all upper case.
        by_mac = {k.upper(): v for k, v in ips_by_mac.items()}
        reported = {}
        #
        for packet, metadata in self.enumeration_packets(timeout_s):
            logging.trace(f"Enumeration {metadata=}")
            mac_id = metadata.get("mac_id")  # Note that this value is upper case
            peer_ip = metadata.get("peer_ip")
            new_peer_ip = by_mac.get(mac_id)
            if new_peer_ip is None:
                # This is true if any of (mac_id, peer_ip, new_peer_ip) aren't provided.
                continue
            if new_peer_ip == peer_ip:
                if not reported.get(mac_id):
                    logging.info(f"Found {mac_id=} found using {peer_ip=}")
                    reported[mac_id] = True
                # We're good.
                if one_time:
                    if len(reported) == len(by_mac):
                        return
                continue
            # At this point, let's update that thing.
            # Set our local ARP cache so that we don't generate
            # an ARP request to the new IP-- the client doesn't
            # know it's IP yet so it won't be able to answer.
            local_device = metadata["interface"]
            local_mac = Hololink._local_mac(local_device)
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
                r = ArpWrapper.arp_set(socket_fd, local_device, new_peer_ip, mac_id)
                if r != 0:
                    raise Exception(f"Unable to set IP address, errno={r}.")
            finally:
                s.close()
            # Now send the bootp reply that reconfigures it.
            reply = self._make_bootp_reply(
                metadata,
                new_peer_ip,
                local_ip,
            )
            self._send_bootp_reply(new_peer_ip, reply, metadata["interface_index"])
            if mac_id in reported:
                del reported[mac_id]

    def _make_bootp_reply(self, metadata, new_device_ip, local_ip):
        reply = bytearray(1000)
        serializer = Serializer(reply)
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
        hardware_address = bytearray(16)
        hardware_address_length = metadata["hardware_address_length"]
        hardware_address[:hardware_address_length] = metadata["hardware_address"]
        serializer.append_buffer(hardware_address)
        host_name = bytearray(64)
        serializer.append_buffer(host_name)
        file_name = bytearray(128)
        serializer.append_buffer(file_name)
        vendor_specific = bytearray(64)
        serializer.append_buffer(vendor_specific)
        return serializer.data()

    def _send_bootp_reply(self, peer_ip, reply, interface_index):
        anc = (
            socket.SOL_IP,
            HololinkEnumerator.IP_PKTINFO,
            struct.pack("III", interface_index, 0, 0),
        )
        flags = 0
        address = (peer_ip, self._bootp_reply_port)
        logging.trace(f"{reply=} {anc=} {flags=} {peer_ip=} {self._bootp_reply_port=}")
        self._bootp_socket.sendmsg((reply,), (anc,), flags, address)

    @staticmethod
    def enumerated(timeout_s=None):
        """Yields back a dict for every enumeration message received
        with metadata about the connection.  Note that if data changes,
        e.g. board reset, this routine won't know to invalidate the old
        data; so you may get one or two stale messages around reset.
        """
        enumerator = HololinkEnumerator()
        data_plane_by_peer_ip = {}
        for packet, metadata in enumerator.enumeration_packets(timeout_s):
            logging.debug(f"Enumeration {metadata=}")
            peer_ip = metadata.get("peer_ip")
            if peer_ip is None:
                continue
            channel_metadata = data_plane_by_peer_ip.setdefault(peer_ip, {})
            channel_metadata.update(metadata)
            # transaction_id actually indicates which data plane instance we're talking to
            transaction_id = metadata.get("transaction_id")  # may return None
            logging.trace(f"{transaction_id=}")
            channel_configuration = BOOTP_TRANSACTION_ID_MAP.get(
                transaction_id
            )  # may return None
            if channel_configuration is not None:
                logging.trace(f"{channel_configuration.configuration_address=}")
                channel_metadata["configuration_address"] = (
                    channel_configuration.configuration_address
                )
                channel_metadata["vip_mask"] = channel_configuration.vip_mask
            # Do we have the information we need?
            logging.debug(f"{channel_metadata=}")
            if HololinkDataChannel._enumerated(channel_metadata):
                yield channel_metadata

    @staticmethod
    def find_channel(channel_ip, timeout_s=20):
        for channel_metadata in HololinkEnumerator.enumerated(timeout_s):
            peer_ip = channel_metadata.get("peer_ip")
            if peer_ip == channel_ip:
                return channel_metadata
        # We only get here if we time out.
        s = f"Device with {channel_ip=} not found."
        logging.error(s)
        raise NotFoundException(s)
