# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import collections
import ctypes
import logging
import multiprocessing
import os
import queue
import select
import socket
import struct
import threading
import time
import traceback
import zlib

import mock_camera
import numpy as np
import nvtx
import utils

import hololink as hololink_module

# How much shared memory for the generated image
# should be allocated?  We'll fault if we try to create an
# image that won't fit.  A 4k image is like 32M.
image_size_limit = 64 * 1024 * 1024
image_memory = multiprocessing.Array(ctypes.c_uint8, image_size_limit)
# total-size, height, width, bpp
image_memory_metadata = multiprocessing.Array(ctypes.c_uint32, [0, 0, 0, 0])

DataPlaneConfiguration = collections.namedtuple(
    "DataPlaneConfiguration", ["hif_address"]
)
data_plane_map = {
    0: DataPlaneConfiguration(0x02000300),
    1: DataPlaneConfiguration(0x02010300),
}

SensorConfiguration = collections.namedtuple(
    "SensorConfiguration",
    ["sensor_interface", "vp_mask", "frame_end_event", "sif_address", "vp_address"],
)
sensor_map = {
    0: SensorConfiguration(
        0, 0x01, hololink_module.Hololink.Event.SIF_0_FRAME_END, 0x01000000, 0x1000
    ),
    1: SensorConfiguration(
        2, 0x04, hololink_module.Hololink.Event.SIF_1_FRAME_END, 0x01010000, 0x1080
    ),
}

PAGE_SIZE = 128

MS_PER_SEC = 1000
US_PER_SEC = 1000 * MS_PER_SEC
NS_PER_SEC = 1000 * US_PER_SEC

icrc_initializer = zlib.crc32(bytes([0xFF] * 8))


def generate_image(bayer_height, bayer_width, bayer_format, pixel_format):
    image, bayer_image = utils.make_image(
        bayer_height, bayer_width, bayer_format, pixel_format
    )
    logging.debug("bayer_image=%s bytes" % (bayer_image.nbytes,))
    # Publish our image so that test code can check that
    # we decoded to the same thing.
    global image_memory, image_memory_metadata
    with image_memory.get_lock():
        s = image.ravel()
        length = len(s)
        u = np.frombuffer(image_memory.get_obj(), dtype=np.uint8, count=length)
        np.copyto(u, s)
        image_memory_metadata[0] = length
        image_memory_metadata[1] = image.shape[0]
        image_memory_metadata[2] = image.shape[1]
        image_memory_metadata[3] = image.shape[2]
    # Build up the CSI stream
    frame_start_size = 0
    frame_end_size = 0
    line_start_size = 0
    line_end_size = 0
    frame_start = np.array([255] * frame_start_size, dtype=np.uint8)
    line_start = np.array([255] * line_start_size, dtype=np.uint8)
    line_end = np.array([255] * line_end_size, dtype=np.uint8)
    frame_end = np.array([255] * frame_end_size, dtype=np.uint8)
    csi_image_data = np.concatenate([frame_start])
    # These guys need to be 8-byte aligned.
    assert (len(bayer_image[0]) & 7) == 0
    for line in bayer_image:
        csi_image_data = np.concatenate(
            [
                csi_image_data,
                line_start,
                line,
                line_end,
            ]
        )
    csi_image_data = np.concatenate([csi_image_data, frame_end])
    return csi_image_data


class InfinibandFormatter:
    def __init__(
        self, source_ip, source_port, destination_ip, destination_port, qp, rkey
    ):
        self._source_ip = source_ip
        self._source_port = source_port
        self._destination_ip = destination_ip
        self._destination_port = destination_port
        self._qp = qp
        self._rkey = rkey

    def _format_ip(self, tos, payload_size, ttl, header_checksum):
        ip_size = 20
        ip = struct.pack(
            "!BBHHHBBH4s4s",
            0x45,  # version/header length
            tos,
            ip_size + payload_size,
            0,  # identification, not used without fragmentation
            0x4000,  # don't ever fragment these
            ttl,
            17,  # UDP protocol
            header_checksum,
            self._source_ip,
            self._destination_ip,
        )
        assert len(ip) == ip_size
        return ip

    def _format_udp(self, payload_size, udp_checksum):
        udp_size = 8
        udp = struct.pack(
            "!HHHH",
            self._source_port,
            self._destination_port,
            udp_size + payload_size,
            udp_checksum,
        )
        assert len(udp) == udp_size
        return udp

    def _format_ib_packet(self, ib):
        icrc_size = 4
        udp_payload_size = len(ib) + icrc_size
        udp = self._format_udp(payload_size=udp_payload_size, udp_checksum=0xFFFF)
        ip_payload_size = len(udp) + len(ib) + icrc_size
        ip = self._format_ip(
            tos=0xFF, payload_size=ip_payload_size, ttl=0xFF, header_checksum=0xFFFF
        )
        #
        computed_crc = zlib.crc32(ip, icrc_initializer)
        computed_crc = zlib.crc32(udp, computed_crc)
        computed_crc = zlib.crc32(ib, computed_crc)
        # flip the bytes in the CRC
        icrc = struct.unpack("<I", struct.pack(">I", computed_crc))[0]
        # Now build the actual packet
        udp = self._format_udp(payload_size=udp_payload_size, udp_checksum=0)
        ip = self._format_ip(
            tos=0, payload_size=ip_payload_size, ttl=0x40, header_checksum=0
        )
        packet = struct.pack(
            "!%ds%ds%dsI" % (len(ip), len(udp), len(ib)),
            ip,
            udp,
            ib,
            icrc,
        )
        return packet

    def format_write(self, psn, address, content):
        # form a packet for iCRC calculation
        opcode = 0x2A
        content_size = len(content)
        ib = struct.pack(
            "!BBHIIQII%ds" % content_size,
            opcode,
            0,  # ib flags
            0xFFFF,  # partition
            (0xFF << 24) | self._qp,
            psn,
            address,
            self._rkey,
            content_size,
            content,
            # Don't include iCRC here; _format_ib_packet adds it
        )
        return self._format_ib_packet(ib)

    def format_write_immediate(self, psn, address, content, immediate_value):
        # form a packet for iCRC calculation
        opcode = 0x2B
        content_size = len(content)
        ib = struct.pack(
            "!BBHIIQIII%ds" % content_size,
            opcode,
            0,  # ib flags
            0xFFFF,  # partition
            (0xFF << 24) | self._qp,
            psn,
            address,
            self._rkey,
            content_size,
            immediate_value,
            content,
            # Don't include iCRC here; _format_ib_packet adds it
        )
        return self._format_ib_packet(ib)


class I2c:
    def __init__(self, server, address):
        self._server = server
        self._address = address
        self.memory_write(hololink_module.I2C_REG_STATUS, 0)
        # if we use self.memory_write on this next one then we'll
        # get a callback to set_control, which we're not ready for yet.
        self._server._memory[address + hololink_module.I2C_REG_CONTROL] = 0

    def done(self):
        self.memory_write(hololink_module.I2C_REG_STATUS, hololink_module.I2C_DONE)

    def memory_read(self, reg):
        return self._server.memory_read(self._address + reg)

    def memory_write(self, reg, value):
        self._server.memory_write(self._address + reg, value)

    def set_control(self, value):
        #
        if (value & hololink_module.I2C_START) == 0:
            # Host is clearing the start bit; we also clear the I2C_DONE here
            # Note that it's fine to do this whether I2C_DONE is set or not.
            assert (
                self.memory_read(hololink_module.I2C_REG_STATUS)
                & hololink_module.I2C_BUSY
            ) == 0
            self.memory_write(hololink_module.I2C_REG_STATUS, 0)
            return False
        # We're setting the I2C_START
        assert (
            self.memory_read(hololink_module.I2C_REG_STATUS) & hololink_module.I2C_BUSY
        ) == 0
        self.memory_write(hololink_module.I2C_REG_STATUS, hololink_module.I2C_BUSY)
        # for now we only use 7-bit addresses
        assert (value & hololink_module.I2C_10B_ADDRESS) == 0
        peripheral_i2c_address = (value >> 16) & 0x7F
        assert peripheral_i2c_address == mock_camera.I2C_ADDRESS
        #
        bus_en = self.memory_read(hololink_module.I2C_REG_BUS_EN)
        num_bytes = self.memory_read(hololink_module.I2C_REG_NUM_BYTES)
        write_byte_count = (num_bytes >> 0) & 0x1FF
        read_byte_count = (num_bytes >> 16) & 0x1FF
        logging.debug(
            f"{bus_en=} {peripheral_i2c_address=:#x} {write_byte_count=} {read_byte_count=}"
        )
        # Register reads look like this
        if (write_byte_count == 2) and (read_byte_count == 4):
            # MockCamera wants the MSB of the address at the lowest address
            m = self.memory_read(hololink_module.I2C_REG_DATA_BUFFER)
            register_id_msb = m & 0xFF
            register_id_lsb = (m >> 8) & 0xFF
            register_id = (register_id_msb << 8) | register_id_lsb
            value = self._server.get_i2c_register(register_id)
            value_bytes = [
                (value >> 0) & 0xFF,
                (value >> 8) & 0xFF,
                (value >> 16) & 0xFF,
                (value >> 24) & 0xFF,
            ]
            msb_value = (
                (value_bytes[0] << 24)
                | (value_bytes[1] << 16)
                | (value_bytes[2] << 8)
                | (value_bytes[3] << 0)
            )
            self.memory_write(hololink_module.I2C_REG_DATA_BUFFER, msb_value)
        # Register writes look like this
        elif (write_byte_count == 6) and (read_byte_count == 0):
            # MockCamera wants the MSB of the address at the lowest address
            w0 = self.memory_read(hololink_module.I2C_REG_DATA_BUFFER + 0)
            w1 = self.memory_read(hololink_module.I2C_REG_DATA_BUFFER + 4)
            b = bytearray(
                [
                    (w0 >> 0) & 0xFF,
                    (w0 >> 8) & 0xFF,
                    (w0 >> 16) & 0xFF,
                    (w0 >> 24) & 0xFF,
                    (w1 >> 0) & 0xFF,
                    (w1 >> 8) & 0xFF,
                ]
            )
            deserializer = hololink_module.Deserializer(b)
            register_id = deserializer.next_uint16_be()
            value = deserializer.next_uint32_be()
            self._server.set_i2c_register(register_id, value)
        else:
            assert False and "Unexpected register access."
        return True


class MockServer:
    def __init__(self):
        self._control_udp_port = 8192
        self._data_udp_port = 12288
        self._done = False  # continue the main loop?
        self._control_r, self._control_w = os.pipe()
        self._run = False  # publish video frames?
        self._psn = 0x1000
        self._csi_image_data = None
        self._bayer_height = 0
        self._bayer_width = 0
        self._bayer_format = hololink_module.sensors.csi.BayerFormat.RGGB
        self._pixel_format = hololink_module.sensors.csi.PixelFormat.RAW_8
        self._camera_watchdog_trigger = 0
        self._frame_time_s = 1.0
        self._camera_version = 12344312
        self._hsb_ip_version = 0x2504
        self._fpga_date = 20250528
        self._serial_number = 0xAA55
        self._memory = {
            hololink_module.HSB_IP_VERSION: self._hsb_ip_version,
            hololink_module.FPGA_DATE: self._fpga_date,
        }
        self._data_plane = 0
        self._hif_address = data_plane_map[self._data_plane].hif_address
        self._memory[self._hif_address + hololink_module.DP_PACKET_SIZE] = 0
        self._memory[self._hif_address + hololink_module.DP_VP_MASK] = 0
        self._sensor = 0
        self._vp_address = sensor_map[self._sensor].vp_address
        self._memory[self._vp_address + hololink_module.DP_QP] = 0
        self._memory[self._vp_address + hololink_module.DP_RKEY] = 0
        self._memory[self._vp_address + hololink_module.DP_ADDRESS_0] = 0
        self._memory[self._vp_address + hololink_module.DP_ADDRESS_1] = 0
        self._memory[self._vp_address + hololink_module.DP_ADDRESS_2] = 0
        self._memory[self._vp_address + hololink_module.DP_ADDRESS_3] = 0
        self._memory[self._vp_address + hololink_module.DP_BUFFER_LENGTH] = 0
        self._memory[self._vp_address + hololink_module.DP_BUFFER_MASK] = 0
        self._memory[self._vp_address + hololink_module.DP_HOST_MAC_LOW] = 0
        self._memory[self._vp_address + hololink_module.DP_HOST_MAC_HIGH] = 0
        self._memory[self._vp_address + hololink_module.DP_HOST_IP] = 0
        self._memory[self._vp_address + hololink_module.DP_HOST_UDP_PORT] = 0
        self._sif_address = sensor_map[self._sensor].sif_address
        self._enumeration_metadata = {
            "vp_mask": sensor_map[self._sensor].vp_mask,
            "data_plane": self._data_plane,
            "sensor": self._sensor,
            "sif_address": self._sif_address,
            "vp_address": self._vp_address,
            "hif_address": self._hif_address,
            "frame_end_event": int(sensor_map[self._sensor].frame_end_event),
        }
        self._i2c = I2c(self, hololink_module.I2C_CTRL)
        self._i2c_trigger = None
        self._sequencer_queue = queue.Queue()
        self._sequencer = threading.Thread(target=self.run_sequencer, daemon=True)

    def udp_server(self, lock):
        #
        logging.debug("Starting.")
        self._sequencer.start()
        control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        control_socket.bind(("", self._control_udp_port))
        data_socket = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_RAW)
        data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        data_socket.bind(("", self._data_udp_port))
        message = bytearray(hololink_module.UDP_PACKET_SIZE)
        reply = bytearray(1500)
        frame_number = 0
        next_page = 0
        latched_sequence = 0

        now = time.monotonic()
        # frames come in at this rate.
        frame_trigger = now + self._frame_time_s
        logging.info("Ready.")
        lock.release()
        while not self._done:
            trigger = frame_trigger
            if (self._i2c_trigger is not None) and (self._i2c_trigger < trigger):
                trigger = self._i2c_trigger
            now = time.monotonic()
            timeout = None
            if trigger > now:
                timeout = trigger - now
            cr = [control_socket, self._control_r]
            cw = []
            cx = []
            r, w, x = select.select(cr, cw, cx, timeout)

            with nvtx.annotate("update"):
                now = time.monotonic()

                if self._control_r in r:
                    message = os.read(self._control_r, len(message))
                    logging.trace("got control message=%s" % (message,))
                    if message == b"exit":
                        break

                if control_socket in r:
                    with nvtx.annotate("udp-request"):
                        length, peer = control_socket.recvfrom_into(message)
                        logging.trace('Received "%s".' % (message[:length],))
                        deserializer = hololink_module.Deserializer(message)
                        cmd_code = deserializer.next_uint8()
                        flags = deserializer.next_uint8()
                        send_reply = flags & hololink_module.REQUEST_FLAGS_ACK_REQUEST
                        sequence = deserializer.next_uint16_be()
                        deserializer.next_uint8()  # reserved
                        deserializer.next_uint8()  # reserved
                        #
                        serializer = hololink_module.Serializer(reply)
                        reply_cmd_code = 0x80 | cmd_code
                        serializer.append_uint8(reply_cmd_code)
                        serializer.append_uint8(flags)
                        serializer.append_uint16_be(sequence)
                        if cmd_code == hololink_module.WR_DWORD:
                            address = deserializer.next_uint32_be()
                            value = deserializer.next_uint32_be()
                            status = self.memory_write(address, value)
                            serializer.append_uint8(status)
                            serializer.append_uint8(0)  # reserved; aligns the next data
                            serializer.append_uint32_be(address)
                            serializer.append_uint32_be(value)
                            serializer.append_uint32_be(latched_sequence)
                        elif cmd_code == hololink_module.RD_DWORD:
                            address = deserializer.next_uint32_be()
                            value = self.memory_read(address)
                            status = hololink_module.RESPONSE_SUCCESS
                            serializer.append_uint8(status)
                            serializer.append_uint8(0)  # reserved; aligns the next data
                            serializer.append_uint32_be(address)
                            serializer.append_uint32_be(value)
                            serializer.append_uint32_be(latched_sequence)
                            send_reply = True
                        else:
                            serializer.append_uint8(
                                hololink_module.RESPONSE_INVALID_CMD
                            )
                            send_reply = True
                        if send_reply:
                            control_socket.sendto(reply[: serializer.length()], peer)

                if (self._i2c_trigger is not None) and (now >= self._i2c_trigger):
                    with nvtx.annotate("i2c"):
                        self._i2c_trigger = None
                        self._i2c.done()
                        # NOTE I2C busy -> 0 event here.

                if now >= frame_trigger:
                    with nvtx.annotate("frame-trigger"):
                        frame_trigger += self._frame_time_s
                        if self._run:
                            if now >= self._camera_watchdog_trigger:
                                self._run = False
                                logging.error("Watchdog timeout; stopping frames.")
                        if self._run:
                            timestamp = time.time_ns()
                            # Advance "page", which tells us the host address we're going to write to
                            page_mask = self.memory_read(
                                self._vp_address + hololink_module.DP_BUFFER_MASK
                            )
                            for i in range(32):
                                page = next_page
                                next_page = (next_page + 1) % 32
                                if (page_mask & (1 << page)) == 1:
                                    break
                            else:
                                logging.error("Page mask is 0; skipping this frame.")
                                continue
                            frame_number += 1
                            logging.debug("frame_number=%s" % (frame_number,))
                            #
                            ip_address = self.memory_read(
                                self._vp_address + hololink_module.DP_HOST_IP
                            )
                            destination_ip = [
                                (ip_address >> 24) & 0xFF,
                                (ip_address >> 16) & 0xFF,
                                (ip_address >> 8) & 0xFF,
                                (ip_address >> 0) & 0xFF,
                            ]
                            ip = "%d.%d.%d.%d" % (
                                destination_ip[0],
                                destination_ip[1],
                                destination_ip[2],
                                destination_ip[3],
                            )
                            target_udp_port = self.memory_read(
                                self._vp_address + hololink_module.DP_HOST_UDP_PORT
                            )
                            payload_size = (
                                self.memory_read(
                                    self._hif_address + hololink_module.DP_PACKET_SIZE
                                )
                                * PAGE_SIZE
                            )
                            assert payload_size > 0
                            address_map = {
                                0: hololink_module.DP_ADDRESS_0,
                                1: hololink_module.DP_ADDRESS_1,
                                2: hololink_module.DP_ADDRESS_2,
                                3: hololink_module.DP_ADDRESS_3,
                            }
                            address = self.memory_read(
                                self._vp_address + address_map[page]
                            )
                            address <<= 7
                            qp = self.memory_read(
                                self._vp_address + hololink_module.DP_QP
                            )
                            rkey = self.memory_read(
                                self._vp_address + hololink_module.DP_RKEY
                            )
                            source_ip, source_interface, source_mac = (
                                hololink_module.local_ip_and_mac(ip, target_udp_port)
                            )
                            source_ip = socket.inet_aton(source_ip)
                            source_port = self._data_udp_port
                            formatter = InfinibandFormatter(
                                source_ip,
                                source_port,
                                bytes(destination_ip),
                                target_udp_port,
                                qp,
                                rkey,
                            )
                            with nvtx.annotate("write-frame"):
                                # the last packet is a bit different; don't include that here
                                s, e = 0, payload_size
                                csi_image_length = len(self._csi_image_data)
                                logging.debug(
                                    f"{self._vp_address=:#X} {csi_image_length=} {payload_size=} {ip=} {target_udp_port=}"
                                )
                                while s < csi_image_length:
                                    packet = formatter.format_write(
                                        self._psn,
                                        address + s,
                                        bytes(self._csi_image_data[s:e]),
                                    )
                                    data_socket.sendto(packet, (ip, target_udp_port))
                                    s, e = e, e + payload_size
                                    self._psn += 1
                                assert (page & ~0xFF) == 0
                                immediate_value = (page & 0xFF) | (
                                    (self._psn & 0xFFFFFF) << 8
                                )
                                overall_crc = (
                                    0  # We'll add this when we watermark checking demos
                                )
                                timestamp_s, timestamp_ns = ns_to_sns(timestamp)
                                bytes_written = csi_image_length
                                metadata_timestamp = time.time_ns()
                                metadata_s, metadata_ns = ns_to_sns(metadata_timestamp)
                                metadata = struct.pack(
                                    "!IIIQIQHHQI",
                                    0,  # flags
                                    self._psn,
                                    overall_crc,
                                    timestamp_s,
                                    timestamp_ns,
                                    bytes_written,
                                    0,
                                    frame_number & 0xFFFF,
                                    metadata_s,
                                    metadata_ns,
                                )
                                frame_size = self.memory_read(
                                    self._vp_address + hololink_module.DP_BUFFER_LENGTH
                                )
                                metadata_offset = hololink_module.round_up(
                                    frame_size, PAGE_SIZE
                                )
                                metadata_address = address + metadata_offset
                                metadata_packet = formatter.format_write_immediate(
                                    self._psn,
                                    metadata_address,
                                    metadata,
                                    immediate_value,
                                )
                                data_socket.sendto(
                                    metadata_packet, (ip, target_udp_port)
                                )
                                self._psn += 1

    def memory_write(self, address, value):
        logging.debug("Writing 0x%X to 0x%X." % (value, address))
        self._memory[address] = value
        if address == (hololink_module.I2C_CTRL + hololink_module.I2C_REG_CONTROL):
            if self._i2c.set_control(value):
                # make the host do a bit of polling work here
                # note that we assume that we won't be
                # using both I2Cs at the same time.
                assert self._i2c_trigger is None
                self._i2c_trigger = time.monotonic() + 0.001
        elif (address == hololink_module.CTRL_EVT_SW_EVENT) and (value == 0x1):
            event = hololink_module.Hololink.Event.SW_EVENT
            self._sequencer_queue.put(event)
        return hololink_module.RESPONSE_SUCCESS

    def memory_read(self, address):
        value = self._memory[address]
        logging.debug("Read 0x%X from 0x%X." % (value, address))
        return value

    def get_i2c_register(self, register_id):
        logging.debug("get_i2c_register(register_id=0x%X)" % (register_id,))
        if register_id == mock_camera.VERSION:
            # version ID
            return self._camera_version
        assert False and "get_i2c_register invalid register ID."

    def set_i2c_register(self, register_id, value):
        logging.debug(
            "set_i2c_register(register_id=%d(0x%X), value=%d(0x%X))"
            % (register_id, register_id, value, value)
        )
        if register_id == mock_camera.RESET:
            pass
        elif register_id == mock_camera.WIDTH:
            self._bayer_width = value
        elif register_id == mock_camera.HEIGHT:
            self._bayer_height = value
        elif register_id == mock_camera.BAYER_FORMAT:
            self._bayer_format = hololink_module.sensors.csi.BayerFormat(value)
        elif register_id == mock_camera.PIXEL_FORMAT:
            self._pixel_format = hololink_module.sensors.csi.PixelFormat(value)
        elif register_id == mock_camera.RUN:
            if value == 0:
                self._run = False
            else:
                self._run = True
        elif register_id == mock_camera.WATCHDOG:
            self._camera_watchdog_trigger = time.monotonic() + value
        elif register_id == mock_camera.FRAMES_PER_MINUTE:
            self._frame_time_s = 60.0 / value
            logging.trace(f"{self._frame_time_s=}")
        elif register_id == mock_camera.INITIALIZE:

            self._csi_image_data = generate_image(
                self._bayer_height,
                self._bayer_width,
                self._bayer_format,
                self._pixel_format,
            )
        else:
            assert False and "set_i2c_register invalid register ID."

    def close(self):
        os.write(self._control_w, b"exit")

    def run_sequencer(self):
        DONE = 0xFFFFFFFF
        while True:
            event = self._sequencer_queue.get()
            logging.debug(f"{event=}")
            # Run it.
            vector = hololink_module.APB_RAM + int(event) * 4
            instruction_pointer = self.memory_read(vector)
            command = None
            op_bit = 32

            def next_value():
                nonlocal instruction_pointer
                value = self.memory_read(instruction_pointer)
                logging.debug(f"{instruction_pointer=:#x} {value=:#x}")
                instruction_pointer += 4
                return value

            while True:
                # Notice the order in which memory is arranged--
                # getting a command word, even if it's unused,
                # always comes first.  (It'd be unused if
                # the first instruction was DONE.)
                if op_bit >= 32:
                    command = next_value()
                    op_bit = 0
                    continue
                # NOTE that reading DONE from memory overrides
                # anything in the command word, so get that operand
                # first, and exit if it's DONE
                operand = next_value()
                if operand == DONE:
                    break
                opcode = (command >> op_bit) & 3
                op_bit += 2
                if opcode == hololink_module.Sequencer.Op.POLL:
                    # NOTE THAT WE DON'T ACCOUNT FOR TIMEOUT YET
                    address, match, mask = operand, next_value(), next_value()
                    while True:
                        current = self.memory_read(address)
                        masked = current & mask
                        logging.debug(
                            f"{address=:#x} {current=:#x} {masked=:#x} {match=:#x}"
                        )
                        if masked == match:
                            break
                        time.sleep(0.001)
                elif opcode == hololink_module.Sequencer.Op.RD:
                    address = operand
                    current = self.memory_read(address)
                    self.memory_write(instruction_pointer, current)
                    instruction_pointer += 4
                elif opcode == hololink_module.Sequencer.Op.WR:
                    address = operand
                    value = next_value()
                    self.memory_write(address, value)
                else:
                    assert False and f"Unexpected {opcode=}"


def ns_to_sns(timestamp):
    s, ns = divmod(timestamp, NS_PER_SEC)
    return s, ns


class TestServer:
    def __init__(self, mock_camera=None):
        """If mock_camera is None, then we'll start one and use that."""
        self._mock_camera = mock_camera
        self._process = None
        self._lock = multiprocessing.Lock()
        self._server = MockServer()

    def __enter__(self):
        logging.debug("__enter__")
        if self._mock_camera is None:
            self._lock.acquire()
            self._process = multiprocessing.Process(
                target=self._run,
                name="udp-server",
            )
            self._process.start()
            self._server_address = "127.0.0.1"
            self._lock.acquire()
        else:
            self._server_address = self._mock_camera
        self._control_udp_port = self._server._control_udp_port
        return self

    def __exit__(self, *args, **kwargs):
        logging.debug("__exit__")
        if self._process is not None:
            self._server.close()
            self._process.join()

    def _run(self):
        try:
            self._server.udp_server(self._lock)
        except Exception as e:
            logging.error("Caught %s (%s)" % (e, type(e)))
            tb = "".join(traceback.format_exc())
            for s in tb.split("\n"):
                logging.error(s)

    def address(self):
        return self._server_address

    def control_port(self):
        return self._control_udp_port

    def get_image(self):
        logging.debug("Fetching image.")
        global image_memory, image_memory_metadata
        with image_memory.get_lock():
            length, height, width, bpp = image_memory_metadata[0:4]
            logging.debug(
                "length=%s height=%s width=%s bpp=%s" % (length, height, width, bpp)
            )
            u = np.frombuffer(image_memory.get_obj(), dtype=np.uint8, count=length)
            r = np.array(u).reshape(height, width, bpp)
        return r

    def channel_metadata(self):
        metadata = {
            "peer_ip": self._server_address,
            "control_port": self._control_udp_port,
            "hsb_ip_version": self._server._hsb_ip_version,
            "fpga_uuid": "a0578d60-9353-41ec-9278-d56544eebbe3",
            "serial_number": f"{self._server._serial_number}",
            "sequence_number_checking": 0,
        }
        metadata.update(self._server._enumeration_metadata)
        return hololink_module.Metadata(metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level to display",
    )
    args = parser.parse_args()
    hololink_module.logging_level(args.log_level)
    # provide it a dummy lock; no one else is running to synchronize with this.
    lock = multiprocessing.Lock()
    lock.acquire()
    server = MockServer()
    server.udp_server(lock)


if __name__ == "__main__":
    main()
