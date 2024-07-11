# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ctypes
import logging
import multiprocessing
import os
import select
import socket
import struct
import time
import traceback

import hololink.sensors.udp_cam as uc
import numpy as np
import nvtx
import utils

import hololink as hololink_module

bayer_width = None
bayer_height = None
bayer_format = hololink_module.sensors.csi.BayerFormat.RGGB
pixel_format = hololink_module.sensors.csi.PixelFormat.RAW_8
done = False  # terminate the main loop
run = False  # publish bayer data
camera_watchdog_trigger = 0
frame_time_s = 1.0
udp_port = 10203
control_r, control_w = os.pipe()

# How much shared memory for the generated image
# should be allocated?  We'll fault if we try to create an
# image that won't fit.  A 4k image is like 32M.
image_size_limit = 64 * 1024 * 1024
image_memory = multiprocessing.Array(ctypes.c_uint8, image_size_limit)
# total-size, height, width, bpp
image_memory_metadata = multiprocessing.Array(ctypes.c_uint32, [0, 0, 0, 0])

csi_image_data = None
csi_image_length = None


def generate_image(memory, port):
    global bayer_height, bayer_width, bayer_format, pixel_format
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
    frame_start_size = 4
    frame_end_size = 4
    line_start_size = 4
    line_end_size = 2
    frame_start = np.array([255] * frame_start_size, dtype=np.uint8)
    line_start = np.array([255] * line_start_size, dtype=np.uint8)
    line_end = np.array([255] * line_end_size, dtype=np.uint8)
    frame_end = np.array([255] * frame_end_size, dtype=np.uint8)
    global csi_image_data, csi_image_length
    csi_image_data = np.concatenate([frame_start])
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
    csi_image_length = len(csi_image_data)


def get_cam_i2c_register(register_id):
    logging.trace("get_cam_i2c_register(register_id=0x%X)" % (register_id,))
    if register_id == hololink_module.sensors.udp_cam.VERSION:
        # version ID
        return 12344312
    assert False and "get_i2c_register invalid register ID."


def set_cam_i2c_register(register_id, value, memory, ix):
    logging.trace(
        "set_cam_i2c_register(register_id=%d(0x%X), value=%d(0x%X))"
        % (register_id, register_id, value, value)
    )
    if register_id == hololink_module.sensors.udp_cam.RESET:
        pass
    elif register_id == hololink_module.sensors.udp_cam.WIDTH:
        global bayer_width
        bayer_width = value
    elif register_id == hololink_module.sensors.udp_cam.HEIGHT:
        global bayer_height
        bayer_height = value
    elif register_id == hololink_module.sensors.udp_cam.BAYER_FORMAT:
        global bayer_format
        bayer_format = hololink_module.sensors.csi.BayerFormat(value)
    elif register_id == hololink_module.sensors.udp_cam.PIXEL_FORMAT:
        global pixel_format
        pixel_format = hololink_module.sensors.csi.PixelFormat(value)
    elif register_id == hololink_module.sensors.udp_cam.RUN:
        global run
        if value == 0:
            run = False
        else:
            run = True
    elif register_id == hololink_module.sensors.udp_cam.WATCHDOG:
        global camera_watchdog_trigger
        camera_watchdog_trigger = time.monotonic() + value
    elif register_id == hololink_module.sensors.udp_cam.FRAMES_PER_MINUTE:
        global frame_time_s
        frame_time_s = 60.0 / value
        logging.trace("frame_time_s=%s" % (frame_time_s,))
    elif register_id == hololink_module.sensors.udp_cam.INITIALIZE:
        generate_image(memory, uc.VIRTUAL_PORTS[ix])
    else:
        assert False and "set_i2c_register invalid register ID."


def cam_i2c(memory, i2c_address):
    reg_control = i2c_address + 0
    reg_num_bytes = i2c_address + 4
    # reg_clk_ctrl = i2c_address + 8
    reg_data_buffer = i2c_address + 16
    #
    control = memory[reg_control]
    peripheral_i2c_address = (control >> 16) & 0x7F
    enable = control & hololink_module.I2C_CORE_EN
    start = control & hololink_module.I2C_START
    busy = control & hololink_module.I2C_BUSY
    assert enable and start and busy
    assert peripheral_i2c_address == hololink_module.sensors.udp_cam.I2C_ADDRESS
    #
    num_bytes = memory[reg_num_bytes]
    write_byte_count = (num_bytes >> 0) & 0xFF
    read_byte_count = (num_bytes >> 8) & 0xFF
    # Register reads look like this
    if (write_byte_count == 2) and (read_byte_count == 4):
        # udpcam wants the MSB of the address at the lowest address
        register_id_msb = memory[reg_data_buffer] & 0xFF
        register_id_lsb = (memory[reg_data_buffer] >> 8) & 0xFF
        register_id = (register_id_msb << 8) | register_id_lsb
        value = get_cam_i2c_register(register_id)
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
        memory[reg_data_buffer] = msb_value
    # Register writes look like this
    elif (write_byte_count == 6) and (read_byte_count == 0):
        # udpcam wants the MSB of the address at the lowest address
        w0 = memory[reg_data_buffer]
        w1 = memory[reg_data_buffer + 4]
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
        set_cam_i2c_register(register_id, value, memory, 0)
    else:
        assert False and "Unexpected register access."


def main(lock=None):
    logging.debug("Starting.")
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    global udp_port
    udp_socket.bind(("", udp_port))
    message = bytearray(8192)
    reply = bytearray(1500)
    image_count = 0

    global done, frame_time_s

    # I2C interfaces
    memory = {
        hololink_module.CAM_I2C_CTRL: 0,
        hololink_module.BL_I2C_CTRL: 0,
        hololink_module.FPGA_VERSION: 1,
        hololink_module.FPGA_DATE: 20230707,
    }
    for port in uc.VIRTUAL_PORTS:
        u = {
            hololink_module.DP_PACKET_SIZE + port: 0,
            hololink_module.DP_HOST_MAC_LOW + port: 0,
            hololink_module.DP_HOST_MAC_HIGH + port: 0,
            hololink_module.DP_HOST_IP + port: 0,
            hololink_module.DP_HOST_UDP_PORT + port: 0,
            hololink_module.DP_VIP_MASK
            + port: 0,  # which sensor port connects to which ethernet?
            #
            hololink_module.DP_ROCE_CFG + port: 0,
            #
            hololink_module.DP_ROCE_VADDR_MSB_0 + port: 0,
            hololink_module.DP_ROCE_VADDR_LSB_0 + port: 0,
            hololink_module.DP_ROCE_BUF_END_MSB_0 + port: 0,
            hololink_module.DP_ROCE_BUF_END_LSB_0 + port: 0,
            hololink_module.DP_ROCE_RKEY_0 + port: 0,
        }
        memory.update(u)

    now = time.monotonic()
    # frames come in at this rate.
    frame_trigger = now + frame_time_s
    cam_i2c_trigger = None
    logging.info("Ready.")
    if lock is not None:
        lock.release()
    while not done:
        trigger = frame_trigger
        if (cam_i2c_trigger is not None) and (cam_i2c_trigger < trigger):
            trigger = cam_i2c_trigger
        now = time.monotonic()
        timeout = None
        if trigger > now:
            timeout = trigger - now
        global control_r
        cr = [udp_socket, control_r]
        cw = []
        cx = []
        r, w, x = select.select(cr, cw, cx, timeout)

        with nvtx.annotate("update"):
            now = time.monotonic()

            if control_r in r:
                message = os.read(control_r, 8192)
                logging.trace("got control message=%s" % (message,))
                if message == b"exit":
                    return

            if udp_socket in r:
                with nvtx.annotate("udp-request"):
                    length, peer = udp_socket.recvfrom_into(message)
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
                    serializer.append_uint8(cmd_code)
                    serializer.append_uint8(flags)
                    serializer.append_uint16_be(sequence)
                    if cmd_code == hololink_module.WR_DWORD:
                        address = deserializer.next_uint32_be()
                        value = deserializer.next_uint32_be()
                        logging.debug("Writing 0x%X to 0x%X." % (value, address))
                        if address == hololink_module.CAM_I2C_CTRL:
                            if (value & hololink_module.I2C_START) and (
                                value & hololink_module.I2C_CORE_EN
                            ):
                                memory[address] = value | hololink_module.I2C_BUSY
                                cam_i2c(memory, hololink_module.CAM_I2C_CTRL)
                                # make the host do a bit of polling work here
                                # note that we assume that we won't be
                                # using both I2Cs at the same time.
                                assert cam_i2c_trigger is None
                                cam_i2c_trigger = time.monotonic() + 0.001
                            elif value & hololink_module.I2C_DONE_CLEAR:
                                memory[address] &= ~hololink_module.I2C_DONE
                        elif address == hololink_module.BL_I2C_CTRL:
                            if (value & hololink_module.I2C_START) and (
                                value & hololink_module.I2C_CORE_EN
                            ):
                                memory[address] = value | hololink_module.I2C_BUSY
                                cam_i2c(memory, hololink_module.BL_I2C_CTRL)
                                # make the host do a bit of polling work here
                                # note that we assume that we won't be
                                # using both I2Cs at the same time.
                                assert cam_i2c_trigger is None
                                cam_i2c_trigger = time.monotonic() + 0.001
                            elif value & hololink_module.I2C_DONE_CLEAR:
                                memory[address] &= ~hololink_module.I2C_DONE
                        else:
                            memory[address] = value
                        serializer.append_uint8(hololink_module.RESPONSE_SUCCESS)
                    elif cmd_code == hololink_module.RD_DWORD:
                        address = deserializer.next_uint32_be()
                        value = memory[address]
                        logging.debug("Read 0x%X from 0x%X." % (value, address))
                        serializer.append_uint8(hololink_module.RESPONSE_SUCCESS)
                        serializer.append_uint8(0)  # reserved; aligns the next data
                        serializer.append_uint32_be(address)
                        serializer.append_uint32_be(value)
                        send_reply = True
                    else:
                        serializer.append_uint8(hololink_module.RESPONSE_INVALID_CMD)
                        send_reply = True
                    if send_reply:
                        udp_socket.sendto(reply[: serializer.length()], peer)

            if (cam_i2c_trigger is not None) and (now >= cam_i2c_trigger):
                with nvtx.annotate("cam-i2c"):
                    cam_i2c_trigger = None
                    control = memory[hololink_module.CAM_I2C_CTRL]
                    control &= ~hololink_module.I2C_BUSY
                    control |= hololink_module.I2C_DONE
                    memory[hololink_module.CAM_I2C_CTRL] = control
                    #
                    control = memory[hololink_module.BL_I2C_CTRL]
                    control &= ~hololink_module.I2C_BUSY
                    control |= hololink_module.I2C_DONE
                    memory[hololink_module.BL_I2C_CTRL] = control

            if now >= frame_trigger:
                with nvtx.annotate("frame-trigger"):
                    frame_trigger += frame_time_s
                    # Increment the frame counter.
                    # csi_image_data[1] += 1
                    global run
                    if run:
                        if now >= camera_watchdog_trigger:
                            run = False
                            logging.error("Watchdog timeout; stopping frames.")
                    if run:
                        #
                        image_count += 1
                        logging.debug("image_count=%s" % (image_count,))
                        #
                        port = uc.VIRTUAL_PORTS[0]
                        ip_address = memory[hololink_module.DP_HOST_IP + port]
                        ip = [
                            (ip_address >> 24) & 0xFF,
                            (ip_address >> 16) & 0xFF,
                            (ip_address >> 8) & 0xFF,
                            (ip_address >> 0) & 0xFF,
                        ]
                        ip = "%d.%d.%d.%d" % (ip[0], ip[1], ip[2], ip[3])
                        udp_port = memory[hololink_module.DP_HOST_UDP_PORT + port]
                        global csi_image_data, csi_image_length
                        payload_size = (
                            memory[hololink_module.DP_PACKET_SIZE + port] - 78
                        )
                        address = memory[hololink_module.DP_ROCE_VADDR_MSB_0 + port]
                        address <<= 32
                        address |= memory[hololink_module.DP_ROCE_VADDR_LSB_0 + port]
                        qp = memory[hololink_module.DP_ROCE_CFG + port] & 0xFF_FFFF
                        rkey = memory[hololink_module.DP_ROCE_RKEY_0 + port]
                        with nvtx.annotate("write-frame"):
                            # the last packet is a bit different; don't include that here
                            s, e = 0, payload_size
                            logging.debug(
                                f"{port=} {csi_image_length=} {payload_size=}"
                            )
                            while e < csi_image_length:
                                packet = format_write(
                                    qp,
                                    address + s,
                                    bytes(csi_image_data[s:e]),
                                    rkey=rkey,
                                )
                                udp_socket.sendto(packet, (ip, udp_port))
                                s, e = e, e + payload_size
                            last_packet = format_write_immediate(
                                qp,
                                address + s,
                                bytes(csi_image_data[s:e]),
                                rkey=rkey,
                                immediate_value=image_count,
                            )
                            udp_socket.sendto(last_packet, (ip, udp_port))


psn = 0x1000


def format_write(qp, address, content, rkey):
    opcode = 0x2A
    flags = 0
    pkey = 0x1234
    becn = 0
    global psn
    ack_request = 0
    n = len(content)
    icrc = 0x12344321
    r = struct.pack(
        "!BBHIIQII%dsI" % n,
        opcode,
        flags,
        pkey,
        (becn << 24) | qp,
        (ack_request << 24) | psn,
        address,
        rkey,
        n,
        content,
        icrc,
    )
    psn += 1
    return r


def format_write_immediate(qp, address, content, rkey, immediate_value):
    opcode = 0x2B
    flags = 0
    pkey = 0x1234
    becn = 0
    global psn
    ack_request = 0
    n = len(content)
    icrc = 0x12344321
    r = struct.pack(
        "!BBHIIQIII%dsI" % n,
        opcode,
        flags,
        pkey,
        (becn << 24) | qp,
        (ack_request << 24) | psn,
        address,
        rkey,
        n,
        immediate_value,
        content,
        icrc,
    )
    psn += 1
    return r


def main_wrapper(lock):
    try:
        main(lock)
    except Exception as e:
        logging.error("Caught %s (%s)" % (e, type(e)))
        tb = "".join(traceback.format_exc())
        for s in tb.split("\n"):
            logging.info(s)


class TestServer:
    def __init__(self, udpcam=None):
        """If udpcam is None, then we'll start one and use that."""
        self._udpcam = udpcam
        self._process = None

    def __enter__(self):
        logging.debug("__enter__")
        if self._udpcam is None:
            lock = multiprocessing.Lock()
            lock.acquire()
            self._process = multiprocessing.Process(
                target=main_wrapper,
                name="udp-server",
                daemon=True,
                args=(lock,),
            )
            self._process.start()
            global udp_port
            self._server_address = "127.0.0.1"
            lock.acquire()
        else:
            self._server_address = self._udpcam
        self._control_port = udp_port
        return self

    def __exit__(self, *args, **kwargs):
        logging.debug("__exit__")
        if self._process is not None:
            global control_w
            os.write(control_w, b"exit")
            self._process.join()

    def address(self):
        return self._server_address

    def control_port(self):
        return self._control_port

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
            "control_port": self._control_port,
            "configuration_address": 0x1A00,
            "serial_number": "AA55",
            "cpnx_version": 0x2402,
            "vip_mask": 1,
        }
        return hololink_module.Metadata(metadata)


if __name__ == "__main__":
    main()
