# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import fcntl
import logging
import os
import struct
import threading

from hololink.sensors.imx274 import li_i2c_expander

import hololink as hololink_module

from . import v4l2_ids

HOLOLINK_I2C_TRANSFER = 1
HOLOLINK_I2C_OPEN = 2

I2C_M_RD = 1

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1


class LinuxCamera:
    def __init__(
        self,
        hololink_channel,
        hololink_i2c_device,
        hololink_video_device,
        device_i2c_address,
    ):
        self._hololink_channel = hololink_channel
        self._hololink_i2c_device = hololink_i2c_device
        self._hololink_video_device = hololink_video_device
        self._hololink = hololink_channel.hololink()
        self._camera_i2c_address = hololink_module.CAM_I2C_CTRL
        self._device_i2c_address = device_i2c_address

    def setup_clock(self):
        # set the clock driver.
        self._hololink.setup_clock(
            hololink_module.renesas_bajoran_lite_ts1.device_configuration()
        )

    def configure(self, height, width, bayer_format, pixel_format, frame_rate_s):
        # Set up a kernel-mode I2C bus for this guy.
        self._i2c = self._hololink.get_i2c(self._camera_i2c_address)
        self.setup_i2c()
        #
        self._video_fd = os.open(self._hololink_video_device, os.O_RDWR)
        self.configure_camera(height, width, bayer_format, pixel_format, frame_rate_s)
        self._pixel_format = pixel_format
        self._bayer_format = bayer_format
        self._height = height
        self._width = width

    def get_exposure(self):
        arg = struct.pack("II", v4l2_ids.V4L2_CID_EXPOSURE, 0)
        fcntl.ioctl(self._video_fd, v4l2_ids.VIDIOC_G_CTRL, arg)
        cid, value = struct.unpack("II", arg)
        return value

    def set_exposure(self, exposure):
        arg = struct.pack("II", v4l2_ids.V4L2_CID_EXPOSURE, exposure)
        fcntl.ioctl(self._video_fd, v4l2_ids.VIDIOC_S_CTRL, arg)

    def pixel_format(self):
        return self._pixel_format

    def bayer_format(self):
        return self._bayer_format

    def start(self):
        """Tell the camera to start publishing frame data."""
        arg = ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
        fcntl.ioctl(self._video_fd, v4l2_ids.VIDIOC_STREAMON, arg)

    def stop(self):
        arg = ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
        fcntl.ioctl(self._video_fd, v4l2_ids.VIDIOC_STREAMOFF, arg)

    # NOTE THIS IS IMX274 SPECIFIC; FIND A BETTER
    # WAY TO DO THIS
    def configure_converter(self, converter):
        align_64 = hololink_module.sensors.csi.align_64
        # We get 175 bytes of metadata in RAW10 mode
        (
            frame_start_size,
            frame_end_size,
            line_start_size,
            line_end_size,
        ) = self._hololink.csi_size()
        assert self._pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_10
        # We get 175 bytes of metadata in RAW10 mode
        metadata_size = align_64(line_start_size + 175 + line_end_size)
        converter.configure(
            self._width,
            self._height,
            self._pixel_format,
            frame_start_size + metadata_size,
            frame_end_size,
            line_start_size,
            line_end_size,
            margin_top=8,  # sensor has 8 lines of optical black before the real image data starts
        )

    def configure_camera(self, height, width, bayer_format, pixel_format, frame_rate_s):
        # FIXME -- we need a way to set this flag
        # before each I2C transaction; so that needs the
        # driver to help.
        i2c_expander = li_i2c_expander.LII2CExpander(
            self._hololink, self._camera_i2c_address
        )
        i2c_expander.configure(li_i2c_expander.I2C_Expander_Output_EN.OUTPUT_1.value)

    def setup_i2c(self):
        self._hololink_i2c_device_fd = os.open(self._hololink_i2c_device, os.O_RDWR)
        logging.debug(f"Opened {self._hololink_i2c_device_fd=}")
        # We always get a HOLOLINK_I2C_OPEN request first, telling us what the
        # new I2C bus number is. We'll throw an exception if it's not exactly
        # as expected.
        command_bytes = os.read(self._hololink_i2c_device_fd, 8192)
        ds = hololink_module.Deserializer(command_bytes)
        command = ds.next_uint16_le()
        if command != HOLOLINK_I2C_OPEN:
            raise Exception(f"Unexpected {command=}.")
        self._i2c_device_number = ds.next_uint16_le()
        logging.debug(f"{self._i2c_device_number=}")
        self._i2c_proxy_thread = threading.Thread(target=self._i2c_proxy, daemon=True)
        self._i2c_proxy_thread.start()
        return self._i2c_device_number

    def _i2c_proxy(self):
        # We only get here after we've handled HOLOLINK_I2C_OPEN.
        # fetch all the commands from the driver.
        while True:
            command_bytes = os.read(self._hololink_i2c_device_fd, 8192)
            if len(command_bytes) == 0:
                return
            ds = hololink_module.Deserializer(command_bytes)
            command = ds.next_uint16_le()
            if command == HOLOLINK_I2C_TRANSFER:
                key = ds.next_uint16_le()
                num = ds.next_uint16_le()
                # Fetch the total request; this
                # way we can connect write+read requests
                # into a single atomic transfer.
                request = []
                for i in range(num):
                    addr, flags, length = (
                        ds.next_uint16_le(),
                        ds.next_uint16_le(),
                        ds.next_uint16_le(),
                    )
                    request.append((addr, flags, length))
                reply = bytearray(struct.pack("<HH", HOLOLINK_I2C_TRANSFER, key))
                while len(request):
                    addr, flags, length = request.pop(0)
                    logging.trace(f"{addr=:#X} {flags=:#X} {length=}")
                    write = (flags & I2C_M_RD) == 0
                    if write:
                        data = ds.next_buffer(length)
                    more = len(request) > 0
                    if write and more:
                        next_addr, next_flags, next_length = request[0]
                        next_write = (next_flags & I2C_M_RD) == 0
                        if (addr == next_addr) and not next_write:
                            logging.trace(f"followed with read length={next_length}")
                            request.pop(0)
                            r = self._i2c.i2c_transaction(addr, data, next_length)
                            reply.extend(r)
                            continue
                    if write:
                        self._i2c.i2c_transaction(addr, data, 0)
                        continue
                    if not write:
                        r = self._i2c.i2c_transaction(addr, [], length)
                        reply.extend(r)
                        continue
                    # we don't ever get here.
                    assert False
                logging.trace(f"{len(reply)=}")
                os.write(self._hololink_i2c_device_fd, reply)
            else:
                logging.error(f"Unexpected {command=:#X}; ignoring.")
