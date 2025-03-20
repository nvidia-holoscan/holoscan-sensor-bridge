"""
SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import time
from collections import OrderedDict

import hololink as hololink_module

from . import ar0234_mode, li_i2c_expander

# Camera info
DRIVER_NAME = "AR0234"
VERSION = 1

# Camera I2C address.
CAM_A_I2C_ADDRESS = 0b00010000
CAM_B_I2C_ADDRESS = 0b00011000


class Ar0234Cam:
    def __init__(
        self,
        hololink_channel,
        i2c_controller_address=hololink_module.CAM_I2C_CTRL,
        expander_configuration=0,
    ):
        self._hololink = hololink_channel.hololink()
        self._i2c = self._hololink.get_i2c(i2c_controller_address)
        self._mode = ar0234_mode.Ar0234_Mode.Unknown
        # Configure i2c expander on the Leopard board
        self._i2c_expander = li_i2c_expander.LII2CExpander(
            self._hololink, i2c_controller_address
        )
        if expander_configuration == 1:
            self._i2c_expander_configuration = (
                li_i2c_expander.I2C_Expander_Output_EN.OUTPUT_2
            )
        else:
            self._i2c_expander_configuration = (
                li_i2c_expander.I2C_Expander_Output_EN.OUTPUT_1
            )

    def setup_clock(self):
        # set the clock driver.
        logging.debug("setup_clock")
        self._hololink.setup_clock(
            hololink_module.renesas_bajoran_lite_ts1.device_configuration()
        )

    def configure(self, ar0234_mode_set):
        # Make sure this is a version we know about.
        version = self.get_version()
        logging.info("version=%s" % (version,))
        assert version == VERSION

        # configure the camera based on the mode
        self.configure_camera(ar0234_mode_set)

    def start(self):
        """Start Streaming"""
        self._running = True
        #
        # Setting these register is time-consuming.
        for reg, val in ar0234_mode.ar0234_start:
            if reg == ar0234_mode.AR0234_TABLE_WAIT_MS:
                time.sleep(val / 1000)  # the val is in ms
            else:
                self.set_register(CAM_A_I2C_ADDRESS, reg, val)
                self.set_register(CAM_B_I2C_ADDRESS, reg, val)

    def stop(self):
        """Stop Streaming"""
        for reg, val in ar0234_mode.ar0234_stop:
            if reg == ar0234_mode.AR0234_TABLE_WAIT_MS:
                time.sleep(val / 1000)  # the val is in ms
            else:
                self.set_register(CAM_A_I2C_ADDRESS, reg, val)
                self.set_register(CAM_B_I2C_ADDRESS, reg, val)
        self._running = False

    def get_version(self):
        # TODO: get the version or the name of the sensor from the sensor
        return VERSION

    def get_register(self, register):
        logging.debug("get_register(register=%d(0x%X))" % (register, register))
        self._i2c_expander.configure(self._i2c_expander_configuration.value)
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(register)
        read_byte_count = 2
        reply = self._i2c.i2c_transaction(
            CAM_A_I2C_ADDRESS, write_bytes[: serializer.length()], read_byte_count
        )
        deserializer = hololink_module.Deserializer(reply)
        r = deserializer.next_uint16_be()
        logging.debug(
            "get_register(register=%d(0x%X))=%d(0x%X)" % (register, register, r, r)
        )
        return r

    def set_register(self, i2c_address, register, value, timeout=None):
        logging.debug(
            "set_register(i2c address=%d(0x%X), register=%d(0x%X), value=%d(0x%X))"
            % (i2c_address, i2c_address, register, register, value, value)
        )
        self._i2c_expander.configure(self._i2c_expander_configuration.value)
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(register)
        serializer.append_uint16_be(value)
        read_byte_count = 0
        self._i2c.i2c_transaction(
            i2c_address,
            write_bytes[: serializer.length()],
            read_byte_count,
            timeout=timeout,
        )
    
    def set_max929x_register(self, i2c_address, register, value, timeout=None):
        logging.debug(
            "set_max929x_register(i2c address=%d(0x%X), register=%d(0x%X), value=%d(0x%X))"
            % (i2c_address, i2c_address, register, register, value, value)
        )
        self._i2c_expander.configure(self._i2c_expander_configuration.value)
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(register)
        serializer.append_uint8(value)
        read_byte_count = 0
        self._i2c.i2c_transaction(
            i2c_address,
            write_bytes[: serializer.length()],
            read_byte_count,
            timeout=timeout,
        )

    def configure_camera(self, ar0234_mode_set):
        self.set_mode(ar0234_mode_set)

        max929x_list = []
        mode_list = []

        if (
            ar0234_mode_set.value
            == ar0234_mode.Ar0234_Mode.AR0234_MODE_1920X1200_120FPS.value
        ):
            mode_list = ar0234_mode.ar0234_mode_1920X1200_120fps
        elif (
            ar0234_mode_set.value
            == ar0234_mode.Ar0234_Mode.AR0234_MODE_1920X1200_30FPS.value
        ):
            max929x_list = ar0234_mode.max929x_serdes_ar0234
            mode_list = ar0234_mode.ar0234_mode_1920X1200_30fps
        else:
            logging.error(f"{ar0234_mode_set} mode is not present.")

        for i2c_addr, reg, val in max929x_list:
            self.set_max929x_register(i2c_addr, reg, val)
            if (reg == 0x0010 or reg == 0x03e2):
                time.sleep(150 / 1000)

        for reg, val in mode_list:
            if reg == ar0234_mode.AR0234_TABLE_WAIT_MS:
                logging.debug(f"sleep {val} us")
                time.sleep(val / 1000)  # the val is in ms
            else:
                self.set_register(CAM_A_I2C_ADDRESS, reg, val)
                self.set_register(CAM_B_I2C_ADDRESS, reg, val)
        
        self.set_max929x_register(ar0234_mode.MAX929x_SER_I2C_ADDRESS, 0x0332, 0x00)
        time.sleep(200 / 1000)
        self.set_max929x_register(ar0234_mode.MAX929x_SER_I2C_ADDRESS, 0x0332, 0xF0)

    def set_gain_reg(self, value):
        if value < 0x00:
            logging.warn(f"Gain value {value} is lower than the minimum.")
            value = 0x00
        value = value & 0x7F
        self.set_register(CAM_A_I2C_ADDRESS, ar0234_mode.REG_G, value)
        self.set_register(CAM_B_I2C_ADDRESS, ar0234_mode.REG_G, value)
        time.sleep(ar0234_mode.AR0234_WAIT_MS / 1000)

    def set_mode(self, ar0234_mode_set):
        if ar0234_mode_set.value < len(ar0234_mode.Ar0234_Mode):
            self._mode = ar0234_mode_set
            mode = ar0234_mode.imx_frame_format[self._mode.value]
            self._height = mode.height
            self._width = mode.width
            self._pixel_format = mode.pixel_format
        else:
            logging.error("Incorrect mode for AR0234")
            self._mode = -1

    def configure_converter(self, converter):
        logging.debug(f"configure_converter:width={self._width},height={self._height},bpp={self._pixel_format}")
        (
            frame_start_size,
            frame_end_size,
            line_start_size,
            line_end_size,
        ) = self._hololink.csi_size()
        assert self._pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_10
        converter.configure(
            self._width,
            self._height,
            self._pixel_format,
            frame_start_size,
            frame_end_size,
            line_start_size,
            line_end_size,
            margin_top=0,
        )

    def pixel_format(self):
        return self._pixel_format

    def bayer_format(self):
        return hololink_module.sensors.csi.BayerFormat.GRBG
