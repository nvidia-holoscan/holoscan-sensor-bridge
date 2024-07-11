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

from . import imx274_mode, li_i2c_expander

# Camera info
DRIVER_NAME = "IMX274-DUAL"
VERSION = 1

# Camera I2C address.
CAM_I2C_ADDRESS = 0b00011010


class Imx274Cam:
    def __init__(
        self,
        hololink_channel,
        i2c_controller_address=hololink_module.CAM_I2C_CTRL,
        expander_configuration=0,
    ):
        self._hololink = hololink_channel.hololink()
        self._i2c = self._hololink.get_i2c(i2c_controller_address)
        self._mode = imx274_mode.Imx274_Mode.Unknown
        # Configure i2c expander on the Leopard board for dual Imx274
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
        self._hololink.setup_clock(
            hololink_module.renesas_bajoran_lite_ts1.device_configuration()
        )

    def configure(self, imx274_mode_set):
        # Make sure this is a version we know about.
        version = self.get_version()
        logging.info("version=%s" % (version,))
        assert version == VERSION

        # configure the camera based on the mode
        self.configure_camera(imx274_mode_set)

    def start(self):
        """Start Streaming"""
        self._running = True
        #
        # Setting these register is time-consuming.
        for reg, val in imx274_mode.imx274_start:
            if (
                reg == imx274_mode.IMX274_TABLE_WAIT_MS
                or reg == imx274_mode.IMX274_TABLE_END
            ):
                time.sleep(val / 1000)  # the val is in ms
            else:
                self.set_register(reg, val)

    def stop(self):
        """Stop Streaming"""
        for reg, val in imx274_mode.imx274_stop:
            if (
                reg == imx274_mode.IMX274_TABLE_WAIT_MS
                or reg == imx274_mode.IMX274_TABLE_END
            ):
                time.sleep(val / 1000)  # the val is in ms
            else:
                self.set_register(reg, val)
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
        read_byte_count = 1
        reply = self._i2c.i2c_transaction(
            CAM_I2C_ADDRESS, write_bytes[: serializer.length()], read_byte_count
        )
        deserializer = hololink_module.Deserializer(reply)
        r = deserializer.next_uint32_be()
        # since the register value is only 8b
        r = r & 0xFF
        logging.debug(
            "get_register(register=%d(0x%X))=%d(0x%X)" % (register, register, r, r)
        )
        return r

    def set_register(self, register, value, timeout=None):
        logging.debug(
            "set_register(register=%d(0x%X), value=%d(0x%X))"
            % (register, register, value, value)
        )
        self._i2c_expander.configure(self._i2c_expander_configuration.value)
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(register)
        serializer.append_uint8(value)
        read_byte_count = 0
        self._i2c.i2c_transaction(
            CAM_I2C_ADDRESS,
            write_bytes[: serializer.length()],
            read_byte_count,
            timeout=timeout,
        )

    def configure_camera(self, imx274_mode_set):
        self.set_mode(imx274_mode_set)

        mode_list = OrderedDict()

        if (
            imx274_mode_set.value
            == imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS.value
        ):
            mode_list = imx274_mode.imx274_mode_3840X2160_60fps
        elif (
            imx274_mode_set.value
            == imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS.value
        ):
            mode_list = imx274_mode.imx274_mode_1920x1080_60fps
        else:
            logging.error(f"{imx274_mode_set} mode is not present.")

        for reg, val in mode_list:
            if (
                reg == imx274_mode.IMX274_TABLE_WAIT_MS
                or reg == imx274_mode.IMX274_TABLE_END
            ):
                time.sleep(val / 1000)  # the val is in ms
            else:
                self.set_register(reg, val)

    def set_exposure_reg(self, value=0x0C):
        """
        IMX274 has the minimum limit of 12 or 0x0c to be set for the exposure.
        """
        if value < 0x0C:
            logging.warn(f"Exposure value {value} is lower than the minimum.")
            value = 0x0C

        if value > 0xFFFF:
            logging.warn(f"Exposure value {value} is higher than the maximum.")
            value = 0xFFFF

        reg_value = format(value, "04x")
        self.set_register(int(imx274_mode.REG_EXP_LSB, 16), int(reg_value[2:4], 16))
        self.set_register(int(imx274_mode.REG_EXP_MSB, 16), int(reg_value[0:2], 16))
        time.sleep(int(imx274_mode.IMX274_WAIT_MS, 16) / 1000)

    def set_digital_gain_reg(self, value=0x0000):
        """
        IMX274 can only have 0(1), 1(2), 2(4), 3(8), 4(16), 5(32), 6(64) value only.
        """
        reg_value = 0x0000
        if value >= 0x40:
            reg_value = 0x06
        elif value >= 0x20:
            reg_value = 0x05
        elif value >= 0x10:
            reg_value = 0x04
        elif value >= 0x08:
            reg_value = 0x03
        elif value >= 0x04:
            reg_value = 0x02
        elif value >= 0x02:
            reg_value = 0x01

        self.set_register(int(imx274_mode.REG_DG, 16), reg_value)
        time.sleep(int(imx274_mode.IMX274_WAIT_MS, 16) / 1000)

    def set_analog_gain_reg(self, value=0x0C):
        if value < 0x00:
            logging.warn(f"AG value {value} is lower than the minimum.")
            value = 0x00

        if value > 0xFFFF:
            logging.warn(f"AG value {value} is more than maximum.")
            value = 0xFFFF

        reg_value = format(value, "04x")
        self.set_register(int(imx274_mode.REG_AG_LSB, 16), int(reg_value[2:4], 16))
        self.set_register(int(imx274_mode.REG_AG_MSB, 16), int(reg_value[0:2], 16))
        time.sleep(int(imx274_mode.IMX274_WAIT_MS, 16) / 1000)

    def set_mode(self, imx274_mode_set):
        if imx274_mode_set.value < len(imx274_mode.Imx274_Mode):
            self._mode = imx274_mode_set
            mode = imx274_mode.imx_frame_format[self._mode.value]
            self._height = mode.height
            self._width = mode.width
            self._pixel_format = mode.pixel_format
        else:
            logging.error("Incorrect mode for IMX274")
            self._mode = -1

    def configure_converter(self, converter):
        (
            frame_start_size,
            frame_end_size,
            line_start_size,
            line_end_size,
        ) = self._hololink.csi_size()
        assert self._pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_10
        # We get 175 bytes of metadata in RAW10 mode
        metadata_size = line_start_size + 175 + line_end_size
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

    def pixel_format(self):
        return self._pixel_format

    def bayer_format(self):
        return hololink_module.sensors.csi.BayerFormat.RGGB

    def test_pattern(self, pattern=None):
        """If pattern==None then we disable test mode."""
        if pattern is None:
            self.set_register(0x303C, 0)
            self.set_register(0x377F, 0)
            self.set_register(0x3781, 0)
            self.set_register(0x370B, 0)
        else:
            self.set_register(0x303C, 0x11)
            self.set_register(0x370E, 0x01)
            self.set_register(0x377F, 0x01)
            self.set_register(0x3781, 0x01)
            self.set_register(0x370B, 0x11)
            self.set_register(0x303D, pattern)
