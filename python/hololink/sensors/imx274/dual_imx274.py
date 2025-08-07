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
        i2c_bus=hololink_module.CAM_I2C_BUS,
        expander_configuration=0,
    ):
        self._hololink_channel = hololink_channel
        self._hololink = hololink_channel.hololink()
        self._i2c_bus = i2c_bus
        self._i2c = self._hololink.get_i2c(i2c_bus)
        self._mode = imx274_mode.Imx274_Mode.Unknown
        # Configure i2c expander on the Leopard board for dual Imx274
        self._i2c_expander = li_i2c_expander.LII2CExpander(self._hololink, i2c_bus)
        self._instance = expander_configuration
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
            if reg == imx274_mode.IMX274_TABLE_WAIT_MS:
                time.sleep(val / 1000)  # the val is in ms
            else:
                self.set_register(reg, val)

    def stop(self):
        """Stop Streaming"""
        for reg, val in imx274_mode.imx274_stop:
            if reg == imx274_mode.IMX274_TABLE_WAIT_MS:
                time.sleep(val / 1000)  # the val is in ms
            else:
                self.set_register(reg, val)
        # Let the egress buffer drain.
        time.sleep(0.1)
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
        r = deserializer.next_uint8()
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
        elif (
            imx274_mode_set.value
            == imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS_12BITS.value
        ):
            mode_list = imx274_mode.imx274_mode_3840X2160_60fps_12bits
        else:
            logging.error(f"{imx274_mode_set} mode is not present.")

        for reg, val in mode_list:
            if reg == imx274_mode.IMX274_TABLE_WAIT_MS:
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

        self.set_register(imx274_mode.REG_EXP_LSB, (value >> 8) & 0xFF)
        self.set_register(imx274_mode.REG_EXP_MSB, value & 0xFF)
        time.sleep(imx274_mode.IMX274_WAIT_MS / 1000)

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

        self.set_register(imx274_mode.REG_DG, reg_value)
        time.sleep(imx274_mode.IMX274_WAIT_MS / 1000)

    def set_analog_gain_reg(self, value=0x0C):
        if value < 0x00:
            logging.warn(f"AG value {value} is lower than the minimum.")
            value = 0x00

        if value > 0xFFFF:
            logging.warn(f"AG value {value} is more than maximum.")
            value = 0xFFFF

        self.set_register(imx274_mode.REG_AG_LSB, (value >> 8) & 0xFF)
        self.set_register(imx274_mode.REG_AG_MSB, value & 0xFF)
        time.sleep(imx274_mode.IMX274_WAIT_MS / 1000)

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
        # where do we find the first received byte?
        start_byte = converter.receiver_start_byte()
        transmitted_line_bytes = converter.transmitted_line_bytes(
            self._pixel_format, self._width
        )
        received_line_bytes = converter.received_line_bytes(transmitted_line_bytes)
        # We get 175 bytes of metadata preceding the image data.
        start_byte += converter.received_line_bytes(175)
        if self._pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_10:
            # sensor has 8 lines of optical black before the real image data starts
            start_byte += received_line_bytes * 8
        elif self._pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_12:
            # sensor has 16 lines of optical black before the real image data starts
            start_byte += received_line_bytes * 16
        else:
            raise Exception(f"Incorrect pixel format={self._pixel_format} for IMX274.")
        converter.configure(
            start_byte,
            received_line_bytes,
            self._width,
            self._height,
            self._pixel_format,
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

    def test_pattern_update(self, pattern):
        self.set_register(0x303D, pattern)

    def synchronized_test_pattern_update(self, pattern):
        sequencer = self._hololink_channel.frame_end_sequencer()
        self.synchronized_set_register(sequencer, 0x303D, pattern)
        sequencer.enable()

    def synchronized_set_register(self, sequencer, register, value):
        # Set the I2C expander
        self._i2c_expander.synchronized_configure(
            sequencer, self._i2c_expander_configuration.value
        )
        # Set the command to write the register
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(register)
        serializer.append_uint8(value)
        _, _, status_index = self._i2c.encode_i2c_request(
            sequencer,
            peripheral_i2c_address=CAM_I2C_ADDRESS,  # only 7-bit for now
            write_bytes=serializer.data(),
            read_byte_count=0,
        )
