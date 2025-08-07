"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from enum import Enum

import hololink as hololink_module

from . import li_i2c_expander

# Camera info
DRIVER_NAME = "e-CAM0M30_TOF"
VERSION = 1

CAM_I2C_ADDRESS = 0x42
GPIO_EXP_I2C_ADDRESS = 0x22

# GPIO EXPANDER I2C BUS ADDRESS
GPIO_EXP_I2C_CTRL_ADDR = 0x3

CMD_ID_VERSION = 0x4300
CMD_ID_INIT_CAM = 0x4304
CMD_ID_GET_STATUS = 0x4305
CMD_ID_DE_INIT_CAM = 0x4306
CMD_ID_STREAM_ON = 0x4307
CMD_ID_STREAM_OFF = 0x4308
CMD_ID_STREAM_CONFIG = 0x4309
CMD_ID_GET_CTRL = 0x4310
CMD_ID_SET_CTRL = 0x4311

CMD_ID_LANE_CONFIG = 0x4317

MCU_TRANSACTION_END = -1


class ECam0M30Tof_Mode(Enum):
    EDEPTH_MODE_DEPTH_IR = 0
    EDEPTH_MODE_DEPTH = 1
    EDEPTH_MODE_IR = 2
    Unknown = 3


class ECam0M30Tof:
    def __init__(
        self,
        hololink_channel,
        i2c_bus=hololink_module.CAM_I2C_BUS,
        gpio_exp_i2c_controller_address=GPIO_EXP_I2C_CTRL_ADDR,
        expander_configuration=0,
        depth_range=1,
    ):
        self._hololink = hololink_channel.hololink()
        self._i2c = self._hololink.get_i2c(i2c_bus)
        self._gpio_exp_i2c = self._hololink.get_i2c(gpio_exp_i2c_controller_address)
        self._mode = ECam0M30Tof_Mode.Unknown
        self._depth_range = depth_range

        # Configure i2c expander on the Econ board for ECam0M30Tof DepthVista sensor
        self._i2c_expander = li_i2c_expander.LII2CExpander(self._hololink, i2c_bus)
        self._i2c_expander_configuration = (
            li_i2c_expander.I2C_Expander_Output_EN.OUTPUT_1
        )

    def setup_clock(self):
        # set the clock driver.
        self._hololink.setup_clock(
            hololink_module.renesas_bajoran_lite_ts1.device_configuration()
        )

    def get_version(self):
        return VERSION

    def mcu_set_cmd(self, register, value, count):
        self._i2c_expander.configure(self._i2c_expander_configuration.value)
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(register)

        if value != MCU_TRANSACTION_END:
            crc = 0x00
            val = (value & 0xFF00) >> 8
            serializer.append_uint8(val)
            crc ^= val

            val = value & 0x00FF
            serializer.append_uint8(val)
            crc ^= val

            serializer.append_uint8(crc)

        read_byte_count = count
        self._i2c.i2c_transaction(
            CAM_I2C_ADDRESS,
            write_bytes[: serializer.length()],
            read_byte_count,
            timeout=None,
        )
        time.sleep(100 / 1000)

    def configure(self, ecam0m30_tof_mode, depth_range=1):
        # Make sure this is a version we know about.
        version = self.get_version()
        logging.info("version=%s" % (version,))
        assert version == VERSION

        # configure the sensor based on the mode
        self.configure_sensor(ecam0m30_tof_mode, depth_range)

    def set_cmd(self, register, value):
        self._i2c_expander.configure(self._i2c_expander_configuration.value)
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(register)

        if value != MCU_TRANSACTION_END:
            crc = 0x00
            val = (value & 0xFF00) >> 8
            serializer.append_uint8(val)
            crc ^= val

            val = value & 0x00FF
            serializer.append_uint8(val)
            crc ^= val

            serializer.append_uint8(crc)

        read_byte_count = 0
        self._i2c.i2c_transaction(
            CAM_I2C_ADDRESS,
            write_bytes[: serializer.length()],
            read_byte_count,
            timeout=None,
        )
        time.sleep(200 / 1000)

    def cam_reset(self):
        self.set_gpio_exp_cmd(0x84, 0x00)
        self.set_gpio_exp_cmd(0x8C, 0x00)
        self.set_gpio_exp_cmd(0x84, 0x00)
        self.set_gpio_exp_cmd(0x84, 0x05)

    def set_gpio_exp_cmd(self, register, value):
        logging.debug(
            "set_register(register=%d(0x%X), value=%d(0x%X))"
            % (register, register, value, value)
        )
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        data = (register << 8) | value
        serializer.append_uint16_be(data)
        read_byte_count = 0
        self._gpio_exp_i2c.i2c_transaction(
            GPIO_EXP_I2C_ADDRESS,
            write_bytes[: serializer.length()],
            read_byte_count,
            timeout=None,
        )
        time.sleep(200 / 1000)

    def get_status(self):
        self.set_cmd(CMD_ID_GET_STATUS, 0x0001)
        self._i2c_expander.configure(self._i2c_expander_configuration.value)
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(CMD_ID_GET_STATUS)
        serializer.append_uint8(0xFF)
        read_byte_count = 0
        reply = self._i2c.i2c_transaction(
            CAM_I2C_ADDRESS,
            write_bytes[: serializer.length()],
            read_byte_count,
            timeout=None,
        )
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(CMD_ID_GET_STATUS)
        read_byte_count = 5
        reply = self._i2c.i2c_transaction(
            CAM_I2C_ADDRESS,
            write_bytes[: serializer.length()],
            read_byte_count,
            timeout=None,
        )
        deserializer = hololink_module.Deserializer(reply)
        r = deserializer.next_uint8()
        c = deserializer.next_uint16_be()
        r = deserializer.next_uint8()
        r = deserializer.next_uint8()
        time.sleep(200 / 1000)
        return r, c

    def set_stream_on(self):
        logging.debug("start stream***")
        self.set_cmd(CMD_ID_STREAM_ON, 0x0000)
        self.set_cmd(CMD_ID_STREAM_ON, MCU_TRANSACTION_END)
        while True:
            r, c = self.get_status()
            if r == 0x00 or c == 0x0000:
                break

    def set_stream_off(self):
        self.set_cmd(CMD_ID_STREAM_OFF, 0x0000)
        self.set_cmd(CMD_ID_STREAM_OFF, MCU_TRANSACTION_END)

    def set_cam_init(self):
        self.set_cmd(CMD_ID_INIT_CAM, 0x0000)
        self.set_cmd(CMD_ID_INIT_CAM, MCU_TRANSACTION_END)

    def set_cam_deinit(self):
        self.set_cmd(CMD_ID_DE_INIT_CAM, 0x0000)
        self.set_cmd(CMD_ID_DE_INIT_CAM, MCU_TRANSACTION_END)

    def start(self):
        """Start Streaming"""
        self._running = True
        self.set_stream_depth_range()
        while True:
            r, _ = self.get_status()
            if r == 0x00:
                break
        self.set_stream_on()
        while True:
            r, _ = self.get_status()
            if r == 0x00:
                break
        self.set_stream_mode()
        while True:
            r, _ = self.get_status()
            if r == 0x00:
                break
        self.set_stream_mask()
        while True:
            r, _ = self.get_status()
            if r == 0x00:
                break

    def stop(self):
        """Stop Streaming"""
        self.set_stream_off()
        while True:
            r, _ = self.get_status()
            if r == 0x00:
                break
        self.set_cam_deinit()
        while True:
            r, _ = self.get_status()
            if r == 0x00:
                break
        self._running = False

    def set_mode(self, ecam0m30_tof_mode):
        self._mode = ecam0m30_tof_mode
        if ecam0m30_tof_mode == ECam0M30Tof_Mode.EDEPTH_MODE_DEPTH_IR:
            number_of_planes = 2
        elif ecam0m30_tof_mode == ECam0M30Tof_Mode.EDEPTH_MODE_DEPTH:
            number_of_planes = 1
        elif ecam0m30_tof_mode == ECam0M30Tof_Mode.EDEPTH_MODE_IR:
            number_of_planes = 1
        else:
            raise Exception(
                f"Incorrect camera mode={ecam0m30_tof_mode} for Ecam0m30tof."
            )

        self._width = 640
        self._height = 480 * number_of_planes
        self._pixel_format = hololink_module.sensors.csi.PixelFormat.RAW_12
        self._framerate = 30

    # TODO: Bayer format needs to be depth or IR
    def bayer_format(self):
        return hololink_module.sensors.csi.BayerFormat.GBRG

    def pixel_format(self):
        return self._pixel_format

    def configure_sensor(self, ecam0m30_tof_mode, depth_range):
        self._depth_range = depth_range
        self.set_mode(ecam0m30_tof_mode)
        self.set_cam_init()
        time.sleep(5)
        self.get_status()
        self.set_stream_off()
        self.set_stream_config()
        while True:
            r, c = self.get_status()
            if r == 0x00:
                break

    def set_stream_mask(self):
        self.set_cmd(CMD_ID_SET_CTRL, 0x000B)
        self._i2c_expander.configure(self._i2c_expander_configuration.value)
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(CMD_ID_SET_CTRL)
        serializer.append_uint8(0x00)
        serializer.append_uint8(0x00)
        serializer.append_uint8(0x00)
        serializer.append_uint8(0x9A)
        serializer.append_uint8(0x20)
        serializer.append_uint8(0x00)
        serializer.append_uint8(0x01)
        crc = 0xBB
        serializer.append_uint8(0x00)
        serializer.append_uint8(0x00)
        serializer.append_uint8(0x00)
        if self._depth_range == 0:
            serializer.append_uint8(0x40)
            crc ^= 0x40
        elif self._depth_range == 1:
            serializer.append_uint8(0x96)
            crc ^= 0x96
        serializer.append_uint8(crc)
        read_byte_count = 0
        self._i2c.i2c_transaction(
            CAM_I2C_ADDRESS,
            write_bytes[: serializer.length()],
            read_byte_count,
            timeout=None,
        )
        time.sleep(300 / 1000)

    def set_stream_depth_range(self):
        self.set_cmd(CMD_ID_SET_CTRL, 0x000B)
        self._i2c_expander.configure(self._i2c_expander_configuration.value)
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(CMD_ID_SET_CTRL)
        serializer.append_uint8(0x00)
        serializer.append_uint8(0x02)
        serializer.append_uint8(0x00)
        serializer.append_uint8(0x9A)
        serializer.append_uint8(0x20)
        serializer.append_uint8(0x02)
        serializer.append_uint8(0x01)
        crc = 0xBB
        serializer.append_uint8(0x00)
        serializer.append_uint8(0x00)
        serializer.append_uint8(0x00)
        if self._depth_range == 0:
            serializer.append_uint8(0x00)
        elif self._depth_range == 1:
            serializer.append_uint8(0x01)
            crc = 0xBA
        serializer.append_uint8(crc)
        read_byte_count = 0
        self._i2c.i2c_transaction(
            CAM_I2C_ADDRESS,
            write_bytes[: serializer.length()],
            read_byte_count,
            timeout=None,
        )
        time.sleep(300 / 1000)

    def set_stream_mode(self):
        self.set_cmd(CMD_ID_SET_CTRL, 0x000B)
        self._i2c_expander.configure(self._i2c_expander_configuration.value)
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(CMD_ID_SET_CTRL)
        serializer.append_uint8(0x00)
        serializer.append_uint8(0x03)
        serializer.append_uint8(0x00)
        serializer.append_uint8(0x9A)
        serializer.append_uint8(0x20)
        serializer.append_uint8(0x03)
        serializer.append_uint8(0x01)
        serializer.append_uint8(0x00)
        serializer.append_uint8(0x00)
        serializer.append_uint8(0x00)
        crc = 0xBB
        if self._mode == ECam0M30Tof_Mode.EDEPTH_MODE_DEPTH_IR:
            serializer.append_uint8(0x00)
        elif self._mode == ECam0M30Tof_Mode.EDEPTH_MODE_DEPTH:
            serializer.append_uint8(0x01)
            crc = 0xBA
        elif self._mode == ECam0M30Tof_Mode.EDEPTH_MODE_IR:
            serializer.append_uint8(0x02)
            crc = 0xB9
        serializer.append_uint8(crc)
        read_byte_count = 0
        self._i2c.i2c_transaction(
            CAM_I2C_ADDRESS,
            write_bytes[: serializer.length()],
            read_byte_count,
            timeout=None,
        )
        time.sleep(300 / 1000)

    def set_stream_config(self):
        self.set_cmd(CMD_ID_STREAM_CONFIG, 0x000E)
        self._i2c_expander.configure(self._i2c_expander_configuration.value)
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(CMD_ID_STREAM_CONFIG)
        serializer.append_uint8(0x00)
        serializer.append_uint8(0x00)
        serializer.append_uint8(0x20)
        serializer.append_uint8(0x32)
        serializer.append_uint8(0x31)
        serializer.append_uint8(0x59)
        serializer.append_uint8(0x02)
        serializer.append_uint8(0x80)
        serializer.append_uint8(0x01)
        serializer.append_uint8(0xE0)
        serializer.append_uint8(0x00)
        serializer.append_uint8(0x1E)
        serializer.append_uint8(0x00)
        serializer.append_uint8(0x01)
        serializer.append_uint8(0x06)
        read_byte_count = 0
        self._i2c.i2c_transaction(
            CAM_I2C_ADDRESS,
            write_bytes[: serializer.length()],
            read_byte_count,
            timeout=None,
        )
        time.sleep(300 / 1000)

    def configure_converter(self, converter):
        # where do we find the first received byte?
        start_byte = converter.receiver_start_byte()
        transmitted_line_bytes = converter.transmitted_line_bytes(
            self._pixel_format, self._width
        )
        received_line_bytes = converter.received_line_bytes(transmitted_line_bytes)
        converter.configure(
            start_byte,
            received_line_bytes,
            self._width,
            self._height,
            self._pixel_format,
        )
