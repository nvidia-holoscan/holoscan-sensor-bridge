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


Driver for the E-CON IMX715 camera module


"""

import logging
import time

import hololink as hololink_module

from . import li_i2c_expander

# Camera info
DRIVER_NAME = "IMX715"
VERSION = 1.2

CAM_I2C_ADDRESS = 0x42
GPIO_EXP_I2C_ADDRESS = 0x22

# GPIO EXPANDER I2C BUS ADDRESS
GPIO_EXP_I2C_CTRL_ADDR = 0x04000600

CMD_ID_INIT_CAM = 0x4304
CMD_ID_DE_INIT_CAM = 0x4306
CMD_ID_STREAM_ON = 0x4307
CMD_ID_STREAM_OFF = 0x4308
CMD_ID_STREAM_CONFIG = 0x4309
CMD_ID_SET_CTRL = 0x4311
CMD_ID_LANE_CONFIG = 0x4317
CMD_ID_SET_TEST_PATTERN = 0x43B0
CTRL_INDEX_GAIN = 0x0000
CTRL_INDEX_EXPOSURE = 0x0001
CTRL_INDEX_TRIGGER = 0x0008
CTRL_INDEX_FRAME_RATE = 0x0003
TEGRA_CAMERA_CID_GAIN = 0x009A2009
TEGRA_CAMERA_CID_EXPOSURE = 0x009A200A
TEGRA_CAMERA_CID_TRIGGER = 0x009A2015
TEGRA_CAMERA_CID_FRAME_RATE = 0x009A200B

DEFAULT_VALUE = -1
IMX715_MIN_GAIN = 0
IMX715_MAX_GAIN = 300
IMX715_MIN_FRAMERATE = 1
IMX715_MAX_FRAMERATE = 132
IMX715_MIN_EXPOSURE = 125
IMX715_MAX_EXPOSURE = 4194304
MCU_TRANSACTION_END = -1

IMX715_MODE_3840X2160_30FPS_12BPP = 0
IMX715_MODE_3840X2160_60FPS_10BPP = 1
IMX715_MODE_1220X1080_60FPS_12BPP = 2


class Imx715Cam:
    def __init__(
        self,
        hololink_channel,
        i2c_bus=hololink_module.CAM_I2C_BUS,
        gpio_exp_i2c_controller_address=GPIO_EXP_I2C_CTRL_ADDR,
        expander_configuration=0,
    ):
        self._hololink = hololink_channel.hololink()
        self._i2c = self._hololink.get_i2c(i2c_bus)
        self._gpio_exp_i2c = self._hololink.get_i2c(gpio_exp_i2c_controller_address)
        self._mode = IMX715_MODE_3840X2160_30FPS_12BPP
        self._gain = DEFAULT_VALUE
        self._frame_rate = DEFAULT_VALUE
        self._exposure = DEFAULT_VALUE
        self._trigger_mode = 0
        # Configure i2c expander on the Econ board for IMX715
        self._i2c_expander = li_i2c_expander.LII2CExpander(self._hololink, i2c_bus)
        if expander_configuration == 0:
            self._i2c_expander_configuration = (
                li_i2c_expander.I2C_Expander_Output_EN.OUTPUT_1
            )
        elif expander_configuration == 2:
            self._i2c_expander_configuration = (
                li_i2c_expander.I2C_Expander_Output_EN.OUTPUT_3
            )
        else:
            raise Exception(f"Invalid {expander_configuration=}")

    def setup_clock(self):
        # set the clock driver.
        self._hololink.setup_clock(
            hololink_module.renesas_bajoran_lite_ts1.device_configuration()
        )

    def validate_gain(self, gain_val):
        if gain_val == DEFAULT_VALUE:
            self._gain = DEFAULT_VALUE
            logging.debug("Setting default GAIN values based on IMX715 Mode")
        else:
            if gain_val < IMX715_MIN_GAIN:
                self._gain = IMX715_MIN_GAIN
                logging.debug(
                    f"GAIN {gain_val} value is INVALID, Setting MIN GAIN values({IMX715_MIN_GAIN})"
                )
            elif gain_val > IMX715_MAX_GAIN:
                self._gain = IMX715_MAX_GAIN
                logging.debug(
                    f"GAIN {gain_val} value is INVALID, Setting MAX GAIN values({IMX715_MAX_GAIN})"
                )
            else:
                self._gain = gain_val

    def validate_frame_rate(self, frame_rate_val):
        if frame_rate_val == DEFAULT_VALUE:
            self._frame_rate = self._framerate << 22
            logging.debug(
                "Setting default FRAME RATE({self._frame_rate}) values based on IMX715 Mode"
            )
        else:
            if frame_rate_val < IMX715_MIN_FRAMERATE:
                self._frame_rate = IMX715_MIN_FRAMERATE << 22
                logging.info(
                    f"Framerate {frame_rate_val} value is INVALID, setting supported min framerate ({IMX715_MIN_FRAMERATE})"
                )

            elif (frame_rate_val > IMX715_MAX_FRAMERATE) or (
                frame_rate_val > self._framerate
            ):
                logging.info(
                    f"Framerate {frame_rate_val} value is INVALID, setting supported max framerate ({self._framerate})"
                )
                self._frame_rate = self._framerate << 22

            else:
                self._frame_rate = frame_rate_val << 22

    def validate_exposure(self, exp_val):
        if exp_val == DEFAULT_VALUE:
            self._exposure = DEFAULT_VALUE
            logging.debug("Setting default EXPOSURE values based on IMX715 Mode")
        else:
            value = (exp_val << 22) // 1000
            if value < IMX715_MIN_EXPOSURE:
                self._exposure = IMX715_MIN_EXPOSURE
                logging.info(
                    f"Exposure {exp_val} value is INVALID, setting supported min exposure (0)ms"
                )
            elif value > IMX715_MAX_EXPOSURE:
                self._exposure = IMX715_MAX_EXPOSURE
                logging.info(
                    f"Exposure {exp_val} value is INVALID, setting supported min exposure (1000)ms"
                )
            else:
                self._exposure = value

    def configure(self, imx715_mode, gain_val, frame_rate_val, exp_val, trigger_val):
        # Make sure this is a version we know about.
        version = self.get_version()
        logging.info("version=%s" % (version,))
        assert version == VERSION

        # configure the camera based on the mode
        self.configure_camera(imx715_mode)

        # validate the camera controls based on the mode
        self.validate_gain(gain_val)
        self.validate_frame_rate(frame_rate_val)
        self.validate_exposure(exp_val)
        self._trigger_mode = trigger_val

    def set_stream_on(self):
        self.set_cmd(CMD_ID_STREAM_ON, 0x0000)
        self.set_cmd(CMD_ID_STREAM_ON, MCU_TRANSACTION_END)

    def set_stream_off(self):
        self.set_cmd(CMD_ID_STREAM_OFF, 0x0000)
        self.set_cmd(CMD_ID_STREAM_OFF, MCU_TRANSACTION_END)

    def set_cam_deinit(self):
        self.set_cmd(CMD_ID_DE_INIT_CAM, 0x0000)
        self.set_cmd(CMD_ID_DE_INIT_CAM, MCU_TRANSACTION_END)

    def start(self):
        """Start Streaming"""
        self._running = True
        if self._trigger_mode != 0:
            self.set_gpio_exp_cmd(0x86, 0x00)
            self.set_gpio_exp_cmd(0x8E, 0x00)
            self.set_gpio_exp_cmd(0x86, 0x00)
            self.set_gpio_exp_cmd(0x86, 0x01)
        self.set_ctrl(
            CTRL_INDEX_TRIGGER, TEGRA_CAMERA_CID_TRIGGER, self._trigger_mode, 1
        )
        self.set_stream_on()
        self.set_ctrl(CTRL_INDEX_GAIN, TEGRA_CAMERA_CID_GAIN, self._gain, 2)
        self.set_ctrl(
            CTRL_INDEX_FRAME_RATE, TEGRA_CAMERA_CID_FRAME_RATE, self._frame_rate, 2
        )
        self.set_ctrl(CTRL_INDEX_EXPOSURE, TEGRA_CAMERA_CID_EXPOSURE, self._exposure, 2)

    def stop(self):
        """Stop Streaming"""
        self.set_stream_off()
        self.set_cam_deinit()
        self._running = False

    def get_version(self):
        return VERSION

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
        self.set_gpio_exp_cmd(0x84, 0x00, ignore_nak=True)
        self.set_gpio_exp_cmd(0x8C, 0x00, ignore_nak=True)
        self.set_gpio_exp_cmd(0x84, 0x00, ignore_nak=True)
        self.set_gpio_exp_cmd(0x84, 0x05, ignore_nak=True)

    def set_gpio_exp_cmd(self, register, value, ignore_nak=False):
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
            ignore_nak=ignore_nak,
        )

        time.sleep(200 / 1000)

    def set_ctrl(self, ctrl_index, ctrl_id, value, ctrl_type):
        if ctrl_type == 1:
            payload_len = 11
            ctrl_val_len = 4
        else:
            payload_len = 20
            ctrl_val_len = 8

        crc = 0x00
        self.set_cmd(CMD_ID_SET_CTRL, payload_len)
        self._i2c_expander.configure(self._i2c_expander_configuration.value)
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(CMD_ID_SET_CTRL)

        val = (ctrl_index & 0xFF00) >> 8
        serializer.append_uint8(val)
        crc ^= val

        val = ctrl_index & 0x00FF
        serializer.append_uint8(val)
        crc ^= val

        val = (ctrl_id & 0xFF000000) >> 24
        serializer.append_uint8(val)
        crc ^= val

        val = (ctrl_id & 0x00FF0000) >> 16
        serializer.append_uint8(val)
        crc ^= val

        val = (ctrl_id & 0x0000FF00) >> 8
        serializer.append_uint8(val)
        crc ^= val

        val = ctrl_id & 0x000000FF
        serializer.append_uint8(val)
        crc ^= val

        val = ctrl_type  # CTRL TYPE
        serializer.append_uint8(val)
        crc ^= val

        if ctrl_type == 1:
            val = (value & 0xFF000000) >> 24
            serializer.append_uint8(val)
            crc ^= val

            val = (value & 0x00FF0000) >> 16
            serializer.append_uint8(val)
            crc ^= val

            val = (value & 0x0000FF00) >> 8
            serializer.append_uint8(val)
            crc ^= val

            val = value & 0x000000FF
            serializer.append_uint8(val)
            crc ^= val

        else:
            val = 0x8  # 8byte data
            serializer.append_uint8(val)
            crc ^= val

            val = (ctrl_val_len & 0xFF000000) >> 24
            serializer.append_uint8(val)
            crc ^= val

            val = (ctrl_val_len & 0x00FF0000) >> 16
            serializer.append_uint8(val)
            crc ^= val

            val = (ctrl_val_len & 0x0000FF00) >> 8
            serializer.append_uint8(val)
            crc ^= val

            val = ctrl_val_len & 0x000000FF
            serializer.append_uint8(val)
            crc ^= val

            serializer.append_uint8(0x00)
            serializer.append_uint8(0x00)
            serializer.append_uint8(0x00)
            serializer.append_uint8(0x00)

            val = (value & 0xFF000000) >> 24
            serializer.append_uint8(val)
            crc ^= val

            val = (value & 0x00FF0000) >> 16
            serializer.append_uint8(val)
            crc ^= val

            val = (value & 0x0000FF00) >> 8
            serializer.append_uint8(val)
            crc ^= val

            val = value & 0x000000FF
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
        time.sleep(300 / 1000)

    def set_lane_cfg(self, lane):
        self.set_cmd(CMD_ID_LANE_CONFIG, 0x0002)
        self.set_cmd(CMD_ID_LANE_CONFIG, lane)

    def set_cam_init(self):
        self.set_cmd(CMD_ID_INIT_CAM, 0x0000)
        self.set_cmd(CMD_ID_INIT_CAM, MCU_TRANSACTION_END)

    def set_stream_mode(self):
        self.set_cmd(CMD_ID_SET_CTRL, 0x000B)
        self._i2c_expander.configure(self._i2c_expander_configuration.value)
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(CMD_ID_SET_CTRL)
        serializer.append_uint8(0x00)
        serializer.append_uint8(0x07)
        serializer.append_uint8(0x00)
        serializer.append_uint8(0x9A)
        serializer.append_uint8(0x20)
        serializer.append_uint8(0x08)
        serializer.append_uint8(0x01)
        serializer.append_uint8(0x00)
        serializer.append_uint8(0x00)
        serializer.append_uint8(0x00)
        crc = 0xB4
        serializer.append_uint8(self._mode)
        crc ^= self._mode
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
        if self._mode == IMX715_MODE_3840X2160_30FPS_12BPP:
            serializer.append_uint8(0x00)
            serializer.append_uint8(0x32)
            serializer.append_uint8(0x31)
            serializer.append_uint8(0x42)
            serializer.append_uint8(0x47)
            serializer.append_uint8(0x0F)
            serializer.append_uint8(0x00)
            serializer.append_uint8(0x08)
            serializer.append_uint8(0x70)
            serializer.append_uint8(0x00)
            serializer.append_uint8(0x1E)
            crc = 0x6E
        elif self._mode == IMX715_MODE_3840X2160_60FPS_10BPP:
            serializer.append_uint8(0x01)
            serializer.append_uint8(0x30)
            serializer.append_uint8(0x31)
            serializer.append_uint8(0x42)
            serializer.append_uint8(0x47)
            serializer.append_uint8(0x0F)
            serializer.append_uint8(0x00)
            serializer.append_uint8(0x08)
            serializer.append_uint8(0x70)
            serializer.append_uint8(0x00)
            serializer.append_uint8(0x3C)
            crc = 0x4F
        elif self._mode == IMX715_MODE_1220X1080_60FPS_12BPP:
            serializer.append_uint8(0x02)
            serializer.append_uint8(0x32)
            serializer.append_uint8(0x31)
            serializer.append_uint8(0x42)
            serializer.append_uint8(0x47)
            serializer.append_uint8(0x07)
            serializer.append_uint8(0x98)
            serializer.append_uint8(0x04)
            serializer.append_uint8(0x48)
            serializer.append_uint8(0x00)
            serializer.append_uint8(0x3C)
            crc = 0xEA
        else:
            logging.info("********************Invalid sensor mode********************")
        serializer.append_uint8(0x00)
        serializer.append_uint8(0x01)
        serializer.append_uint8(crc)
        read_byte_count = 0
        self._i2c.i2c_transaction(
            CAM_I2C_ADDRESS,
            write_bytes[: serializer.length()],
            read_byte_count,
            timeout=None,
        )
        time.sleep(300 / 1000)

    def set_mode(self, imx715_mode):
        if imx715_mode == IMX715_MODE_3840X2160_30FPS_12BPP:
            self._width = 3840
            self._height = 2160
            self._pixel_format = hololink_module.sensors.csi.PixelFormat.RAW_12
            self._framerate = 30
        elif imx715_mode == IMX715_MODE_3840X2160_60FPS_10BPP:
            self._width = 3840
            self._height = 2160
            self._pixel_format = hololink_module.sensors.csi.PixelFormat.RAW_10
            self._framerate = 60
        elif imx715_mode == IMX715_MODE_1220X1080_60FPS_12BPP:
            self._width = 1920
            self._height = 1080
            self._pixel_format = hololink_module.sensors.csi.PixelFormat.RAW_12
            self._framerate = 60
        else:
            raise Exception(f"Invalid value for {imx715_mode=}.")
        self._mode = imx715_mode

    def configure_camera(self, imx715_mode):
        self.set_mode(imx715_mode)
        self.set_lane_cfg(0x0004)  # Mipi lane: 4
        self.set_cam_init()
        self.set_stream_mode()
        self.set_stream_config()

    def configure_converter(self, converter):
        # where do we find the first received byte?
        start_byte = converter.receiver_start_byte()
        if self._mode == IMX715_MODE_3840X2160_30FPS_12BPP:
            transmitted_line_bytes = converter.transmitted_line_bytes(
                self._pixel_format, self._width
            )
            received_line_bytes = converter.received_line_bytes(transmitted_line_bytes)
            start_byte += 37 * received_line_bytes  # black lines
            trailing_bytes = 0
        elif self._mode == IMX715_MODE_3840X2160_60FPS_10BPP:
            transmitted_line_bytes = converter.transmitted_line_bytes(
                self._pixel_format, self._width
            )
            received_line_bytes = converter.received_line_bytes(transmitted_line_bytes)
            start_byte += 41 * received_line_bytes  # black lines
            trailing_bytes = 0
        elif self._mode == IMX715_MODE_1220X1080_60FPS_12BPP:
            transmitted_line_bytes = converter.transmitted_line_bytes(
                self._pixel_format, self._width
            )
            transmitted_line_bytes += 38  # extra bytes between lines
            received_line_bytes = converter.received_line_bytes(transmitted_line_bytes)
            start_byte += 21 * received_line_bytes  # black lines
            trailing_bytes = 14 * received_line_bytes
        else:
            raise Exception(f"Unexpected camera {self._mode=}")
        converter.configure(
            start_byte,
            received_line_bytes,
            self._width,
            self._height,
            self._pixel_format,
            trailing_bytes,
        )

    def pixel_format(self):
        return self._pixel_format

    def bayer_format(self):
        return hololink_module.sensors.csi.BayerFormat.GBRG

    def test_pattern(self, pattern):
        self.set_cmd(CMD_ID_SET_TEST_PATTERN, 0x0002)
        self.set_cmd(CMD_ID_SET_TEST_PATTERN, pattern)
