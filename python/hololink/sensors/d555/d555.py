"""
SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

import logging
import time

import hololink as hololink_module
from .model import Endianness, DataWidth

from . import d555_mode
from .d555_mode import RealSense_StreamCommand, RealSense_StreamId

# Camera info
DRIVER_NAME = "REALSENSE-D555"
VERSION = 1

MUX_I2C_ADDR = 0x1A


class RealsenseCamD555:
    def __init__(
        self,
        hololink_channel,
        stream_id=RealSense_StreamId.DEPTH,
        i2c_controller_address=hololink_module.I2C_CTRL,
    ):
        self._hololink = hololink_channel.hololink()
        self._i2c_controller_address = i2c_controller_address

        # default values
        self._running = False
        self._pixel_format = hololink_module.operators.ImageDecoderOp.PixelFormat.Z16
        self._width = 640
        self._height = 360
        self._mode = 0
        self._stream_id = stream_id
        self._stream_profile = 0

    def setup_clock(self):
        pass

    def set_mode(self, realsense_mode):
        logging.info(f"[Realsense Camera] Mode set to: {realsense_mode}")
        self._mode = realsense_mode
        mode_index = realsense_mode.value

        if self._stream_id == RealSense_StreamId.DEPTH:
            profiles = d555_mode.depth_stream_profiles
        elif self._stream_id == RealSense_StreamId.RGB:
            profiles = d555_mode.rgb_stream_profiles
        else:
            logging.error("Incorrect mode for Realsense D555 camera.")
            self._mode = -1
            return

        self._stream_profile = mode_index
        stream_info = profiles[mode_index]
        self._height = stream_info.height
        self._width = stream_info.width
        self._pixel_format = stream_info.pixel_format

    def power_on(self):
        """Enable power to RealSense"""
        time.sleep(0.1)

    def configure(self, camera_mode):
        """Configure the camera (if needed)"""
        self.power_on()
        logging.info("Sending RealSense config")

        # configure the camera based on the mode
        self.configure_camera(camera_mode)

    def start_camera_stream(self):
        """Start Streaming"""
        logging.info("RealSenseCam: start_camera_stream()")

        data_high = (self._stream_id.value << 8) | RealSense_StreamCommand.SET_PROFILE.value
        data_low = self._stream_profile
        self.set_register(address=MUX_I2C_ADDR, register=data_low, value=data_high,\
            reg_size=DataWidth.BITS_16, val_size=DataWidth.BITS_16, endian=Endianness.LITTLE)

        data_high = (self._stream_id.value << 8) | RealSense_StreamCommand.START_STREAM.value
        data_low = 0
        self.set_register(address=MUX_I2C_ADDR, register=data_low, value=data_high,\
            reg_size=DataWidth.BITS_16, val_size=DataWidth.BITS_16, endian=Endianness.LITTLE)

        return True

    def stop_camera_stream(self):
        """Stop Streaming"""
        logging.info("RealSenseCam: stop_camera_stream()")
        data_high = (self._stream_id.value << 8) | RealSense_StreamCommand.STOP_STREAM.value
        data_low = 0
        self.set_register(address=MUX_I2C_ADDR, register=data_low, value=data_high,\
            reg_size=DataWidth.BITS_16, val_size=DataWidth.BITS_16, endian=Endianness.LITTLE)


    def start(self):
        """Start Streaming"""
        logging.info("RealSenseCam: start()")

        #
        # Setting these register is time-consuming.
        logging.info(
            "RealSenseCam: Stopping any previous streaming before starting new one."
        )
        
        # If the camera stream is already running, stop it first
        # self.stop_camera_stream()

        logging.info("RealSenseCam: Attempting to start streaming...")

        self.start_camera_stream()
        self._running = True

    def stop(self):
        logging.info("RealSenseCam: stop()")
        self.stop_camera_stream()
        self._running = False

    def set_register(
        self,
        address,
        register,
        value,
        reg_size: DataWidth,
        val_size: DataWidth,
        endian: Endianness,
    ):
        logging.debug(
            f"WRITE >> address=0x{int(address):X} set_register(register=0x{int(register):04X}, value=0x{int(value):04X})"
        )

        write_bytes = bytearray(4)
        serializer = hololink_module.Serializer(write_bytes)

        if reg_size == DataWidth.BITS_32:
            if endian == Endianness.BIG:
                serializer.append_uint32_be(register)
            else:
                serializer.append_uint32_le(register)
        elif reg_size == DataWidth.BITS_16:
            if endian == Endianness.BIG:
                serializer.append_uint16_be(register)
            else:
                serializer.append_uint16_le(register)
        elif reg_size == DataWidth.BITS_8:
            serializer.append_uint8(register)

        if val_size == DataWidth.BITS_32:
            if endian == Endianness.BIG:
                serializer.append_uint32_be(value)
            else:
                serializer.append_uint32_le(value)
        elif val_size == DataWidth.BITS_16:
            if endian == Endianness.BIG:
                serializer.append_uint16_be(value)
            else:
                serializer.append_uint16_le(value)
        elif val_size == DataWidth.BITS_8:
            serializer.append_uint8(value)


        reg_data_buffer_ =  self._i2c_controller_address + 16 # Offset for register data buffer
        value = int.from_bytes(write_bytes[: serializer.length()], byteorder='little') 
        self._hololink.write_uint32(reg_data_buffer_, value, timeout=None)


    def configure_converter(self, converter):
        logging.debug(f"[Realsense] Configuring converter")

        converter.configure(
            self._width,
            self._height,
            self._pixel_format,
            0,
            0,
            0,
            0,
        )

    def configure_camera(self, realsense_mode):
        """Configure the camera with the specified mode."""
        self.set_mode(realsense_mode)

    def set_digital_gain_reg(self, val):
        logging.info(f"[Realsense] Digital gain set to: {val}")

    def pixel_format(self):
        return self._pixel_format

    def test_pattern(self, enable=False):
        logging.info("Test pattern control is not implemented for RealSense.")
