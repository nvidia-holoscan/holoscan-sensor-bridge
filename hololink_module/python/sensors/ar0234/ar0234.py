# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import struct
import time

from hololink_module.sensors import csi

from . import ar0234_mode

# AR0234 supports two strap-selectable I2C addresses.
CAM_A_I2C_ADDRESS = 0x10
CAM_B_I2C_ADDRESS = 0x18

CAM_I2C_BUS = 1


class Ar0234Cam:

    # CHIP_VERSION register (16-bit). Reads back AR0234's identity value.
    DEV_ID_REG = 0x3000
    DEV_ID = 0x0A56

    def __init__(
        self,
        hololink,
        i2c_address=CAM_A_I2C_ADDRESS,
        i2c_bus=CAM_I2C_BUS,
        skip_i2c=False,
    ):
        self._i2c_address = i2c_address
        self._i2c_bus = i2c_bus

        self._i2c = hololink.get_i2c(i2c_bus, i2c_address)
        self._mode = ar0234_mode.Ar0234_Mode.Unknown
        self._skip_i2c = skip_i2c
        self._width = None
        self._height = None
        self._pixel_format = None
        self._fsync = False
        self._running = False

    def get_i2c_address(self):
        return self._i2c_address

    def set_i2c_address(self, i2c_address):
        self._i2c_address = i2c_address

    def apply_register_table(self, table):
        for reg, val in table:
            if reg == ar0234_mode.AR0234_TABLE_WAIT_MS:
                time.sleep(val / 1000)
            else:
                self.set_register(reg, val)

    def start(self):
        """Start streaming"""
        start_sequence = (
            ar0234_mode.ar0234_start_fsync if self._fsync else ar0234_mode.ar0234_start
        )
        self.apply_register_table(start_sequence)
        self._running = True

    def stop(self):
        """Stop streaming"""
        self.apply_register_table(ar0234_mode.ar0234_stop)
        self._running = False

    def get_register(self, register):
        if self._skip_i2c:
            logging.debug("Skip i2c")
            return 0
        write_bytes = struct.pack(">H", register)
        read_byte_count = 2
        reply = self._i2c.i2c_transaction(
            self._i2c_address,
            write_bytes,
            read_byte_count,
        )
        value = struct.unpack(">H", reply)[0]
        logging.debug(
            "get_register(i2c_addr=0x%02X, register=0x%04X) = 0x%04X",
            self._i2c_address,
            register,
            value,
        )
        return value

    def set_register(self, register, value, timeout=None):
        if self._skip_i2c:
            logging.debug("Skip i2c")
            return
        write_bytes = struct.pack(">H", register) + struct.pack(">H", value)
        read_byte_count = 0
        logging.debug(
            "set_register(i2c_addr=0x%02X, register=0x%04X, value=0x%04X)",
            self._i2c_address,
            register,
            value,
        )
        self._i2c.i2c_transaction(
            self._i2c_address,
            write_bytes,
            read_byte_count,
        )

    def configure_camera(self, ar0234_mode_set, fsync=False):
        self.set_mode(ar0234_mode_set)
        self._fsync = fsync

        if (
            ar0234_mode_set.value
            == ar0234_mode.Ar0234_Mode.AR0234_MODE_1920X1200_60FPS.value
        ):
            mode_list = ar0234_mode.ar0234_mode_1920X1200_60fps
        else:
            raise ValueError(f"No register table for AR0234 mode {ar0234_mode_set!r}")

        self.apply_register_table(mode_list)

    def set_exposure_reg(self, value=0x02DC):
        self.set_register(ar0234_mode.REG_EXP, value)
        time.sleep(ar0234_mode.AR0234_WAIT_MS / 1000)

    def set_mode(self, ar0234_mode_set):
        if not isinstance(
            ar0234_mode_set, ar0234_mode.Ar0234_Mode
        ) or ar0234_mode_set.value >= len(ar0234_mode.ar0234_frame_format):
            raise ValueError(f"Unsupported AR0234 mode: {ar0234_mode_set!r}")
        self._mode = ar0234_mode_set
        mode = ar0234_mode.ar0234_frame_format[self._mode.value]
        self._height = mode.height
        self._width = mode.width
        self._pixel_format = mode.pixel_format

    def configure_converter(self, converter):
        import hololink as _legacy

        legacy_pixel_format = _legacy.sensors.csi.PixelFormat.RAW_10
        # Guard: both enums must share the same integer value.
        assert (
            csi.PixelFormat.RAW_10.value == legacy_pixel_format.value
        ), "module and legacy RAW_10 integer values have diverged"
        assert self._pixel_format == csi.PixelFormat.RAW_10

        logging.debug(
            "configure_converter:width=%s,height=%s,bpp=%s"
            % (self._width, self._height, self._pixel_format)
        )

        module_pf = csi.PixelFormat.RAW_10
        start_byte = converter.receiver_start_byte()
        try:
            # Module CsiConverterV1 path (e.g. FusaCoeCaptureOp): bindings accept
            # uint32_t for pixel_format so pass .value (an integer) explicitly.
            transmitted_line_bytes = converter.transmitted_line_bytes(
                module_pf.value, self._width
            )
            received_line_bytes = converter.received_line_bytes(transmitted_line_bytes)
            converter.configure(
                start_byte,
                received_line_bytes,
                self._width,
                self._height,
                module_pf.value,
                0,
            )
        except TypeError:
            # Legacy CsiConverter path (e.g. CsiToBayerOp)
            transmitted_line_bytes = converter.transmitted_line_bytes(
                legacy_pixel_format, self._width
            )
            received_line_bytes = converter.received_line_bytes(transmitted_line_bytes)
            converter.configure(
                start_byte,
                received_line_bytes,
                self._width,
                self._height,
                legacy_pixel_format,
            )

    def pixel_format(self):
        return self._pixel_format

    def bayer_format(self):
        return csi.BayerFormat.GRBG

    def test_pattern(self, pattern=2):
        self.set_register(ar0234_mode.REG_TP, pattern)
