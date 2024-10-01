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

import logging
import time
from collections import OrderedDict

import hololink as hololink_module

# Camera info
DRIVER_NAME = "IMX477"

# Camera I2C address.
CAM_I2C_ADDRESS = 0b00011010

# Camera default mode
HEIGHT = 2160
WIDTH = 3840
IMX477_TABLE_WAIT_MS = "0000"
IMX477_TABLE_END = "01"
IMX477_WAIT_MS = "01"
IMX477_WAIT_MS_START = "0f"

# Camera Reg
imx477_mode_3840X2160_60fps = OrderedDict(
    [
        (IMX477_TABLE_WAIT_MS, IMX477_WAIT_MS),
        ("e000", "00"),
        ("e07a", "01"),
        ("4ae9", "18"),
        ("4aea", "08"),
        ("f61c", "04"),
        ("f61e", "04"),
        ("4ae9", "21"),
        ("4aea", "80"),
        ("38a8", "1f"),
        ("38a9", "ff"),
        ("38aa", "1f"),
        ("38ab", "ff"),
        ("55d4", "00"),
        ("55d5", "00"),
        ("55d6", "07"),
        ("55d7", "ff"),
        ("55e8", "07"),
        ("55e9", "ff"),
        ("55ea", "00"),
        ("55eb", "00"),
        ("574c", "07"),
        ("574d", "ff"),
        ("574e", "00"),
        ("574f", "00"),
        ("5754", "00"),
        ("5755", "00"),
        ("5756", "07"),
        ("5757", "ff"),
        ("5973", "04"),
        ("5974", "01"),
        ("5d13", "c3"),
        ("5d14", "58"),
        ("5d15", "a3"),
        ("5d16", "1d"),
        ("5d17", "65"),
        ("5d18", "8c"),
        ("5d1a", "06"),
        ("5d1b", "a9"),
        ("5d1c", "45"),
        ("5d1d", "3a"),
        ("5d1e", "ab"),
        ("5d1f", "15"),
        ("5d21", "0e"),
        ("5d22", "52"),
        ("5d23", "aa"),
        ("5d24", "7d"),
        ("5d25", "57"),
        ("5d26", "a8"),
        ("5d37", "5a"),
        ("5d38", "5a"),
        ("5d77", "7f"),
        ("7b75", "0e"),
        ("7b76", "0b"),
        ("7b77", "08"),
        ("7b78", "0a"),
        ("7b79", "47"),
        ("7b7c", "00"),
        ("7b7d", "00"),
        ("8d1f", "00"),
        ("8d27", "00"),
        ("9004", "03"),
        ("9200", "50"),
        ("9201", "6c"),
        ("9202", "71"),
        ("9203", "00"),
        ("9204", "71"),
        ("9205", "01"),
        ("9371", "6a"),
        ("9373", "6a"),
        ("9375", "64"),
        ("991a", "00"),
        ("996b", "8c"),
        ("996c", "64"),
        ("996d", "50"),
        ("9a4c", "0d"),
        ("9a4d", "0d"),
        ("a001", "0a"),
        ("a003", "0a"),
        ("a005", "0a"),
        ("a006", "01"),
        ("a007", "c0"),
        ("a009", "c0"),
        ("3d8a", "01"),
        ("4421", "04"),
        ("7b3b", "01"),
        ("7b4c", "00"),
        ("9905", "00"),
        ("9907", "00"),
        ("9909", "00"),
        ("990b", "00"),
        ("9944", "3c"),
        ("9947", "3c"),
        ("994a", "8c"),
        ("994b", "50"),
        ("994c", "1b"),
        ("994d", "8c"),
        ("994e", "50"),
        ("994f", "1b"),
        ("9950", "8c"),
        ("9951", "1b"),
        ("9952", "0a"),
        ("9953", "8c"),
        ("9954", "1b"),
        ("9955", "0a"),
        ("9a13", "04"),
        ("9a14", "04"),
        ("9a19", "00"),
        ("9a1c", "04"),
        ("9a1d", "04"),
        ("9a26", "05"),
        ("9a27", "05"),
        ("9a2c", "01"),
        ("9a2d", "03"),
        ("9a2f", "05"),
        ("9a30", "05"),
        ("9a41", "00"),
        ("9a46", "00"),
        ("9a47", "00"),
        ("9c17", "35"),
        ("9c1d", "31"),
        ("9c29", "50"),
        ("9c3b", "2f"),
        ("9c41", "6b"),
        ("9c47", "2d"),
        ("9c4d", "40"),
        ("9c6b", "00"),
        ("9c71", "c8"),
        ("9c73", "32"),
        ("9c75", "04"),
        ("9c7d", "2d"),
        ("9c83", "40"),
        ("9c94", "3f"),
        ("9c95", "3f"),
        ("9c96", "3f"),
        ("9c97", "00"),
        ("9c98", "00"),
        ("9c99", "00"),
        ("9c9a", "3f"),
        ("9c9b", "3f"),
        ("9c9c", "3f"),
        ("9ca0", "0f"),
        ("9ca1", "0f"),
        ("9ca2", "0f"),
        ("9ca3", "00"),
        ("9ca4", "00"),
        ("9ca5", "00"),
        ("9ca6", "1e"),
        ("9ca7", "1e"),
        ("9ca8", "1e"),
        ("9ca9", "00"),
        ("9caa", "00"),
        ("9cab", "00"),
        ("9cac", "09"),
        ("9cad", "09"),
        ("9cae", "09"),
        ("9cbd", "50"),
        ("9cbf", "50"),
        ("9cc1", "50"),
        ("9cc3", "40"),
        ("9cc5", "40"),
        ("9cc7", "40"),
        ("9cc9", "0a"),
        ("9ccb", "0a"),
        ("9ccd", "0a"),
        ("9d17", "35"),
        ("9d1d", "31"),
        ("9d29", "50"),
        ("9d3b", "2f"),
        ("9d41", "6b"),
        ("9d47", "42"),
        ("9d4d", "5a"),
        ("9d6b", "00"),
        ("9d71", "c8"),
        ("9d73", "32"),
        ("9d75", "04"),
        ("9d7d", "42"),
        ("9d83", "5a"),
        ("9d94", "3f"),
        ("9d95", "3f"),
        ("9d96", "3f"),
        ("9d97", "00"),
        ("9d98", "00"),
        ("9d99", "00"),
        ("9d9a", "3f"),
        ("9d9b", "3f"),
        ("9d9c", "3f"),
        ("9d9d", "1f"),
        ("9d9e", "1f"),
        ("9d9f", "1f"),
        ("9da0", "0f"),
        ("9da1", "0f"),
        ("9da2", "0f"),
        ("9da3", "00"),
        ("9da4", "00"),
        ("9da5", "00"),
        ("9da6", "1e"),
        ("9da7", "1e"),
        ("9da8", "1e"),
        ("9da9", "00"),
        ("9daa", "00"),
        ("9dab", "00"),
        ("9dac", "09"),
        ("9dad", "09"),
        ("9dae", "09"),
        ("9dc9", "0a"),
        ("9dcb", "0a"),
        ("9dcd", "0a"),
        ("9e17", "35"),
        ("9e1d", "31"),
        ("9e29", "50"),
        ("9e3b", "2f"),
        ("9e41", "6b"),
        ("9e47", "2d"),
        ("9e4d", "40"),
        ("9e6b", "00"),
        ("9e71", "c8"),
        ("9e73", "32"),
        ("9e75", "04"),
        ("9e94", "0f"),
        ("9e95", "0f"),
        ("9e96", "0f"),
        ("9e97", "00"),
        ("9e98", "00"),
        ("9e99", "00"),
        ("9ea0", "0f"),
        ("9ea1", "0f"),
        ("9ea2", "0f"),
        ("9ea3", "00"),
        ("9ea4", "00"),
        ("9ea5", "00"),
        ("9ea6", "3f"),
        ("9ea7", "3f"),
        ("9ea8", "3f"),
        ("9ea9", "00"),
        ("9eaa", "00"),
        ("9eab", "00"),
        ("9eac", "09"),
        ("9ead", "09"),
        ("9eae", "09"),
        ("9ec9", "0a"),
        ("9ecb", "0a"),
        ("9ecd", "0a"),
        ("9f17", "35"),
        ("9f1d", "31"),
        ("9f29", "50"),
        ("9f3b", "2f"),
        ("9f41", "6b"),
        ("9f47", "42"),
        ("9f4d", "5a"),
        ("9f6b", "00"),
        ("9f71", "c8"),
        ("9f73", "32"),
        ("9f75", "04"),
        ("9f94", "0f"),
        ("9f95", "0f"),
        ("9f96", "0f"),
        ("9f97", "00"),
        ("9f98", "00"),
        ("9f99", "00"),
        ("9f9a", "2f"),
        ("9f9b", "2f"),
        ("9f9c", "2f"),
        ("9f9d", "00"),
        ("9f9e", "00"),
        ("9f9f", "00"),
        ("9fa0", "0f"),
        ("9fa1", "0f"),
        ("9fa2", "0f"),
        ("9fa3", "00"),
        ("9fa4", "00"),
        ("9fa5", "00"),
        ("9fa6", "1e"),
        ("9fa7", "1e"),
        ("9fa8", "1e"),
        ("9fa9", "00"),
        ("9faa", "00"),
        ("9fab", "00"),
        ("9fac", "09"),
        ("9fad", "09"),
        ("9fae", "09"),
        ("9fc9", "0a"),
        ("9fcb", "0a"),
        ("9fcd", "0a"),
        ("a14b", "ff"),
        ("a151", "0c"),
        ("a153", "50"),
        ("a155", "02"),
        ("a157", "00"),
        ("a1ad", "ff"),
        ("a1b3", "0c"),
        ("a1b5", "50"),
        ("a1b9", "00"),
        ("a24b", "ff"),
        ("a257", "00"),
        ("a2ad", "ff"),
        ("a2b9", "00"),
        ("b21f", "04"),
        ("b35c", "00"),
        ("b35e", "08"),
        ("0220", "00"),
        ("0221", "11"),
        ("3140", "02"),
        ("3c00", "00"),
        ("3c01", "03"),
        ("3c02", "a2"),
        ("3f0d", "01"),
        ("5748", "07"),
        ("5749", "ff"),
        ("574a", "00"),
        ("574b", "00"),
        ("7b53", "01"),
        ("9369", "73"),
        ("936b", "64"),
        ("936d", "5f"),
        ("9304", "00"),
        ("9305", "00"),
        ("9e9a", "2f"),
        ("9e9b", "2f"),
        ("9e9c", "2f"),
        ("9e9d", "00"),
        ("9e9e", "00"),
        ("9e9f", "00"),
        ("a2a9", "60"),
        ("a2b7", "00"),
        ("e04c", "00"),
        ("e04d", "7f"),
        ("e04e", "00"),
        ("e04f", "1f"),
        ("3e20", "01"),
        ("3e37", "00"),
        ("0114", "03"),
        ("0112", "08"),
        ("0113", "08"),
        ("0808", "00"),
        ("080a", "00"),
        ("080b", "7f"),
        ("080c", "00"),
        ("080d", "4f"),
        ("080e", "00"),
        ("080f", "77"),
        ("0810", "00"),
        ("0811", "5f"),
        ("0812", "00"),
        ("0813", "57"),
        ("0814", "00"),
        ("0815", "4f"),
        ("0816", "01"),
        ("0817", "27"),
        ("0818", "00"),
        ("0819", "3f"),
        ("0820", "41"),
        ("0821", "0A"),
        ("0822", "00"),
        ("0823", "00"),
        ("0101", "00"),
        ("0340", "08"),
        ("0341", "ca"),
        ("0342", "19"),
        ("0343", "80"),
        ("3237", "00"),
        ("3900", "00"),
        ("3901", "00"),
        ("BCF1", "02"),
        ("0136", "18"),
        ("0137", "00"),
        ("0310", "01"),
        ("0305", "04"),
        ("0306", "01"),
        ("0307", "5E"),
        ("0303", "02"),
        ("0301", "05"),
        ("030B", "02"),
        ("0309", "08"),
        ("030d", "04"),
        ("030e", "01"),
        ("030f", "5E"),
        ("00E3", "00"),
        ("00E4", "00"),
        ("0900", "00"),
        ("0901", "00"),
        ("0902", "02"),
        ("0381", "01"),
        ("0383", "01"),
        ("0385", "01"),
        ("0387", "01"),
        ("0408", "00"),
        ("0409", "00"),
        ("040a", "00"),
        ("040b", "00"),
        ("040c", "0f"),
        ("040d", "d8"),  # d8
        ("040e", "08"),  # 08
        ("040f", "70"),  # 38
        ("0401", "00"),
        ("0404", "00"),
        ("0405", "20"),
        ("034C", "0f"),
        ("034D", "00"),
        ("034E", "08"),
        ("034F", "70"),
        ("0200", "07"),
        ("0201", "90"),
        ("0350", "00"),
        ("3F50", "00"),
        ("3F56", "01"),
        ("3F57", "41"),
        ("0204", "01"),
        ("0205", "60"),
        ("3ff9", "00"),
        ("020e", "03"),
        ("020f", "50"),
        ("0210", "05"),
        ("0211", "03"),
        ("0212", "04"),
        ("0213", "6A"),
        ("0214", "03"),
        ("0215", "50"),
        ("3030", "00"),
        ("3032", "01"),
        ("3033", "00"),
        ("0B05", "00"),
        ("0B06", "00"),
        ("3100", "00"),
        ("0600", "00"),
        ("0601", "00"),
        ("0602", "00"),
        ("0603", "00"),
        ("0604", "00"),
        ("0605", "00"),
        ("0606", "00"),
        ("0607", "00"),
        ("0608", "00"),
        ("0609", "00"),
        (IMX477_TABLE_END, IMX477_WAIT_MS),
    ]
)

imx477_mode_3840X2160_60fps = [
    (int(x, 16), int(y, 16)) for x, y in imx477_mode_3840X2160_60fps.items()
]


class Imx477:
    def __init__(self, hololink_channel, camera_id=0):
        if camera_id == 0:
            self.i2c_con_address = hololink_module.CAM_I2C_CTRL
        elif camera_id == 1:
            self.i2c_con_address = hololink_module.CAM_I2C_CTRL + 0x200
        else:
            raise Exception(f"{camera_id=} isn't allowed, only camera_id=0 or 1.")
        self._hololink = hololink_channel.hololink()
        self._i2c = self._hololink.get_i2c(self.i2c_con_address)
        self.cam_id = camera_id

    def configure(self):
        self.set_mode()
        for reg, val in imx477_mode_3840X2160_60fps:
            if reg == IMX477_TABLE_WAIT_MS or reg == IMX477_TABLE_END:
                time.sleep(val / 1000)  # the val is in ms
            else:
                self.set_register(reg, val)

    def set_pattern(self):
        """Set camera mode. Currently supports RAW8 Pixel format only"""
        self.set_register(0x601, 0x2)

    def start(self):
        """Start Streaming"""
        self.set_register(0x100, 0x01),

    def stop(self):
        """Stop Streaming"""
        self.set_register(0x100, 0x0)

    def get_register(self, register):
        logging.debug("get_register(register=%d(0x%X))" % (register, register))
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

    def set_mode(self):
        """Set camera mode. Currently supports RAW8 Pixel format only"""
        self._height = HEIGHT
        self._width = WIDTH
        self._pixel_format = hololink_module.sensors.csi.PixelFormat.RAW_8

    def configure_converter(self, converter):
        (
            frame_start_size,
            frame_end_size,
            line_start_size,
            line_end_size,
        ) = self._hololink.csi_size()
        assert self._pixel_format == hololink_module.sensors.csi.PixelFormat.RAW_8
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
        return hololink_module.sensors.csi.BayerFormat.RGGB
