"""
SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from collections import namedtuple
from enum import Enum

import hololink

# values are on hex number system to be consistent with rest of the list
AR0234_TABLE_WAIT_MS = "ar0234-table-wait-ms"
AR0234_WAIT_MS = 0x01
AR0234_WAIT_MS_START = 0xC8

# MAX929x I2C address
MAX929x_DES_I2C_ADDRESS = 0x40
MAX929x_SER_I2C_ADDRESS = 0x48

# Register addresses for camera properties. They only accept 8bits of value.

# Analog Gain
REG_G = 0x3060


ar0234_start = [
        (0x301A, 0x205C),
        (AR0234_TABLE_WAIT_MS, AR0234_WAIT_MS_START),
    ]

ar0234_stop = [
        (0x301A, 0x2058),
        (AR0234_TABLE_WAIT_MS, AR0234_WAIT_MS),
    ]

# AR0234 MIPI Full resolution:1920X1200 Output:10bit Master Mode 120fps
# value pairs use hex number system
ar0234_mode_1920X1200_120fps = [
        (AR0234_TABLE_WAIT_MS, AR0234_WAIT_MS),
        (0x302A,0x0005),
        (0x302C,0x0001),
        (0x302E,0x0003),
        (0x3030,0x0032),
        (0x3036,0x000A),
        (0x3038,0x0001),
        (0x30B0,0x0028),
        (0x31B0,0x0082),
        (0x31B2,0x005C),
        (0x31B4,0x51C8),
        (0x31B6,0x3257),
        (0x31B8,0x904B),
        (0x31BA,0x030B),
        (0x31BC,0x8E09),
        (0x3354,0x002B),
        (0x31AE,0x0204),
        (0x3002,0x0008),
        (0x3004,0x0008),
        (0x3006,0x04B7),
        (0x3008,0x0787),
        (0x300A,0x04C4),
        (0x300C,0x0264),
        (0x3012,0x0400),
        (0x3060,0x004F),
        (0x31AC,0x0A0A),
        (0x306E,0x9010),
        (0x30A2,0x0001),
        (0x30A6,0x0001),
        (0x3040,0x0000),
        (0x31D0,0x0000),
        (AR0234_TABLE_WAIT_MS, AR0234_WAIT_MS),
    ]

#AR0234 GMSL2_MIPI4Lane_1920x1200@30fps_Pxlclk22.5MHz_Extclk27MHz
max929x_serdes_ar0234 = [
    (MAX929x_SER_I2C_ADDRESS,0x0010,0x31),
    (MAX929x_DES_I2C_ADDRESS,0x0010,0x21),
    (MAX929x_DES_I2C_ADDRESS,0x0330,0x06),
    (MAX929x_DES_I2C_ADDRESS,0x0332,0x4E),
    (MAX929x_DES_I2C_ADDRESS,0x0333,0xE4),
    (MAX929x_DES_I2C_ADDRESS,0x0331,0x77),
    (MAX929x_DES_I2C_ADDRESS,0x0311,0x41),
    (MAX929x_DES_I2C_ADDRESS,0x0308,0x74),
    (MAX929x_DES_I2C_ADDRESS,0x0314,0x6B),
    (MAX929x_DES_I2C_ADDRESS,0x0316,0x22),
    (MAX929x_DES_I2C_ADDRESS,0x0318,0x6B),
    (MAX929x_DES_I2C_ADDRESS,0x031A,0x22),
    (MAX929x_DES_I2C_ADDRESS,0x0002,0xFF),
    (MAX929x_DES_I2C_ADDRESS,0x0053,0x10),
    (MAX929x_DES_I2C_ADDRESS,0x0057,0x11),
    (MAX929x_DES_I2C_ADDRESS,0x005B,0x12),
    (MAX929x_DES_I2C_ADDRESS,0x005F,0x13),
    (MAX929x_SER_I2C_ADDRESS,0x0330,0x04),
    (MAX929x_SER_I2C_ADDRESS,0x0333,0x4E),
    (MAX929x_SER_I2C_ADDRESS,0x0334,0xE4),
    (MAX929x_SER_I2C_ADDRESS,0x040A,0x00),
    (MAX929x_SER_I2C_ADDRESS,0x044A,0xD0),
    (MAX929x_SER_I2C_ADDRESS,0x048A,0xD0),
    (MAX929x_SER_I2C_ADDRESS,0x04CA,0x00),
    (MAX929x_SER_I2C_ADDRESS,0x031D,0x00),
    (MAX929x_SER_I2C_ADDRESS,0x0320,0x2F),
    (MAX929x_SER_I2C_ADDRESS,0x0323,0x2F),
    (MAX929x_SER_I2C_ADDRESS,0x0326,0x00),
    (MAX929x_SER_I2C_ADDRESS,0x0050,0x01),
    (MAX929x_SER_I2C_ADDRESS,0x0051,0x00),
    (MAX929x_SER_I2C_ADDRESS,0x0052,0x02),
    (MAX929x_SER_I2C_ADDRESS,0x0053,0x03),
    (MAX929x_SER_I2C_ADDRESS,0x0332,0xF0),
    (MAX929x_DES_I2C_ADDRESS,0x02BE,0x90),
    (MAX929x_DES_I2C_ADDRESS,0x02BF,0x60),
    (MAX929x_DES_I2C_ADDRESS,0x02CA,0x80),
    (MAX929x_DES_I2C_ADDRESS,0x02CB,0x60),
    (MAX929x_SER_I2C_ADDRESS,0x0005,0x00),
    (MAX929x_DES_I2C_ADDRESS,0x02D3,0x90),
    (MAX929x_DES_I2C_ADDRESS,0x02D4,0x60),
    (MAX929x_DES_I2C_ADDRESS,0x02D6,0x90),
    (MAX929x_DES_I2C_ADDRESS,0x02D7,0x60),
    (MAX929x_SER_I2C_ADDRESS,0x03EF,0xC0),
    (MAX929x_SER_I2C_ADDRESS,0x03E2,0x00),
    (MAX929x_SER_I2C_ADDRESS,0x03EA,0x00),
    (MAX929x_SER_I2C_ADDRESS,0x03EB,0x00),
    (MAX929x_SER_I2C_ADDRESS,0x03E5,0xD4),
    (MAX929x_SER_I2C_ADDRESS,0x03E6,0xDC),
    (MAX929x_SER_I2C_ADDRESS,0x03E7,0xCB),
    (MAX929x_SER_I2C_ADDRESS,0x03F1,0x40),
    (MAX929x_SER_I2C_ADDRESS,0x03E0,0x04),
    (MAX929x_DES_I2C_ADDRESS,0x02D9,0x04),
    (MAX929x_DES_I2C_ADDRESS,0x02DB,0x08),
    (MAX929x_DES_I2C_ADDRESS,0x02DC,0x04),
    (MAX929x_DES_I2C_ADDRESS,0x02DE,0x08),
    (MAX929x_DES_I2C_ADDRESS,0x02BE,0x83),
    (MAX929x_DES_I2C_ADDRESS,0x02BF,0x11),
    (MAX929x_SER_I2C_ADDRESS,0x02BC,0x04),
    (MAX929x_SER_I2C_ADDRESS,0x02BE,0x11),
    (MAX929x_SER_I2C_ADDRESS,0x0003,0x40),
    (MAX929x_DES_I2C_ADDRESS,0x02C4,0x83),
    (MAX929x_DES_I2C_ADDRESS,0x02C5,0x12),
    (MAX929x_SER_I2C_ADDRESS,0x02BF,0x04),
    (MAX929x_SER_I2C_ADDRESS,0x02C1,0x12),
]
ar0234_mode_1920X1200_30fps = [
	(AR0234_TABLE_WAIT_MS, AR0234_WAIT_MS_START),
	(0x301A,0x00D9),
    (0x30B0,0x0028),
    (0x31AE,0x0204),
    (0x3030,0x0064),
    (0x302E,0x0006),
    (0x302C,0x0004),
    (0x302A,0x0005),
    (0x3038,0x0004),
    (0x3036,0x000A),
    (0x31B0,0x002F),
    (0x31B2,0x002C),
    (0x31B4,0x1144),
    (0x31B6,0x00C7),
    (0x31B8,0x3047),
    (0x31BA,0x0103),
    (0x31BC,0x8583),
    (0x31D0,0x0000),
    (0x3002,0x0008),
    (0x3004,0x0008),
    (0x3006,0x04B7),
    (0x3008,0x0787),
    (0x3064,0x1802),
    (0x300A,0x04C6),
    (0x300C,0x0264),
    (0x30A2,0x0001),
    (0x30A6,0x0001),
    (0x3012,0x02DC),
    (0x3060,0x000E),
    (AR0234_TABLE_WAIT_MS, AR0234_WAIT_MS),
]

class Ar0234_Mode(Enum):
    AR0234_MODE_1920X1200_120FPS = 0
    AR0234_MODE_1920X1200_30FPS = 1
    Unknown = 2


frame_format = namedtuple(
    "FrameFormat", ["width", "height", "framerate", "pixel_format"]
)

imx_frame_format = {
    Ar0234_Mode.AR0234_MODE_1920X1200_120FPS.value: frame_format(1920, 1200, 120, hololink.sensors.csi.PixelFormat.RAW_10),
    Ar0234_Mode.AR0234_MODE_1920X1200_30FPS.value: frame_format(1920, 1200, 30, hololink.sensors.csi.PixelFormat.RAW_10),
}
