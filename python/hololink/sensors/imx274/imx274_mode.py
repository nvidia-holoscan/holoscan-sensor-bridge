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
IMX274_TABLE_WAIT_MS = "imx274-table-wait-ms"
IMX274_WAIT_MS = 0x01
IMX274_WAIT_MS_START = 0x0F

# Register addresses for camera properties. They only accept 8bits of value.

# Analog Gain
REG_AG_MSB = 0x300B
REG_AG_LSB = 0x300A

# Exposure
REG_EXP_MSB = 0x300D
REG_EXP_LSB = 0x300C

# Digital Gain
REG_DG = 0x3012

imx274_start = [
    (0x3000, 0x00),  # mode select streaming on
    (0x303E, 0x02),
    (IMX274_TABLE_WAIT_MS, IMX274_WAIT_MS_START),
    (0x30F4, 0x00),
    (0x3018, 0xA2),
    (IMX274_TABLE_WAIT_MS, IMX274_WAIT_MS_START),
]

imx274_stop = [
    (IMX274_TABLE_WAIT_MS, IMX274_WAIT_MS),
    (0x3000, 0x01),  # mode select streaming off
]

# test pattern
tp_colorbars = [
    # test pattern
    (0x303C, 0x11),
    (0x303D, 0x0B),
    (0x370B, 0x11),
    (0x370E, 0x00),
    (0x377F, 0x01),
    (0x3781, 0x01),
]

# Mode : 3840X2160 10 bits 60fps
# value pairs use hex number system
imx274_mode_3840X2160_60fps = [
    (IMX274_TABLE_WAIT_MS, IMX274_WAIT_MS),
    (0x3000, 0x12),
    (0x3120, 0xF0),
    (0x3122, 0x02),
    (0x3129, 0x9C),
    (0x312A, 0x02),
    (0x312D, 0x02),
    (0x310B, 0x00),
    (0x304C, 0x00),
    (0x304D, 0x03),
    (0x331C, 0x1A),
    (0x3502, 0x02),
    (0x3529, 0x0E),
    (0x352A, 0x0E),
    (0x352B, 0x0E),
    (0x3538, 0x0E),
    (0x3539, 0x0E),
    (0x3553, 0x00),
    (0x357D, 0x05),
    (0x357F, 0x05),
    (0x3581, 0x04),
    (0x3583, 0x76),
    (0x3587, 0x01),
    (0x35BB, 0x0E),
    (0x35BC, 0x0E),
    (0x35BD, 0x0E),
    (0x35BE, 0x0E),
    (0x35BF, 0x0E),
    (0x366E, 0x00),
    (0x366F, 0x00),
    (0x3670, 0x00),
    (0x3671, 0x00),
    (0x30EE, 0x01),
    (0x3304, 0x32),
    (0x3306, 0x32),
    (0x3590, 0x32),
    (0x3686, 0x32),
    # resolution */
    (0x30E2, 0x01),
    (0x30F6, 0x07),
    (0x30F7, 0x01),
    (0x30F8, 0xC6),
    (0x30F9, 0x11),
    (0x3130, 0x78),
    (0x3131, 0x08),
    (0x3132, 0x70),
    (0x3133, 0x08),
    # crop */
    (0x30DD, 0x01),
    (0x30DE, 0x04),
    (0x30E0, 0x03),
    (0x3037, 0x01),
    (0x3038, 0x0C),
    (0x3039, 0x00),
    (0x303A, 0x0C),
    (0x303B, 0x0F),
    # mode setting */
    (0x3004, 0x01),
    (0x3005, 0x01),
    (0x3006, 0x00),
    (0x3007, 0x02),
    (0x300C, 0x0C),
    (0x300D, 0x00),
    (0x300E, 0x00),
    (0x3019, 0x00),
    (0x3A41, 0x08),
    (0x3342, 0x0A),
    (0x3343, 0x00),
    (0x3344, 0x16),
    (0x3345, 0x00),
    (0x3528, 0x0E),
    (0x3554, 0x1F),
    (0x3555, 0x01),
    (0x3556, 0x01),
    (0x3557, 0x01),
    (0x3558, 0x01),
    (0x3559, 0x00),
    (0x355A, 0x00),
    (0x35BA, 0x0E),
    (0x366A, 0x1B),
    (0x366B, 0x1A),
    (0x366C, 0x19),
    (0x366D, 0x17),
    (0x33A6, 0x01),
    (0x306B, 0x05),
    (IMX274_TABLE_WAIT_MS, IMX274_WAIT_MS),
]

# Mode : 3840X2160 12 bits 60fps
# value pairs use hex number system
imx274_mode_3840X2160_60fps_12bits = [
    (IMX274_TABLE_WAIT_MS, IMX274_WAIT_MS),
    (0x3000, 0x12),
    (0x3120, 0xF0),
    (0x3122, 0x02),
    (0x3129, 0x9C),
    (0x312A, 0x02),
    (0x312D, 0x02),
    (0x310B, 0x00),
    (0x304C, 0x00),
    (0x304D, 0x03),
    (0x331C, 0x1A),
    (0x3502, 0x02),
    (0x3529, 0x0E),
    (0x352A, 0x0E),
    (0x352B, 0x0E),
    (0x3538, 0x0E),
    (0x3539, 0x0E),
    (0x3553, 0x00),
    (0x357D, 0x05),
    (0x357F, 0x05),
    (0x3581, 0x04),
    (0x3583, 0x76),
    (0x3587, 0x01),
    (0x35BB, 0x0E),
    (0x35BC, 0x0E),
    (0x35BD, 0x0E),
    (0x35BE, 0x0E),
    (0x35BF, 0x0E),
    (0x366E, 0x00),
    (0x366F, 0x00),
    (0x3670, 0x00),
    (0x3671, 0x00),
    (0x30EE, 0x01),
    (0x3304, 0x32),
    (0x3306, 0x32),
    (0x3590, 0x32),
    (0x3686, 0x32),
    # resolution */
    (0x30E2, 0x00),
    (0x30F6, 0xED),
    (0x30F7, 0x01),
    (0x30F8, 0x08),
    (0x30F9, 0x13),
    (0x3130, 0x94),
    (0x3131, 0x08),
    (0x3132, 0x70),
    (0x3133, 0x08),
    # crop */
    (0x30DD, 0x01),
    (0x30DE, 0x04),  # crop 18 lines - 12h
    (0x30E0, 0x03),  # 6 lines of ignored area
    (0x3037, 0x01),
    (0x3038, 0x0C),  # 12 H lines to crop
    (0x3039, 0x00),
    (0x303A, 0x0C),  # next cut at 3852
    (0x303B, 0x0F),
    # mode setting */
    (0x3004, 0x00),
    (0x3005, 0x07),
    (0x3006, 0x00),
    (0x3007, 0x02),
    (0x300C, 0x0C),
    (0x300D, 0x00),
    (0x300E, 0x00),
    (0x3019, 0x00),
    (0x3A41, 0x10),
    (0x3342, 0xFF),
    (0x3343, 0x01),
    (0x3344, 0xFF),
    (0x3345, 0x01),
    (0x3528, 0x0F),
    (0x3554, 0x00),
    (0x3555, 0x00),
    (0x3556, 0x00),
    (0x3557, 0x00),
    (0x3558, 0x00),
    (0x3559, 0x1F),
    (0x355A, 0x1F),
    (0x35BA, 0x0F),
    (0x366A, 0x00),
    (0x366B, 0x00),
    (0x366C, 0x00),
    (0x366D, 0x00),
    (0x33A6, 0x01),
    (0x306B, 0x07),
    (IMX274_TABLE_WAIT_MS, IMX274_WAIT_MS),
]

# Mode : 1920x1080 10 bits 60fps
# value pairs use hex number system
imx274_mode_1920x1080_60fps = [
    (IMX274_TABLE_WAIT_MS, IMX274_WAIT_MS),
    (0x3000, 0x12),  # mode select streaming on
    # input freq. 24M
    (0x3120, 0xF0),
    (0x3122, 0x02),
    (0x3129, 0x9C),
    (0x312A, 0x02),
    (0x312D, 0x02),
    (0x310B, 0x00),
    (0x304C, 0x00),
    (0x304D, 0x03),
    (0x331C, 0x1A),
    (0x3502, 0x02),
    (0x3529, 0x0E),
    (0x352A, 0x0E),
    (0x352B, 0x0E),
    (0x3538, 0x0E),
    (0x3539, 0x0E),
    (0x3553, 0x00),
    (0x357D, 0x05),
    (0x357F, 0x05),
    (0x3581, 0x04),
    (0x3583, 0x76),
    (0x3587, 0x01),
    (0x35BB, 0x0E),
    (0x35BC, 0x0E),
    (0x35BD, 0x0E),
    (0x35BE, 0x0E),
    (0x35BF, 0x0E),
    (0x366E, 0x00),
    (0x366F, 0x00),
    (0x3670, 0x00),
    (0x3671, 0x00),
    (0x30EE, 0x01),
    (0x3304, 0x32),
    (0x3306, 0x32),
    (0x3590, 0x32),
    (0x3686, 0x32),
    # resolution
    (0x30E2, 0x02),
    (0x30F6, 0x04),
    (0x30F7, 0x01),
    (0x30F8, 0x0C),
    (0x30F9, 0x12),
    (0x3130, 0x40),
    (0x3131, 0x04),
    (0x3132, 0x38),
    (0x3133, 0x04),
    # crop
    (0x30DD, 0x01),
    (0x30DE, 0x07),
    (0x30DF, 0x00),
    (0x30E0, 0x04),
    (0x30E1, 0x00),
    (0x3037, 0x01),
    (0x3038, 0x0C),
    (0x3039, 0x00),
    (0x303A, 0x0C),
    (0x303B, 0x0F),
    # mode setting
    (0x3004, 0x02),
    (0x3005, 0x21),
    (0x3006, 0x00),
    (0x3007, 0xB1),
    (0x300C, 0x08),  # SHR: Minimum 8
    (0x300D, 0x00),
    (0x3019, 0x00),
    (0x3A41, 0x08),
    (0x3342, 0x0A),
    (0x3343, 0x00),
    (0x3344, 0x1A),
    (0x3345, 0x00),
    (0x3528, 0x0E),
    (0x3554, 0x00),
    (0x3555, 0x01),
    (0x3556, 0x01),
    (0x3557, 0x01),
    (0x3558, 0x01),
    (0x3559, 0x00),
    (0x355A, 0x00),
    (0x35BA, 0x0E),
    (0x366A, 0x1B),
    (0x366B, 0x1A),
    (0x366C, 0x19),
    (0x366D, 0x17),
    (0x33A6, 0x01),
    (0x306B, 0x05),
    (IMX274_TABLE_WAIT_MS, IMX274_WAIT_MS),
]


class Imx274_Mode(Enum):
    IMX274_MODE_3840X2160_60FPS = 0
    IMX274_MODE_1920X1080_60FPS = 1
    IMX274_MODE_3840X2160_60FPS_12BITS = 2
    Unknown = 3


frame_format = namedtuple(
    "FrameFormat", ["width", "height", "framerate", "pixel_format"]
)

imx_frame_format = []
imx_frame_format.insert(
    Imx274_Mode.IMX274_MODE_3840X2160_60FPS.value,
    frame_format(3840, 2160, 60, hololink.sensors.csi.PixelFormat.RAW_10),
)
imx_frame_format.insert(
    Imx274_Mode.IMX274_MODE_1920X1080_60FPS.value,
    frame_format(1920, 1080, 60, hololink.sensors.csi.PixelFormat.RAW_10),
)
imx_frame_format.insert(
    Imx274_Mode.IMX274_MODE_3840X2160_60FPS_12BITS.value,
    frame_format(3840, 2160, 60, hololink.sensors.csi.PixelFormat.RAW_12),
)
