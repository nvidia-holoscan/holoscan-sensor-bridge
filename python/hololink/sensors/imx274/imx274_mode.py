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

from collections import OrderedDict, namedtuple
from enum import Enum

import hololink

# values are on hex number system to be consistent with rest of the list
IMX274_TABLE_WAIT_MS = "0000"
IMX274_TABLE_END = "01"
IMX274_WAIT_MS = "01"
IMX274_WAIT_MS_START = "0f"

# Register addresses for camera properties. They only accept 8bits of value.

# Analog Gain
REG_AG_MSB = "300B"
REG_AG_LSB = "300A"

# Exposure
REG_EXP_MSB = "300D"
REG_EXP_LSB = "300C"

# Digital Gain
REG_DG = "3012"

imx274_start = OrderedDict(
    [
        ("3000", "00"),  # mode select streaming on
        ("303E", "02"),
        (IMX274_TABLE_WAIT_MS, IMX274_WAIT_MS_START),
        ("30F4", "00"),
        ("3018", "A2"),
        (IMX274_TABLE_END, IMX274_WAIT_MS_START),
    ]
)
imx274_start = [(int(x, 16), int(y, 16)) for x, y in imx274_start.items()]


imx274_stop = OrderedDict(
    [
        (IMX274_TABLE_WAIT_MS, IMX274_WAIT_MS),
        ("3000", "01"),  # mode select streaming off
        (IMX274_TABLE_END, "00"),
    ]
)
imx274_stop = [(int(x, 16), int(y, 16)) for x, y in imx274_stop.items()]

# test pattern
tp_colorbars = OrderedDict(
    [
        # test pattern
        ("303C", "11"),
        ("303D", "0B"),
        ("370B", "11"),
        ("370E", "00"),
        ("377F", "01"),
        ("3781", "01"),
    ]
)
tp_colorbars = [(int(x, 16), int(y, 16)) for x, y in tp_colorbars.items()]

# Mode : 3840X2160 10 bits 60fps
# value pairs use hex number system
imx274_mode_3840X2160_60fps = OrderedDict(
    [
        (IMX274_TABLE_WAIT_MS, IMX274_WAIT_MS),
        ("3000", "12"),
        ("3120", "F0"),
        ("3122", "02"),
        ("3129", "9c"),
        ("312A", "02"),
        ("312D", "02"),
        ("310B", "00"),
        ("304C", "00"),
        ("304D", "03"),
        ("331C", "1A"),
        ("3502", "02"),
        ("3529", "0E"),
        ("352A", "0E"),
        ("352B", "0E"),
        ("3538", "0E"),
        ("3539", "0E"),
        ("3553", "00"),
        ("357D", "05"),
        ("357F", "05"),
        ("3581", "04"),
        ("3583", "76"),
        ("3587", "01"),
        ("35BB", "0E"),
        ("35BC", "0E"),
        ("35BD", "0E"),
        ("35BE", "0E"),
        ("35BF", "0E"),
        ("366E", "00"),
        ("366F", "00"),
        ("3670", "00"),
        ("3671", "00"),
        ("30EE", "01"),
        ("3304", "32"),
        ("3306", "32"),
        ("3590", "32"),
        ("3686", "32"),
        # resolution */
        ("30E2", "01"),
        ("30F6", "07"),
        ("30F7", "01"),
        ("30F8", "C6"),
        ("30F9", "11"),
        ("3130", "78"),
        ("3131", "08"),
        ("3132", "70"),
        ("3133", "08"),
        # crop */
        ("30DD", "01"),
        ("30DE", "04"),
        ("30E0", "03"),
        ("3037", "01"),
        ("3038", "0C"),
        ("3039", "00"),
        ("303A", "0C"),
        ("303B", "0F"),
        # mode setting */
        ("3004", "01"),
        ("3005", "01"),
        ("3006", "00"),
        ("3007", "02"),
        ("300C", "0C"),
        ("300D", "00"),
        ("300E", "00"),
        ("3019", "00"),
        ("3A41", "08"),
        ("3342", "0A"),
        ("3343", "00"),
        ("3344", "16"),
        ("3345", "00"),
        ("3528", "0E"),
        ("3554", "1F"),
        ("3555", "01"),
        ("3556", "01"),
        ("3557", "01"),
        ("3558", "01"),
        ("3559", "00"),
        ("355A", "00"),
        ("35BA", "0E"),
        ("366A", "1B"),
        ("366B", "1A"),
        ("366C", "19"),
        ("366D", "17"),
        ("33A6", "01"),
        ("306B", "05"),
        (IMX274_TABLE_END, IMX274_WAIT_MS),
    ]
)
imx274_mode_3840X2160_60fps = [
    (int(x, 16), int(y, 16)) for x, y in imx274_mode_3840X2160_60fps.items()
]

# Mode : 1920x1080 10 bits 60fps
# value pairs use hex number system
imx274_mode_1920x1080_60fps = OrderedDict(
    [
        (IMX274_TABLE_WAIT_MS, IMX274_WAIT_MS),
        ("3000", "12"),  # mode select streaming on
        # input freq. 24M
        ("3120", "F0"),
        ("3122", "02"),
        ("3129", "9c"),
        ("312A", "02"),
        ("312D", "02"),
        ("310B", "00"),
        ("304C", "00"),
        ("304D", "03"),
        ("331C", "1A"),
        ("3502", "02"),
        ("3529", "0E"),
        ("352A", "0E"),
        ("352B", "0E"),
        ("3538", "0E"),
        ("3539", "0E"),
        ("3553", "00"),
        ("357D", "05"),
        ("357F", "05"),
        ("3581", "04"),
        ("3583", "76"),
        ("3587", "01"),
        ("35BB", "0E"),
        ("35BC", "0E"),
        ("35BD", "0E"),
        ("35BE", "0E"),
        ("35BF", "0E"),
        ("366E", "00"),
        ("366F", "00"),
        ("3670", "00"),
        ("3671", "00"),
        ("30EE", "01"),
        ("3304", "32"),
        ("3306", "32"),
        ("3590", "32"),
        ("3686", "32"),
        # resolution
        ("30E2", "02"),
        ("30F6", "04"),
        ("30F7", "01"),
        ("30F8", "0C"),
        ("30F9", "12"),
        ("3130", "40"),
        ("3131", "04"),
        ("3132", "38"),
        ("3133", "04"),
        # crop
        ("30DD", "01"),
        ("30DE", "07"),
        ("30DF", "00"),
        ("30E0", "04"),
        ("30E1", "00"),
        ("3037", "01"),
        ("3038", "0C"),
        ("3039", "00"),
        ("303A", "0C"),
        ("303B", "0F"),
        # mode setting
        ("3004", "02"),
        ("3005", "21"),
        ("3006", "00"),
        ("3007", "B1"),
        ("300C", "08"),  # SHR: Minimum 8
        ("300D", "00"),
        ("3019", "00"),
        ("3A41", "08"),
        ("3342", "0A"),
        ("3343", "00"),
        ("3344", "1A"),
        ("3345", "00"),
        ("3528", "0E"),
        ("3554", "00"),
        ("3555", "01"),
        ("3556", "01"),
        ("3557", "01"),
        ("3558", "01"),
        ("3559", "00"),
        ("355A", "00"),
        ("35BA", "0E"),
        ("366A", "1B"),
        ("366B", "1A"),
        ("366C", "19"),
        ("366D", "17"),
        ("33A6", "01"),
        ("306B", "05"),
        (IMX274_TABLE_END, IMX274_WAIT_MS),
    ]
)
imx274_mode_1920x1080_60fps = [
    (int(x, 16), int(y, 16)) for x, y in imx274_mode_1920x1080_60fps.items()
]


class Imx274_Mode(Enum):
    IMX274_MODE_3840X2160_60FPS = 0
    IMX274_MODE_1920X1080_60FPS = 1
    Unknown = 2


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
