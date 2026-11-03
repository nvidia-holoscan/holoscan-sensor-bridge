# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Minimal ``hololink`` package of the emulator-only build.

Only ``hololink.emulation`` and the HSB register constants an emulator client needs are
provided; none of the other hololink libraries are present. The constants come from the
extension itself (hsb_config.hpp), so they match the emulator and the names the full
hololink package exports.
"""

from . import emulation
from .emulation._emulation import (
    BL_I2C_BUS,
    CAM_I2C_BUS,
    I2C_BUSY,
    I2C_CTRL,
    I2C_DONE,
    I2C_FSM_ERR,
    I2C_I2C_ERR,
    I2C_REG_BUS_EN,
    I2C_REG_CLK_CNT,
    I2C_REG_CONTROL,
    I2C_REG_DATA_BUFFER,
    I2C_REG_NUM_BYTES,
    I2C_REG_STATUS,
    I2C_START,
)

__all__ = [
    "BL_I2C_BUS",
    "CAM_I2C_BUS",
    "I2C_BUSY",
    "I2C_CTRL",
    "I2C_DONE",
    "I2C_FSM_ERR",
    "I2C_I2C_ERR",
    "I2C_REG_BUS_EN",
    "I2C_REG_CLK_CNT",
    "I2C_REG_CONTROL",
    "I2C_REG_DATA_BUFFER",
    "I2C_REG_NUM_BYTES",
    "I2C_REG_STATUS",
    "I2C_START",
    "emulation",
]
