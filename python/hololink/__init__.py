# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from . import native, operators, renesas_bajoran_lite_ts1, sensors
from .hololink import (
    BL_I2C_CTRL,
    CAM_I2C_CTRL,
    CLNX_SPI_CTRL,
    CPNX_SPI_CTRL,
    FPGA_DATE,
    FPGA_VERSION,
    I2C_BUSY,
    I2C_CORE_EN,
    I2C_DONE,
    I2C_DONE_CLEAR,
    I2C_START,
    RD_DWORD,
    REQUEST_FLAGS_ACK_REQUEST,
    RESPONSE_INVALID_CMD,
    RESPONSE_SUCCESS,
    WR_DWORD,
    Hololink,
    HololinkDataChannel,
    HololinkEnumerator,
    HololinkTimeoutError,
    retry,
    timeout,
)
from .native import Deserializer, Serializer

__all__ = [
    "BL_I2C_CTRL",
    "CAM_I2C_CTRL",
    "CLNX_SPI_CTRL",
    "CPNX_SPI_CTRL",
    "Deserializer",
    "FPGA_DATE",
    "FPGA_VERSION",
    "I2C_BUSY",
    "I2C_CORE_EN",
    "I2C_DONE",
    "I2C_DONE_CLEAR",
    "I2C_START",
    "RD_DWORD",
    "REQUEST_FLAGS_ACK_REQUEST",
    "RESPONSE_INVALID_CMD",
    "RESPONSE_SUCCESS",
    "Serializer",
    "WR_DWORD",
    "Hololink",
    "HololinkDataChannel",
    "HololinkEnumerator",
    "HololinkTimeoutError",
    "native",
    "operators",
    "renesas_bajoran_lite_ts1",
    "retry",
    "sensors",
    "timeout",
]

import logging

trace_level = logging.DEBUG - 5
trace_name = "TRACE"
logging.addLevelName(trace_level, trace_name)
setattr(logging, trace_name, trace_level)


def log_trace(self, message, *args, **kwargs):
    if self.isEnabledFor(trace_level):
        self._log(trace_level, message, *args, **kwargs)


setattr(logging.getLoggerClass(), trace_name.lower(), log_trace)


def log_trace_to_root(message, *args, **kwargs):
    logging.log(trace_level, message, *args, **kwargs)


setattr(logging, trace_name.lower(), log_trace_to_root)

log_format = "%(levelname)s %(relativeCreated)d %(funcName)s %(filename)s:%(lineno)d tid=%(threadName)s -- %(message)s"


def logging_level(n):
    logging.basicConfig(format=log_format, level=n)
