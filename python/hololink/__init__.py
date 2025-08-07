# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from . import (
    emulation,
    operators,
    renesas_bajoran_lite_ts1,
    renesas_bajoran_lite_ts2,
    sensors,
)
from ._hololink import (
    APB_RAM,
    BL_I2C_BUS,
    CAM_I2C_BUS,
    CLNX_SPI_BUS,
    CPNX_SPI_BUS,
    CTRL_EVENT,
    CTRL_EVT_SW_EVENT,
    DP_ADDRESS_0,
    DP_ADDRESS_1,
    DP_ADDRESS_2,
    DP_ADDRESS_3,
    DP_BUFFER_LENGTH,
    DP_BUFFER_MASK,
    DP_HOST_IP,
    DP_HOST_MAC_HIGH,
    DP_HOST_MAC_LOW,
    DP_HOST_UDP_PORT,
    DP_PACKET_SIZE,
    DP_QP,
    DP_RKEY,
    DP_VP_MASK,
    FPGA_DATE,
    HOLOLINK_100G_BOARD_ID,
    HOLOLINK_LITE_BOARD_ID,
    HOLOLINK_NANO_BOARD_ID,
    HSB_IP_VERSION,
    I2C_10B_ADDRESS,
    I2C_BUSY,
    I2C_CTRL,
    I2C_DONE,
    I2C_FSM_ERR,
    I2C_I2C_ERR,
    I2C_I2C_NAK,
    I2C_REG_BUS_EN,
    I2C_REG_CLK_CNT,
    I2C_REG_CONTROL,
    I2C_REG_DATA_BUFFER,
    I2C_REG_NUM_BYTES,
    I2C_REG_STATUS,
    I2C_START,
    LEOPARD_EAGLE_BOARD_ID,
    METADATA_SIZE,
    MICROCHIP_POLARFIRE_BOARD_ID,
    RD_DWORD,
    REQUEST_FLAGS_ACK_REQUEST,
    RESPONSE_INVALID_CMD,
    RESPONSE_SUCCESS,
    SPI_CTRL,
    WR_DWORD,
    AD9986Config,
    BasicEnumerationStrategy,
    BayerFormat,
    CsiConverter,
    DataChannel,
    EnumerationStrategy,
    Enumerator,
    Hololink,
    ImGuiRenderer,
    Metadata,
    NvtxTrace,
    PixelFormat,
    Sequencer,
    Synchronizable,
    Synchronizer,
    Timeout,
    TimeoutError,
    UnsupportedVersion,
    get_traditional_i2c,
    get_traditional_spi,
)
from .hololink_core import (
    PAGE_SIZE,
    UDP_PACKET_SIZE,
    ArpWrapper,
    Deserializer,
    Serializer,
    infiniband_devices,
    local_ip_and_mac,
    local_ip_and_mac_from_socket,
    local_mac,
    round_up,
)

__all__ = [
    "AD9986Config",
    "APB_RAM",
    "ArpWrapper",
    "BL_I2C_BUS",
    "BasicEnumerationStrategy",
    "BayerFormat",
    "CAM_I2C_BUS",
    "CLNX_SPI_BUS",
    "CTRL_EVENT",
    "CTRL_EVT_SW_EVENT",
    "CPNX_SPI_BUS",
    "CsiConverter",
    "DP_ADDRESS_0",
    "DP_ADDRESS_1",
    "DP_ADDRESS_2",
    "DP_ADDRESS_3",
    "DP_BUFFER_LENGTH",
    "DP_BUFFER_MASK",
    "DP_HOST_IP",
    "DP_HOST_MAC_HIGH",
    "DP_HOST_MAC_LOW",
    "DP_HOST_UDP_PORT",
    "DP_PACKET_SIZE",
    "DP_QP",
    "DP_RKEY",
    "DP_VP_MASK",
    "DataChannel",
    "Deserializer",
    "emulation",
    "EnumerationStrategy",
    "Enumerator",
    "FPGA_DATE",
    "HOLOLINK_100G_BOARD_ID",
    "HOLOLINK_LITE_BOARD_ID",
    "HOLOLINK_NANO_BOARD_ID",
    "HSB_IP_VERSION",
    "Hololink",
    "I2C_10B_ADDRESS",
    "I2C_CTRL",
    "I2C_BUSY",
    "I2C_CORE_EN",
    "I2C_DONE",
    "I2C_FSM_ERR",
    "I2C_I2C_ERR",
    "I2C_I2C_NAK",
    "I2C_REG_BUS_EN",
    "I2C_REG_CLK_CNT",
    "I2C_REG_CONTROL",
    "I2C_REG_DATA_BUFFER",
    "I2C_REG_NUM_BYTES",
    "I2C_REG_STATUS",
    "I2C_START",
    "ImGuiRenderer",
    "LEOPARD_EAGLE_BOARD_ID",
    "METADATA_SIZE",
    "MICROCHIP_POLARFIRE_BOARD_ID",
    "Metadata",
    "NvtxTrace",
    "PAGE_SIZE",
    "PixelFormat",
    "RD_DWORD",
    "REQUEST_FLAGS_ACK_REQUEST",
    "RESPONSE_INVALID_CMD",
    "RESPONSE_SUCCESS",
    "SPI_CTRL",
    "Sequencer",
    "Serializer",
    "Synchronizable",
    "Synchronizer",
    "Timeout",
    "TimeoutError",
    "UDP_PACKET_SIZE",
    "UnsupportedVersion",
    "WR_DWORD",
    "core",
    "get_traditional_i2c",
    "get_traditional_spi",
    "infiniband_devices",
    "local_ip_and_mac",
    "local_ip_and_mac_from_socket",
    "local_mac",
    "operators",
    "renesas_bajoran_lite_ts1",
    "renesas_bajoran_lite_ts2",
    "round_up",
    "sensors",
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
