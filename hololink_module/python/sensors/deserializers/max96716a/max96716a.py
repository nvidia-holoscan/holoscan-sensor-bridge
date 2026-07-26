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
from enum import Enum

CAM_I2C_BUS = 1


class Max96716a:

    I2C_ADDRESS = 0x28

    DEV_ID_REG = 0x000D
    DEV_ID = 0xBE

    VIDEO_PIPE_SEL = 0x0161

    # GPIO blocks. Per-pin stride is 3 registers (GPIO_A / GPIO_B / GPIO_C).
    # Pin N base = GPIO_PIN_BASE + 3 * N
    GPIO_PIN_BASE = 0x02B0
    GPIO_ALT_PAGE_OFFSET = 0x5000

    class GmslLink(Enum):
        LINK_A = 0
        LINK_B = 1

    class VideoPipe(Enum):
        PIPE_Y = 0
        PIPE_Z = 1

    def __init__(
        self,
        hololink,
    ):
        self._i2c_address = self.I2C_ADDRESS
        self._i2c_bus = CAM_I2C_BUS

        self._i2c = hololink.get_i2c(self._i2c_bus, self._i2c_address)

    def get_register(self, register, timeout=None):
        write_bytes = struct.pack(">H", register)
        read_byte_count = 1
        reply = self._i2c.i2c_transaction(
            self._i2c_address,
            write_bytes,
            read_byte_count,
        )
        return reply[0]

    def set_register(self, register, value, i2c_address=None, timeout=None):
        if i2c_address is None:
            i2c_address = self._i2c_address
        write_bytes = struct.pack(">H", register) + struct.pack(">B", value)
        read_byte_count = 0
        try:
            self._i2c.i2c_transaction(
                i2c_address,
                write_bytes,
                read_byte_count,
            )
        except RuntimeError as e:
            logging.error(
                "%s - Failed: i2c_addr=0x%02X, register=0x%04X, value=0x%02X",
                e,
                i2c_address,
                register,
                value,
            )
            raise

    def enable_link_exclusive(self, link):
        if link == self.GmslLink.LINK_A:
            self.set_register(0x0F00, 0x01)
        elif link == self.GmslLink.LINK_B:
            self.set_register(0x0F00, 0x02)
        else:
            raise Exception(f"Non-existent link {link}")
        time.sleep(250 / 1000)

    def enable_both_links(self):
        self.set_register(0x0F00, 0x03)
        self.set_register(0x0010, 0x03)  # Reverse splitter mode. Both links enabled.
        time.sleep(250 / 1000)

    def stream_id_to_pipe_mapping(self, link, stream_id, pipe):
        if not isinstance(link, self.GmslLink):
            raise TypeError(f"link must be a Max96716a.GmslLink, got {link!r}")
        if not isinstance(pipe, self.VideoPipe):
            raise TypeError(f"pipe must be a Max96716a.VideoPipe, got {pipe!r}")
        if not isinstance(stream_id, int) or not 0 <= stream_id < 4:
            raise ValueError(f"stream_id must be an int in 0..3, got {stream_id!r}")
        stream_id = stream_id + 4 if link == self.GmslLink.LINK_B else stream_id
        pipe_value = stream_id << 3 if pipe == self.VideoPipe.PIPE_Z else stream_id
        return pipe_value

    def configure_video_pipe(self):
        self.set_register(0x0330, 0x04)  # MIPI 2x4: Port A, PHY0/1; Port B, PHY2/3.
        self.set_register(0x0474, 0x08)  # MIPI TX1: 2 data lanes
        self.set_register(0x04B4, 0x08)  # MIPI TX2: 2 data lanes

    def route_pin_to_gmsl_gpio(self, link, pin, tx_id):
        if not isinstance(link, self.GmslLink):
            raise TypeError(f"link must be a Max96716a.GmslLink, got {link!r}")
        page_offset = 0 if link == self.GmslLink.LINK_A else self.GPIO_ALT_PAGE_OFFSET
        base = page_offset + self.GPIO_PIN_BASE + 3 * pin
        self.set_register(base, 0x02)  # GPIO_A: GPIO_TX_EN=1
        self.set_register(
            base + 1, 0x60 | (tx_id & 0x1F)
        )  # GPIO_B: push-pull, pull-up, GPIO_TX_ID
        self.set_register(base + 2, tx_id & 0x1F)  # GPIO_C: GPIO_RX_ID
