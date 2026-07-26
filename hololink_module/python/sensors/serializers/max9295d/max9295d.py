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

CAM_I2C_BUS = 1


class Max9295d:

    I2C_ADDRESS_DEFAULT = 0x40

    DEV_ID_REG = 0x000D
    DEV_ID = 0x95

    # Control Channel address-translation registers.
    CC_SRC_A_REG = 0x0042
    CC_DST_A_REG = 0x0043
    CC_SRC_B_REG = 0x0044
    CC_DST_B_REG = 0x0045

    # GPIO blocks. Per-pin stride is 3 registers (GPIO_A / GPIO_B / GPIO_C).
    # Pin N base address = GPIO_PIN_BASE + 3 * N
    GPIO_PIN_BASE = 0x02BE

    def __init__(self, hololink, i2c_address=None):
        self._i2c_address = (
            i2c_address if i2c_address is not None else self.I2C_ADDRESS_DEFAULT
        )
        self._i2c_bus = CAM_I2C_BUS

        self._i2c = hololink.get_i2c(self._i2c_bus, self._i2c_address)

    def get_i2c_address(self):
        return self._i2c_address

    def set_i2c_address(self, i2c_address):
        self.set_register(0x0000, i2c_address << 1)
        self._i2c_address = i2c_address

    def get_register(self, register, timeout=None):
        write_bytes = struct.pack(">H", register)
        read_byte_count = 1
        reply = self._i2c.i2c_transaction(
            self._i2c_address,
            write_bytes,
            read_byte_count,
        )
        return reply[0]

    def configure_cc_address_translation(self, src_a, dst_a, src_b, dst_b):
        self.set_register(self.CC_SRC_A_REG, src_a << 1)
        self.set_register(self.CC_DST_A_REG, dst_a << 1)
        self.set_register(self.CC_SRC_B_REG, src_b << 1)
        self.set_register(self.CC_DST_B_REG, dst_b << 1)

    def route_gmsl_gpio_to_pin(self, pin, rx_id):
        base = self.GPIO_PIN_BASE + 3 * pin
        self.set_register(base, 0x04)  # GPIO_A: GPIO_RX_EN=1
        self.set_register(base + 1, 0x60)  # GPIO_B: push-pull, pull-up
        self.set_register(base + 2, rx_id & 0x1F)  # GPIO_C: GPIO_RX_ID

    def set_register(self, register, value, timeout=None):
        write_bytes = struct.pack(">H", register) + struct.pack(">B", value)
        read_byte_count = 0
        try:
            self._i2c.i2c_transaction(
                self._i2c_address,
                write_bytes,
                read_byte_count,
            )
        except RuntimeError as e:
            logging.error(
                "%s - Failed: i2c_addr=0x%02X, register=0x%04X, value=0x%02X",
                e,
                self._i2c_address,
                register,
                value,
            )
            raise
