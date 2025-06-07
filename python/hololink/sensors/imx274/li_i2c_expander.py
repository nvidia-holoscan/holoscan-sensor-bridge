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

from enum import Enum

import hololink


class I2C_Expander_Output_EN(Enum):
    OUTPUT_1 = 0b0001  # for first camera
    OUTPUT_2 = 0b0010  # for another camera
    OUTPUT_3 = 0b0100
    OUTPUT_4 = 0b1000
    default = 0b0000


class LII2CExpander:
    I2C_EXPANDER_ADDRESS = 0b01110000

    def __init__(self, hololink, i2c_address):
        self._i2c = hololink.get_i2c(i2c_address)

    def configure(self, output_en=I2C_Expander_Output_EN.default.value):
        write_bytes = bytearray(100)
        serializer = hololink.Serializer(write_bytes)
        serializer.append_uint8(output_en)  # turn the i2c expander to enable output
        read_byte_count = 0
        self._i2c.i2c_transaction(
            self.I2C_EXPANDER_ADDRESS,
            serializer.data(),
            read_byte_count,
        )

    def synchronized_configure(self, sequencer, output_en):
        write_bytes = bytearray(100)
        serializer = hololink.Serializer(write_bytes)
        serializer.append_uint8(output_en)  # turn the i2c expander to enable output
        read_byte_count = 0
        self._i2c.encode_i2c_request(
            sequencer,
            self.I2C_EXPANDER_ADDRESS,
            serializer.data(),
            read_byte_count,
        )
