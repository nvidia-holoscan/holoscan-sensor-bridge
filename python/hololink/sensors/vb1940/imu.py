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

import hololink as hololink_module


# This driver provides a access to the streaming IMU present
# in VB1940 units; where data is streamed from the sensor
# through an RDMA data channel and can be observed in
# GPU memory.  See the file examples/vb1940_imu_player.py
# for an example of fetching data from this sensor;
# examples/vb1940_stereo_imu_player.py adds live streams
# from both left and right synchronized cameras, all of
# which are available in an HSDK pipeline.
class Imu:
    INT3_INT4_IO_MAP = 0x18
    INT3_INT4_IO_CONF = 0x16
    GYRO_INT_CTRL = 0x15

    ACC_PWR_CTRL = 0x7D
    INT1_IO_CONF = 0x53
    INT2_IO_CONF = 0x54
    INT1_INT2_MAP = 0x58
    ACC_CONF = 0x40
    GYRO_CONF = 0x10

    ACC_I2C_ADDR = 0x19
    GYRO_I2C_ADDR = 0x69

    STREAM_I2C_BASE_ADDR = 0x8010_0000

    ACC_RATE_MAP = {
        12.5: 0xA5,
        25.0: 0xA6,
        50.0: 0xA7,
        100.0: 0xA8,
        200.0: 0xA9,
        400.0: 0xAA,
        800.0: 0xAB,
        1600.0: 0xAC,
    }

    GYRO_RATE_MAP = {
        100: 0x07,
        200: 0x06,
        400: 0x03,
        1000: 0x02,
        2000: 0x01,
    }

    def __init__(
        self,
        hololink,
        i2c_controller_address=0x3,
    ):
        self._hololink = hololink
        self._i2c = hololink.get_i2c(i2c_controller_address)
        self._samples_per_frame = None

    def _i2c_command(self, i2c_addr, write_bytes=None, read_byte_count=0):
        write_array = bytearray(100)
        serializer = hololink_module.Serializer(write_array)
        if write_bytes is not None:
            serializer.append_buffer(bytearray(write_bytes))
        return self._i2c.i2c_transaction(i2c_addr, serializer.data(), read_byte_count)

    def configure(self, samples_per_frame, accelerometer_rate, gyroscope_rate):
        # using float as a key is tricky (because comparing floats is tricky)
        # so let's normalize the input carefully to ensure a match
        key = round(float(accelerometer_rate), 1)
        acc_rate = self.ACC_RATE_MAP.get(key, None)
        if acc_rate is None:
            raise Exception(f"Invalid {accelerometer_rate=}")
        gyro_rate = self.GYRO_RATE_MAP.get(int(gyroscope_rate), None)
        if gyro_rate is None:
            raise Exception(f"Invalid {gyroscope_rate=}")
        self._i2c_command(self.GYRO_I2C_ADDR, [self.INT3_INT4_IO_MAP, 0x81])
        self._i2c_command(self.GYRO_I2C_ADDR, [self.INT3_INT4_IO_CONF, 0x04])
        self._i2c_command(self.GYRO_I2C_ADDR, [self.GYRO_INT_CTRL, 0x80])
        self._i2c_command(self.ACC_I2C_ADDR, [self.ACC_PWR_CTRL, 0x04])
        self._i2c_command(self.ACC_I2C_ADDR, [self.INT1_IO_CONF, 0x0A])
        self._i2c_command(self.ACC_I2C_ADDR, [self.INT2_IO_CONF, 0x0A])
        self._i2c_command(self.ACC_I2C_ADDR, [self.INT1_INT2_MAP, 0x44])
        self._i2c_command(self.ACC_I2C_ADDR, [self.ACC_CONF, acc_rate])
        self._i2c_command(self.GYRO_I2C_ADDR, [self.GYRO_CONF, gyro_rate])
        #
        self._samples_per_frame = samples_per_frame

    def stream_imu_data(self):
        if self._samples_per_frame is None:
            raise ValueError(
                "Can't enable an unconfigured IMU; call configure(...) on this object."
            )
        self._hololink.write_uint32(
            self.STREAM_I2C_BASE_ADDR + 0x04, 0x00000000
        )  # Bus select
        self._hololink.write_uint32(
            self.STREAM_I2C_BASE_ADDR + 0x08, 0x3 + (0x6 << 16)
        )  # 16'h NUM_RD=6, 16'h NUM_WR=16'h
        self._hololink.write_uint32(
            self.STREAM_I2C_BASE_ADDR + 0x0C, 0x00000019
        )  # Clock Count
        self._hololink.write_uint32(
            self.STREAM_I2C_BASE_ADDR + 0x10, 0x004C4B40
        )  # timeout
        # Data
        wr_cmd_acc = (
            (self.ACC_I2C_ADDR << 1)
            | (0x12 << 8)
            | (((self.ACC_I2C_ADDR << 1) | 0x1) << 16)
        )
        self._hololink.write_uint32(self.STREAM_I2C_BASE_ADDR + 0x100, wr_cmd_acc)
        wr_cmd_gyro = (
            (self.GYRO_I2C_ADDR << 1)
            | (0x02 << 8)
            | (((self.GYRO_I2C_ADDR << 1) | 0x1) << 16)
        )
        self._hololink.write_uint32(self.STREAM_I2C_BASE_ADDR + 0x110, wr_cmd_gyro)
        # Jump Table
        self._hololink.write_uint32(
            self.STREAM_I2C_BASE_ADDR + 0x18,
            0x1000 + ((self._samples_per_frame - 1) << 16),
        )
        # Start
        self._hololink.write_uint32(
            self.STREAM_I2C_BASE_ADDR + 0x00, 0x000C
        )  # Dev address and Start i2c Transaction Streaming with internal Device address

    def stop_imu_data(self):
        self._hololink.write_uint32(
            self.STREAM_I2C_BASE_ADDR + 0x00, 0x0
        )  # Stop responding to IMU Interrupts

    def start(self):
        self.stream_imu_data()

    def stop(self):
        self.stop_imu_data()

    def samples_per_frame(self):
        if self._samples_per_frame is None:
            raise ValueError(
                "Can't enable an unconfigured IMU; call configure(...) on this object."
            )
        return self._samples_per_frame
