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

from hololink_module.sensors.ar0234.ar0234 import (
    CAM_A_I2C_ADDRESS,
    CAM_B_I2C_ADDRESS,
    Ar0234Cam,
)
from hololink_module.sensors.serializers.max9295d import Max9295d


class Hawk:
    """A Hawk module: one MAX9295D serializer + two AR0234 sensors (left and
    right), all on one GMSL link to the deserializer."""

    SENSOR_LEFT_I2C_ADDRESS = CAM_A_I2C_ADDRESS  # 0x10, wired to MIPI Port A
    SENSOR_RIGHT_I2C_ADDRESS = CAM_B_I2C_ADDRESS  # 0x18, wired to MIPI Port B

    def __init__(self, hololink, skip_i2c=False):
        self.serializer = Max9295d(hololink)
        self.sensor_left = Ar0234Cam(
            hololink,
            i2c_address=self.SENSOR_LEFT_I2C_ADDRESS,
            skip_i2c=skip_i2c,
        )
        self.sensor_right = Ar0234Cam(
            hololink,
            i2c_address=self.SENSOR_RIGHT_I2C_ADDRESS,
            skip_i2c=skip_i2c,
        )

    @property
    def sensors(self):
        return (self.sensor_left, self.sensor_right)

    # ---- Per-sensor operations (broadcast to both AR0234s) ----

    def configure(self, mode, fsync=False):
        for sensor in self.sensors:
            sensor.configure_camera(mode, fsync)

    def set_mode(self, mode):
        for sensor in self.sensors:
            sensor.set_mode(mode)

    def set_exposure_reg(self, value):
        for sensor in self.sensors:
            sensor.set_exposure_reg(value)

    def test_pattern(self, pattern):
        for sensor in self.sensors:
            sensor.test_pattern(pattern)

    def start(self):
        for sensor in self.sensors:
            sensor.start()

    def stop(self):
        for sensor in self.sensors:
            sensor.stop()

    # ---- Format/dims --- both sensors are identical AR0234s ----

    def configure_converter(self, converter):
        # Both AR0234s share identical geometry; configuring via sensor_left
        # is correct as long as both sensors are programmed in the same mode.
        self.sensor_left.configure_converter(converter)

    def pixel_format(self):
        return self.sensor_left.pixel_format()

    def bayer_format(self):
        return self.sensor_left.bayer_format()

    @property
    def _width(self):
        return self.sensor_left._width

    @property
    def _height(self):
        return self.sensor_left._height

    # ---- Serializer address delegation ----

    def get_serializer_i2c_address(self):
        return self.serializer.get_i2c_address()

    def set_serializer_i2c_address(self, i2c_address):
        self.serializer.set_i2c_address(i2c_address)

    # ---- Sensor I2C address remap (via serializer CC translation) ----

    def remap_sensor_addresses(self, new_left, new_right):
        # All four CC translation registers are written before updating the
        # Python-side addresses, so a partial write failure (RuntimeError)
        # leaves _i2c_address at the pre-remap values — consistent with the
        # hardware not being fully reconfigured.
        self.serializer.configure_cc_address_translation(
            src_a=new_left,
            dst_a=self.SENSOR_LEFT_I2C_ADDRESS,
            src_b=new_right,
            dst_b=self.SENSOR_RIGHT_I2C_ADDRESS,
        )
        self.sensor_left.set_i2c_address(new_left)
        self.sensor_right.set_i2c_address(new_right)
