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

"""HSB emulator: ``import hololink.emulation as hemu``.

One nanobind extension (``_emulation``) provides both this module and the
``hololink.emulation.sensors`` submodule, so the C++ sensor emulators derive from the
``I2CPeripheral`` bound here without any cross-module type lookup.
"""

from . import sensors
from ._emulation import (
    HSB_EMULATOR_CONFIG,
    HSB_LEOPARD_EAGLE_CONFIG,
    AddressValuePair,
    COEDataPlane,
    DataPlane,
    HSBConfiguration,
    HSBEmulator,
    I2CController,
    I2CPeripheral,
    I2CStatus,
    IPAddress,
    RoCEv2DataPlane,
    hsb_config_set_uuid,
)

# The Linux UDP receiver consumes the same RoCEv2 packet stream.
LinuxDataPlane = RoCEv2DataPlane

__all__ = [
    "AddressValuePair",
    "COEDataPlane",
    "DataPlane",
    "HSB_EMULATOR_CONFIG",
    "HSB_LEOPARD_EAGLE_CONFIG",
    "HSBConfiguration",
    "HSBEmulator",
    "I2CController",
    "I2CPeripheral",
    "I2CStatus",
    "IPAddress",
    "RoCEv2DataPlane",
    "LinuxDataPlane",
    "hsb_config_set_uuid",
    "sensors",
]
