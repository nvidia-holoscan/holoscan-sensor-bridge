# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import importlib

# NOTE: This file should be kept in sync with src/hololink/emulation/hololink/emulation/__init__.py
# Both files support different packaging flows (emulator-only vs full SDK)
# Changes to the public API should be reflected in both locations
import sys

from ._emulation import (
    HSB_EMULATOR_CONFIG,
    HSB_LEOPARD_EAGLE_CONFIG,
    COEDataPlane,
    DataPlane,
    HSBConfiguration,
    HSBEmulator,
    I2CController,
    I2CPeripheral,
    I2CStatus,
    IPAddress,
    LinuxDataPlane,
)

__all__ = [
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
    "LinuxDataPlane",
    "sensors",
]

_SUBMODULES = {
    "sensors": "sensors",
}


# Lazily load submodules.
# this is needed to avoid circular imports from loading Vb1940Emulator, which depends on both I2CPeripheral and I2CController
def __getattr__(attr):
    if attr in _SUBMODULES:
        module_name = ".".join([__name__, _SUBMODULES[attr]])
        if module_name in sys.modules:  # cached
            module = sys.modules[module_name]
        else:
            module = importlib.import_module(module_name)  # import
            sys.modules[module_name] = module  # cache
        return module
    raise AttributeError(f"module {__name__} has no attribute {attr}")
