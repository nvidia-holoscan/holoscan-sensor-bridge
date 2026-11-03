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

"""C++ sensor emulators: ``import hololink.emulation.sensors as hemusens``.

Re-exports the ``sensors`` submodule of the ``_emulation`` extension. A nanobind
submodule is an attribute of the extension, not an importable dotted path, hence the
attribute access rather than ``from .._emulation.sensors import ...``.
"""

from .. import _emulation as _ext

_sensors = _ext.sensors

IMX274Emulator = _sensors.IMX274Emulator
TEST_I2C_ADDRESS = _sensors.TEST_I2C_ADDRESS
TestPatternMode = _sensors.TestPatternMode
TestSensor = _sensors.TestSensor
Vb1940Emulator = _sensors.Vb1940Emulator

__all__ = [
    "IMX274Emulator",
    "TEST_I2C_ADDRESS",
    "TestPatternMode",
    "TestSensor",
    "Vb1940Emulator",
]
