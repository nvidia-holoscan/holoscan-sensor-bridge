# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys

# Define operator modules and classes for lazy loading
_MODULES = [
    "imx274",
    "camera",
    "csi",
    "ecam0m30tof",
    "edepth",
    "imx477",
    "imx715",
    "vb1940",
]

# Sensor/I2CExpanderOutputEN/LII2CExpander come from _hololink_sensor, a pybind11 module
# sensors/CMakeLists.txt doesn't build today (broken against the installed HSDK5 holoscan-sdk --
# see that file's own comment). Lazily loading them through __getattr__ below, instead of the
# eager `from . import _hololink_sensor as sensor` this used to do, means importing
# hololink.sensors itself still succeeds; only actually touching one of these three names fails
# (with the normal ModuleNotFoundError for _hololink_sensor, not an import-time crash for every
# caller regardless of whether they need them -- e.g. ecam0m30tof, which has its own local
# li_i2c_expander.py and never touches any of the three).
_OBJECTS = {
    "Sensor": "_hololink_sensor",
    "I2CExpanderOutputEN": "_hololink_sensor",
    "LII2CExpander": "_hololink_sensor",
}

__all__ = [
    "csi",
    "Sensor",
    "I2CExpanderOutputEN",
    "LII2CExpander",
]

__all__.extend(_MODULES)
__all__.extend(_OBJECTS.keys())


# Autocomplete
def __dir__():
    return __all__


# Lazily load modules and classes
def __getattr__(attr):
    if attr in _OBJECTS:
        module_name = ".".join([__name__, _OBJECTS[attr]])
        if module_name in sys.modules:  # cached
            module = sys.modules[module_name]
        else:
            module = importlib.import_module(module_name)  # import
            sys.modules[module_name] = module  # cache
        return getattr(module, attr)
    if attr in _MODULES:
        module_name = f"{__name__}.{attr}"
        if module_name in sys.modules:  # cached
            module = sys.modules[module_name]
        else:
            module = importlib.import_module(module_name)  # import
            sys.modules[module_name] = module  # cache
        return module
    raise AttributeError(f"module {__name__} has no attribute {attr}")
