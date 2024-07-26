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

import importlib
import sys

# Define operator modules and classes for lazy loading
_MODULES = [
    "imx274",
    "udp_cam",
    "csi",
]

_OBJECTS = {
    "LinuxCamera": "linux_camera",
}

__all__ = [
    "csi",
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
