# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter-side VB1940 camera driver bindings.

Per the per-module Python sub-package convention, the camera type
ships in its own pybind extension under
``hololink_module.sensors.vb1940`` rather than the core
``_hololink_py_module`` extension, so applications that don't drive a
VB1940 camera never load this module.
"""

# The core extension must be imported before this one — it registers
# the V1 base types (Module, EnumerationMetadata, Vsync) the camera
# binding depends on.
import hololink_module  # noqa: F401 — side-effect import for V1 type registration

from . import _hololink_module_vb1940 as _native

Vb1940Cam = _native.Vb1940Cam
Vb1940_Mode = _native.Vb1940_Mode
PixelFormat = _native.PixelFormat
BayerFormat = _native.BayerFormat

__all__ = [
    "Vb1940Cam",
    "Vb1940_Mode",
    "PixelFormat",
    "BayerFormat",
]
