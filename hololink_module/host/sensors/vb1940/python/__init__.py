# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter-side VB1940 camera driver bindings.

Vb1940Cam is bound in the single consolidated hololink_module extension
(python/hololink_module/_hololink_module.cpp) alongside the core V1
service types it depends on (Module, EnumerationMetadata, Vsync), so
importing it here also guarantees those are already registered.
"""

from hololink_module import _hololink_module as _native

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
