# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HSB-Lite supplement bindings.

Per the per-module Python sub-package convention, the supplement
type lives in its own pybind extension under
``hololink_module.hsb_lite`` rather than the core
``_hololink_py_module`` extension, so applications that don't need
HSB-Lite-specific surface never load this module.
"""

# The core extension must be imported before this one — it
# registers the V1 base types HsbLiteInterfaceV1 depends on
# (Module, EnumerationMetadata). The pybind module's entry point
# also imports it, but doing so here guarantees ordering even when
# applications import this supplement before triggering the .so
# load.
import hololink_module  # noqa: F401 — side-effect import for V1 type registration

from . import _hololink_module_hsb_lite as _native

HsbLiteInterfaceV1 = _native.HsbLiteInterfaceV1

__all__ = [
    "HsbLiteInterfaceV1",
]
