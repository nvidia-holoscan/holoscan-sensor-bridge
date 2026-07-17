# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Holoscan-coupled operators that consume hololink_module V1 interfaces.

Per the per-module Python sub-package convention, the operator
bindings live in their own pybind extension
(``_hololink_module_operators.so``) rather than the core
``_hololink_py_module`` extension, so applications that don't link
Holoscan never load this module.
"""

# Holoscan's Python pybind module must be imported before the
# operator extension loads — it's the module that registers
# holoscan::Operator (and holoscan::Fragment / Condition / Resource)
# with pybind11's type system, which our operator's pybind class
# references as a base type.
import holoscan  # noqa: F401 — side-effect import for pybind type registration

from . import _hololink_module_operators as _native

# Re-export whatever operators this build registered. Each operator is
# compiled into the extension only when the environment supports it
# (e.g. RoceReceiverOp requires a RoCE-capable build), so the set of
# available names varies by configuration; importing this package always
# succeeds, and an operator appears here iff its capability was built.
__all__ = [_name for _name in dir(_native) if not _name.startswith("_")]
for _name in __all__:
    globals()[_name] = getattr(_native, _name)
