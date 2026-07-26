# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Python surface for hololink_module.

Interfaces are exported under their explicit version name (e.g.
``HololinkInterfaceV1``). There is no unversioned alias: the version an
application depends on is named at the import site, so rebuilding never
silently moves an application onto a newer interface a module may not
support. Adopting a future version is a deliberate edit
(``HololinkInterfaceV1`` -> ``HololinkInterfaceV2``).
"""

import os
import sys

# Load the core extension with RTLD_GLOBAL so its C++ typeinfo
# (RoceDataChannelInterfaceV1, FrameMetadataInterfaceV1, EnumerationMetadata,
# etc.) is visible to the sibling pybind11 extensions
# (_hololink_module_operators, _hololink_module_<module>). Without
# this, each .so resolves typeid against its own local copy and pybind11
# cross-module casts fail with "Unable to cast ... to C++ type '?'".
# The legacy hololink_core uses the same pattern
# (python/hololink/hololink_core/__init__.py).
_have_dlopen_flags = hasattr(sys, "getdlopenflags")
if _have_dlopen_flags:
    _old_dlopen_flags = sys.getdlopenflags()
    sys.setdlopenflags(os.RTLD_LAZY | os.RTLD_GLOBAL)

from . import _hololink_py_module as _native  # noqa: E402

if _have_dlopen_flags:
    sys.setdlopenflags(_old_dlopen_flags)

Adapter = _native.Adapter
Module = _native.Module
EnumerationMetadata = _native.EnumerationMetadata
FrameMetadata = _native.FrameMetadata
LogLevel = _native.LogLevel
AlarmEntry = _native.AlarmEntry

# Version-pinned interface names. Applications name the interface
# revision they were built against explicitly; there is no unversioned
# alias that tracks the latest version.
FrameMetadataInterfaceV1 = _native.FrameMetadataInterfaceV1
I2cLockV1 = _native.I2cLockV1
I2cInterfaceV1 = _native.I2cInterfaceV1
SequencerInterfaceV1 = _native.SequencerInterfaceV1
OscillatorInterfaceV1 = _native.OscillatorInterfaceV1
DataChannelInterfaceV1 = _native.DataChannelInterfaceV1
RoceDataChannelInterfaceV1 = _native.RoceDataChannelInterfaceV1
RoceReceiverInterfaceV1 = _native.RoceReceiverInterfaceV1
VsyncInterfaceV1 = _native.VsyncInterfaceV1
PtpPpsOutputInterfaceV1 = _native.PtpPpsOutputInterfaceV1
HololinkInterfaceV1 = _native.HololinkInterfaceV1
EnumerationInterfaceV1 = _native.EnumerationInterfaceV1
LoggingInterfaceV1 = _native.LoggingInterfaceV1
ReactorV1 = _native.ReactorV1
# Abstract CSI converter the module sensor drivers consume. The native
# implementation is hololink_module.operators.CsiToBayerOp (which
# subclasses this); exported so Python code can also implement the
# converter contract.
CsiConverterV1 = _native.CsiConverterV1

# Convenience logging functions (module equivalents of the legacy
# hololink.hsb_log_* helpers); route through the registered HSB logger and
# no-op when none is set.
hsb_log_trace = _native.hsb_log_trace
hsb_log_debug = _native.hsb_log_debug
hsb_log_info = _native.hsb_log_info
hsb_log_warn = _native.hsb_log_warn
hsb_log_error = _native.hsb_log_error

# Adapter-owned networking constant (mirror of the legacy hololink::core
# value), so callers name no legacy hololink constant.
DEFAULT_MTU = _native.DEFAULT_MTU

# Status codes (hololink/module/status.h) returned by V1 interface methods, so
# Python overrides return e.g. HOLOLINK_MODULE_OK rather than a bare integer.
HOLOLINK_MODULE_OK = _native.HOLOLINK_MODULE_OK
HOLOLINK_MODULE_INVALID_PARAMETER = _native.HOLOLINK_MODULE_INVALID_PARAMETER
HOLOLINK_MODULE_NOT_FOUND = _native.HOLOLINK_MODULE_NOT_FOUND
HOLOLINK_MODULE_NETWORK_ERROR = _native.HOLOLINK_MODULE_NETWORK_ERROR
HOLOLINK_MODULE_TIMEOUT = _native.HOLOLINK_MODULE_TIMEOUT
HOLOLINK_MODULE_ABI_MISMATCH = _native.HOLOLINK_MODULE_ABI_MISMATCH
HOLOLINK_MODULE_INIT_FAILED = _native.HOLOLINK_MODULE_INIT_FAILED
HOLOLINK_MODULE_ENUMERATION_SKIPPED = _native.HOLOLINK_MODULE_ENUMERATION_SKIPPED

__all__ = [
    "Adapter",
    "Module",
    "EnumerationMetadata",
    "FrameMetadata",
    "FrameMetadataInterfaceV1",
    "I2cLockV1",
    "I2cInterfaceV1",
    "SequencerInterfaceV1",
    "OscillatorInterfaceV1",
    "DataChannelInterfaceV1",
    "RoceDataChannelInterfaceV1",
    "RoceReceiverInterfaceV1",
    "VsyncInterfaceV1",
    "PtpPpsOutputInterfaceV1",
    "HololinkInterfaceV1",
    "EnumerationInterfaceV1",
    "LoggingInterfaceV1",
    "LogLevel",
    "ReactorV1",
    "CsiConverterV1",
    "AlarmEntry",
    "DEFAULT_MTU",
    "HOLOLINK_MODULE_OK",
    "HOLOLINK_MODULE_INVALID_PARAMETER",
    "HOLOLINK_MODULE_NOT_FOUND",
    "HOLOLINK_MODULE_NETWORK_ERROR",
    "HOLOLINK_MODULE_TIMEOUT",
    "HOLOLINK_MODULE_ABI_MISMATCH",
    "HOLOLINK_MODULE_INIT_FAILED",
    "HOLOLINK_MODULE_ENUMERATION_SKIPPED",
    "hsb_log_trace",
    "hsb_log_debug",
    "hsb_log_info",
    "hsb_log_warn",
    "hsb_log_error",
]

# Present only in RoCE-enabled builds: the ibv_device_for_peer and
# infiniband_devices bindings link ibverbs and are gated on
# HOLOLINK_BUILD_ROCE, so a build without RoCE omits them.
if hasattr(_native, "ibv_device_for_peer"):
    ibv_device_for_peer = _native.ibv_device_for_peer
    __all__.append("ibv_device_for_peer")
if hasattr(_native, "infiniband_devices"):
    infiniband_devices = _native.infiniband_devices
    __all__.append("infiniband_devices")
