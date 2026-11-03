# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registered hololink_module Operator proxies.

CsiToBayerOp / ImageProcessorOp / PackedFormatConverterOp's ports carry
hololink::CaptureMetadataT, the same generated payload type the native
hololink.operators family uses -- so, like that package's own __init__.py,
capture_metadata must be imported for its side effect (registering the
CaptureMetadataT payload identity) before holoscan.registry.operators()
is called below, or these operators are silently dropped from discovery.
See python/hololink/__init__.py's own comment for the (confirmed, not
assumed) mechanism.

LinuxReceiverOp / RoceReceiverOp / FusaCoeCaptureOp / HsbControllerOp are
manual-row operators (see module_operator_catalog.cpp) that take their live
objects as constructor handles: the hololink_module.EnumerationMetadata itself
as `metadata=`, the camera's start/stop methods (any callables) as
`device_start=`/`device_stop=`, and HsbControllerOp's `sensor_factory=` and
`network_receiver_factory=`. The receivers and HsbControllerOp also need
`frame_context`, the integer value of the CUDA context their receive thread
makes current. All four resolve here automatically via the registry discovery
below -- no separate binding needed per operator.
"""

import capture_metadata  # noqa: F401
from holoscan.registry import operators as _operators

_registry_prefix = "hololink_module.operators."
_operator_types = {
    operator.name.removeprefix(_registry_prefix): operator
    for operator in _operators()
    if operator.name.startswith(_registry_prefix)
}
globals().update(_operator_types)
__all__: list[str] = []
__all__.extend(sorted(_operator_types))
