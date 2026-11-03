# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registered hololink Operator proxies.

The receivers (LinuxReceiverOp, LinuxCoeReceiverOp, RoceReceiverOp,
GpuRoceTransceiverOp) and FusaCoeCaptureOp take their live objects as
constructor handles: pass the `hololink.DataChannel` itself as `data_channel=`,
and the sensor's start/stop methods (any callables) as `device_start=` and
`device_stop=`, e.g.::

    graph.op(hololink.operators.RoceReceiverOp, "receiver",
             data_channel=channel, device_start=camera.start,
             device_stop=camera.stop, frame_context=int(cu_context),
             frame_size=frame_size)

The receivers also need `frame_context`, the integer value of the CUDA context
their receive thread makes current.
"""

from holoscan.registry import operators as _operators

_registry_prefix = "hololink.operators."
_operator_types = {
    operator.name.removeprefix(_registry_prefix): operator
    for operator in _operators()
    if operator.name.startswith(_registry_prefix)
}
globals().update(_operator_types)
__all__: list[str] = []
__all__.extend(sorted(_operator_types))

# The raw receivers, for hosts that receive outside an operator graph (the emulator tests
# drive the emulator this way). Bound in _hololink; the registry above only lists Operators.
from .._hololink import (  # noqa: E402
    LinuxCoeReceiver,
    LinuxCoeReceiverMetadata,
    LinuxReceiver,
    LinuxReceiverMetadata,
)

__all__ += [
    "LinuxCoeReceiver",
    "LinuxCoeReceiverMetadata",
    "LinuxReceiver",
    "LinuxReceiverMetadata",
]
