# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CSI pixel- and Bayer-format enums for module-side sensor drivers.

These mirror the values the existing ``hololink.sensors.csi`` module
exposes so module-side sensor drivers (``hololink_module.sensors.*``)
can describe their output format without depending on the legacy
``hololink`` package.
"""

from enum import Enum


class PixelFormat(Enum):
    """CSI pixel format the sensor emits on the bus.

    Values match the C++ ``hololink::module::sensors::imx274::PixelFormat``
    enum (and the legacy ``hololink::csi::PixelFormat``), so call sites
    can pass ``.value`` straight into legacy operators
    (``ImageProcessorOp``, ``BayerDemosaicOp``).
    """

    RAW_8 = 0
    RAW_10 = 1
    RAW_12 = 2


class BayerFormat(Enum):
    """Bayer color-filter array layout.

    Values match NPP's ``NppiBayerGridPosition`` (and the legacy
    ``hololink::csi::BayerFormat``), so ``.value`` is the int the
    Holoscan demosaic / hololink image-processor operators expect.
    """

    BGGR = 0
    RGGB = 1
    GBRG = 2
    GRBG = 3
