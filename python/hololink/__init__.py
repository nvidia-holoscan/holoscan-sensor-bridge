# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""hololink -- HSDK5 Python bindings for the Holoscan Sensor Bridge.

Importing this module registers every hololink operator with HSDK5's
authoring registry (see operator_catalog.cpp); ``hololink.operators``
exposes them for ``graph.op(...)``.
"""

# capture_metadata is a sibling top-level module generated from capture_metadata.fbs (see
# src/hololink/operators/CMakeLists.txt's capture_metadata_schema_generated target's PYTHON
# flag), placed next to this package rather than inside it. Importing it here -- for its
# side effect, not its contents -- registers the "hololink.CaptureMetadataT" payload identity
# that every operator below with a CaptureMetadataT-typed port (ImageProcessorOp, CsiToBayerOp,
# etc.) needs for the `from . import operators` discovery pass below to find them at all;
# without this, every such operator is silently dropped from `hololink.operators` (confirmed by
# removing this import and hitting exactly that, not assumed). operator_catalog.cpp used to
# register this identity manually instead -- removed once this module's own auto-registration
# made that a hard AUTHORING_REGISTRY_DUPLICATE failure (see that file's own comment).
import capture_metadata  # noqa: F401

from . import _hololink as _native
from ._hololink import (
    BL_I2C_BUS,
    CAM_I2C_BUS,
    CLNX_SPI_BUS,
    CPNX_SPI_BUS,
    CTRL_EVENT,
    FPGA_DATE,
    GPIO,
    HSB_IP_VERSION,
    HSB_LOG_LEVEL_DEBUG,
    HSB_LOG_LEVEL_ERROR,
    HSB_LOG_LEVEL_INFO,
    HSB_LOG_LEVEL_INVALID,
    HSB_LOG_LEVEL_TRACE,
    HSB_LOG_LEVEL_WARN,
    I2C_BUSY,
    I2C_CTRL,
    I2C_DONE,
    I2C_FSM_ERR,
    I2C_I2C_ERR,
    I2C_I2C_NAK,
    I2C_REG_BUS_EN,
    I2C_REG_CLK_CNT,
    I2C_REG_CONTROL,
    I2C_REG_DATA_BUFFER,
    I2C_REG_NUM_BYTES,
    I2C_REG_STATUS,
    I2C_START,
    METADATA_SIZE,
    PAGE_SIZE,
    RD_DWORD,
    REQUEST_FLAGS_ACK_REQUEST,
    RESET_SPI_BUS,
    SPI_CTRL,
    AD9986Config,
    AsyncEventListener,
    BasicEnumerationStrategy,
    BayerFormat,
    CrcChannel,
    DataChannel,
    DataChannels,
    Deserializer,
    EnumerationStrategy,
    Enumerator,
    HololinkDevice,
    I2c,
    ImGuiRenderer,
    Metadata,
    NamedLock,
    NativeImx274Sensor,
    NativeVb1940Sensor,
    PacketizerProgram,
    PixelFormat,
    ReceiverMemoryDescriptor,
    Sequencer,
    Serializer,
    Spi,
    Synchronizable,
    Synchronizer,
    Timeout,
    Uart,
    find_channel,
    get_hsb_log_level,
    get_packetizer_program,
    hsb_log_debug,
    hsb_log_error,
    hsb_log_info,
    hsb_log_trace,
    hsb_log_warn,
    infiniband_device_for_peer,
    infiniband_devices,
    log_timestamp_s,
    resolve_generic_geometry,
    resolve_generic_geometry_from_dimensions,
    resolve_packed_geometry,
    round_up,
    set_hsb_log_level,
    set_uuid_strategy,
    use_mtu,
    use_sensor,
    use_vlan,
)

# Must follow the _hololink import above: it reads HSDK5's authoring registry at import
# time, which _hololink's native module load is what populates.
from . import operators  # isort:skip # noqa: E402

# hololink.sensors is a lazy shim (its __getattr__ imports each sensor package on first
# use), so importing it here costs nothing and keeps `hololink.sensors.<pkg>` reachable as an
# attribute of the top-level package, which the tests and the HSDK4-era drivers rely on.
from . import sensors  # isort:skip # noqa: E402

__all__ = [
    "AD9986Config",
    "AsyncEventListener",
    "BL_I2C_BUS",
    "BasicEnumerationStrategy",
    "BayerFormat",
    "CAM_I2C_BUS",
    "CLNX_SPI_BUS",
    "CPNX_SPI_BUS",
    "CTRL_EVENT",
    "CrcChannel",
    "DataChannel",
    "DataChannels",
    "Deserializer",
    "EnumerationStrategy",
    "Enumerator",
    "FPGA_DATE",
    "GPIO",
    "HSB_IP_VERSION",
    "HSB_LOG_LEVEL_DEBUG",
    "HSB_LOG_LEVEL_ERROR",
    "HSB_LOG_LEVEL_INFO",
    "HSB_LOG_LEVEL_INVALID",
    "HSB_LOG_LEVEL_TRACE",
    "HSB_LOG_LEVEL_WARN",
    "HololinkDevice",
    "I2C_BUSY",
    "I2C_CTRL",
    "I2C_DONE",
    "I2C_FSM_ERR",
    "I2C_I2C_ERR",
    "I2C_I2C_NAK",
    "I2C_REG_BUS_EN",
    "I2C_REG_CLK_CNT",
    "I2C_REG_CONTROL",
    "I2C_REG_DATA_BUFFER",
    "I2C_REG_NUM_BYTES",
    "I2C_REG_STATUS",
    "I2C_START",
    "I2c",
    "ImGuiRenderer",
    "METADATA_SIZE",
    "Metadata",
    "NamedLock",
    "NativeImx274Sensor",
    "NativeVb1940Sensor",
    "PAGE_SIZE",
    "PacketizerProgram",
    "PixelFormat",
    "ReceiverMemoryDescriptor",
    "RD_DWORD",
    "REQUEST_FLAGS_ACK_REQUEST",
    "RESET_SPI_BUS",
    "SPI_CTRL",
    "Sequencer",
    "Serializer",
    "Spi",
    "Synchronizable",
    "Synchronizer",
    "Timeout",
    "Uart",
    "find_channel",
    "get_hsb_log_level",
    "get_packetizer_program",
    "hsb_log_debug",
    "hsb_log_error",
    "hsb_log_info",
    "hsb_log_trace",
    "hsb_log_warn",
    "infiniband_device_for_peer",
    "infiniband_devices",
    "log_timestamp_s",
    "operators",
    "sensors",
    "resolve_generic_geometry",
    "resolve_generic_geometry_from_dimensions",
    "resolve_packed_geometry",
    "round_up",
    "set_hsb_log_level",
    "set_uuid_strategy",
    "use_mtu",
    "use_sensor",
    "use_vlan",
]

# The FuSa geometry helper is available only when the capture components were built.
if hasattr(_native, "resolve_geometry"):
    resolve_geometry = _native.resolve_geometry
    __all__.append("resolve_geometry")
