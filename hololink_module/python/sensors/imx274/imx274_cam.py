# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IMX274 sensor driver written against module V1 handles.

Mirrors the legacy ``hololink.sensors.imx274.dual_imx274.Imx274Cam``
register-table flow but takes ``HololinkInterfaceV1`` /
``I2cInterfaceV1`` / ``I2cLockV1`` module handles directly instead
of the legacy ``DataChannel`` / ``Hololink`` types. Two cameras
share the I2C bus on an HSB-Lite carrier; the per-camera
``LII2CExpander`` selects which one the bus drives before each
transaction. Construct with ``expander_configuration=0`` for the
first camera and ``expander_configuration=1`` for the second.
"""

import logging
import struct
import time
from collections import OrderedDict

from hololink_module.sensors import csi

import hololink_module

from . import imx274_mode, li_i2c_expander

# Camera info
DRIVER_NAME = "IMX274-DUAL"
VERSION = 1

# 7-bit camera I2C address (sensor strapped at 0b00011010).
CAM_I2C_ADDRESS = 0b00011010

# Default I2C bus index. HSB-Lite's camera I2C bus is index 1; callers
# that wire to a different bus pass ``i2c_bus=N`` explicitly.
DEFAULT_CAM_I2C_BUS = 1


class Imx274Cam:
    """IMX274 camera driver bound to a module ``HololinkInterfaceV1``.

    Methods mirror the legacy driver so existing call sites that take
    an ``Imx274Cam`` keep working once they switch to the module
    handles. The constructor takes the module ``HololinkInterfaceV1``
    directly — no ``DataChannel`` indirection — and uses
    ``hololink.get_i2c(bus=..., address=...)`` to reach the camera +
    I2C expander.
    """

    def __init__(
        self,
        hololink_or_metadata,
        oscillator=None,
        i2c_bus=DEFAULT_CAM_I2C_BUS,
        expander_configuration=0,
    ):
        # Two construction forms (matching the C++ overloads):
        #   Imx274Cam(hololink, oscillator, i2c_bus=..., expander_configuration=...)
        #   Imx274Cam(metadata, i2c_bus=...)
        # When called with `metadata`, HololinkInterface +
        # OscillatorInterface are resolved through the process-wide
        # Adapter and expander_configuration is derived from
        # metadata (prefers an explicit "expander_configuration"
        # entry, else uses "data_plane").
        if isinstance(hololink_or_metadata, hololink_module.EnumerationMetadata):
            metadata = hololink_or_metadata
            hololink = hololink_module.HololinkInterfaceV1.get_service(metadata)
            oscillator = hololink_module.OscillatorInterfaceV1.get_service(metadata)
            if "expander_configuration" in metadata:
                expander_configuration = metadata["expander_configuration"]
            else:
                if "data_plane" not in metadata:
                    raise RuntimeError(
                        "While constructing Imx274Cam from enumeration metadata: "
                        "neither 'expander_configuration' nor 'data_plane' is set"
                    )
                expander_configuration = metadata["data_plane"]
        else:
            hololink = hololink_or_metadata
            if oscillator is None:
                raise TypeError(
                    "Imx274Cam(hololink, oscillator, ...) requires an oscillator"
                )

        self._hololink = hololink
        self._oscillator = oscillator
        self._i2c_bus = i2c_bus
        self._i2c = hololink.get_i2c(bus=i2c_bus, address=CAM_I2C_ADDRESS)
        self._mode = imx274_mode.Imx274_Mode.Unknown
        self._instance = expander_configuration

        # The expander sits on the same bus but at a different
        # peripheral address; fetch a separate I2cInterface for it.
        expander_i2c = hololink.get_i2c(
            bus=i2c_bus, address=li_i2c_expander.LII2CExpander.I2C_EXPANDER_ADDRESS
        )
        self._i2c_expander = li_i2c_expander.LII2CExpander(expander_i2c)
        if expander_configuration == 1:
            self._i2c_expander_configuration = (
                li_i2c_expander.I2C_Expander_Output_EN.OUTPUT_2
            )
        else:
            self._i2c_expander_configuration = (
                li_i2c_expander.I2C_Expander_Output_EN.OUTPUT_1
            )

    @staticmethod
    def use_expander_configuration(metadata, expander_configuration):
        """Stamp an explicit ``expander_configuration`` into ``metadata``.

        The metadata-based constructor picks this up instead of falling
        back to ``data_plane``. Application code overrides the LI I2C
        expander output through this method rather than mutating
        ``EnumerationMetadata`` fields by string key — that keeps the
        metadata layout an ``Imx274Cam`` concern, not an application
        concern.
        """
        metadata["expander_configuration"] = expander_configuration

    # ------------------------------------------------------------------
    # Clock + lifecycle
    # ------------------------------------------------------------------

    def setup_clock(self, clock_profile=None):
        """Program the on-board Renesas Bajoran Lite TS1 clock generator.

        The HSB-Lite supplement (when wired in) provides the I2C-write
        sequence the existing tree's ``renesas_bajoran_lite_ts1`` helper
        produced. Callers supply the same kind of profile.
        """
        if clock_profile is None:
            raise ValueError(
                "While calling Imx274Cam.setup_clock: a clock_profile is required "
                "(typically obtained from the HSB-Lite supplement)"
            )
        self._hololink.setup_clock(clock_profile)

    def configure(self, imx274_mode_set):
        version = self.get_version()
        logging.info("version=%s" % (version,))
        assert version == VERSION
        # IMX274 needs a 25 MHz reference clock on the HSB-Lite
        # carrier. The per-data-plane oscillator the supplement
        # publishes either programs that rate (returns True) or
        # rejects the request (returns False) — bail out before
        # touching the sensor if the oscillator can't deliver.
        if not self._oscillator.enable(25_000_000):
            raise RuntimeError(
                "While configuring Imx274Cam: oscillator does not support the "
                "IMX274's 25 MHz reference clock"
            )
        self.configure_camera(imx274_mode_set)

    def start(self):
        """Start streaming."""
        self._running = True
        for reg, val in imx274_mode.imx274_start:
            if reg == imx274_mode.IMX274_TABLE_WAIT_MS:
                time.sleep(val / 1000)
            else:
                self.set_register(reg, val)

    def stop(self):
        """Stop streaming."""
        for reg, val in imx274_mode.imx274_stop:
            if reg == imx274_mode.IMX274_TABLE_WAIT_MS:
                time.sleep(val / 1000)
            else:
                self.set_register(reg, val)
        time.sleep(0.1)  # let the egress buffer drain
        self._running = False

    def get_version(self):
        # The IMX274 doesn't expose a queryable version register;
        # callers rely on the driver's own VERSION constant.
        return VERSION

    # ------------------------------------------------------------------
    # Per-register I/O. Each transaction acquires the per-board I2C
    # lock so two cameras sharing the bus don't tangle their packets.
    # ------------------------------------------------------------------

    def get_register(self, register):
        logging.debug("get_register(register=%d(0x%X))" % (register, register))
        write_bytes = struct.pack(">H", register)  # uint16 BE
        read_byte_count = 1
        i2c_lock = self._hololink.i2c_lock()
        i2c_lock.lock()
        try:
            self._i2c_expander.configure(self._i2c_expander_configuration.value)
            reply = self._i2c.i2c_transaction(
                CAM_I2C_ADDRESS, write_bytes, read_byte_count
            )
        finally:
            i2c_lock.unlock()
        r = reply[0]
        logging.debug(
            "get_register(register=%d(0x%X))=%d(0x%X)" % (register, register, r, r)
        )
        return r

    def set_register(self, register, value):
        logging.debug(
            "set_register(register=%d(0x%X), value=%d(0x%X))"
            % (register, register, value, value)
        )
        write_bytes = struct.pack(">H", register) + struct.pack(">B", value & 0xFF)
        read_byte_count = 0
        i2c_lock = self._hololink.i2c_lock()
        i2c_lock.lock()
        try:
            self._i2c_expander.configure(self._i2c_expander_configuration.value)
            self._i2c.i2c_transaction(
                CAM_I2C_ADDRESS,
                write_bytes,
                read_byte_count,
            )
        finally:
            i2c_lock.unlock()

    # ------------------------------------------------------------------
    # Mode + format helpers
    # ------------------------------------------------------------------

    def configure_camera(self, imx274_mode_set):
        self.set_mode(imx274_mode_set)

        mode_list = OrderedDict()
        if (
            imx274_mode_set.value
            == imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS.value
        ):
            mode_list = imx274_mode.imx274_mode_3840X2160_60fps
        elif (
            imx274_mode_set.value
            == imx274_mode.Imx274_Mode.IMX274_MODE_1920X1080_60FPS.value
        ):
            mode_list = imx274_mode.imx274_mode_1920x1080_60fps
        elif (
            imx274_mode_set.value
            == imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS_12BITS.value
        ):
            mode_list = imx274_mode.imx274_mode_3840X2160_60fps_12bits
        else:
            logging.error(f"{imx274_mode_set} mode is not present.")

        for reg, val in mode_list:
            if reg == imx274_mode.IMX274_TABLE_WAIT_MS:
                time.sleep(val / 1000)
            else:
                self.set_register(reg, val)

    def set_exposure_reg(self, value=0x0C):
        # IMX274 minimum exposure is 12 (0x0C); clamp to [0x0C, 0xFFFF].
        if value < 0x0C:
            logging.warning(f"Exposure value {value} is lower than the minimum.")
            value = 0x0C
        if value > 0xFFFF:
            logging.warning(f"Exposure value {value} is higher than the maximum.")
            value = 0xFFFF
        self.set_register(imx274_mode.REG_EXP_LSB, (value >> 8) & 0xFF)
        self.set_register(imx274_mode.REG_EXP_MSB, value & 0xFF)
        time.sleep(imx274_mode.IMX274_WAIT_MS / 1000)

    def set_digital_gain_reg(self, value=0x0000):
        # IMX274's DG register only accepts gain stops 0..6 (1..64x).
        reg_value = 0x00
        if value >= 0x40:
            reg_value = 0x06
        elif value >= 0x20:
            reg_value = 0x05
        elif value >= 0x10:
            reg_value = 0x04
        elif value >= 0x08:
            reg_value = 0x03
        elif value >= 0x04:
            reg_value = 0x02
        elif value >= 0x02:
            reg_value = 0x01
        self.set_register(imx274_mode.REG_DG, reg_value)
        time.sleep(imx274_mode.IMX274_WAIT_MS / 1000)

    def set_analog_gain_reg(self, value=0x0C):
        if value < 0x00:
            logging.warning(f"AG value {value} is lower than the minimum.")
            value = 0x00
        if value > 0xFFFF:
            logging.warning(f"AG value {value} is higher than the maximum.")
            value = 0xFFFF
        self.set_register(imx274_mode.REG_AG_LSB, (value >> 8) & 0xFF)
        self.set_register(imx274_mode.REG_AG_MSB, value & 0xFF)
        time.sleep(imx274_mode.IMX274_WAIT_MS / 1000)

    def set_mode(self, imx274_mode_set):
        if imx274_mode_set.value < len(imx274_mode.Imx274_Mode):
            self._mode = imx274_mode_set
            mode = imx274_mode.imx_frame_format[self._mode.value]
            self._height = mode.height
            self._width = mode.width
            self._pixel_format = mode.pixel_format
        else:
            logging.error("Incorrect mode for IMX274")
            self._mode = -1

    # ------------------------------------------------------------------
    # Format accessors used by the receiver / converter operators
    # ------------------------------------------------------------------

    def pixel_format(self):
        return self._pixel_format

    def bayer_format(self):
        return csi.BayerFormat.RGGB

    def configure_converter(self, converter):
        # Trains the native module CsiToBayerOp to match this driver's
        # geometry. The converter's pixel-format argument is the module
        # csi.PixelFormat enumerator value (the binding coerces the int to
        # the C++ hololink::module::csi::PixelFormat), so we pass
        # self._pixel_format.value at the boundary.
        if self._pixel_format not in (
            csi.PixelFormat.RAW_8,
            csi.PixelFormat.RAW_10,
            csi.PixelFormat.RAW_12,
        ):
            raise ValueError(
                f"While configuring the converter: pixel format {self._pixel_format} "
                "is not supported by IMX274"
            )
        pixel_format = self._pixel_format.value

        start_byte = converter.receiver_start_byte()
        transmitted_line_bytes = converter.transmitted_line_bytes(
            pixel_format, self._width
        )
        received_line_bytes = converter.received_line_bytes(transmitted_line_bytes)
        # 175 bytes of metadata precede the image data.
        start_byte += converter.received_line_bytes(175)
        if self._pixel_format == csi.PixelFormat.RAW_10:
            # 8 lines of optical black before real image data
            start_byte += received_line_bytes * 8
        elif self._pixel_format == csi.PixelFormat.RAW_12:
            # 16 lines of optical black before real image data
            start_byte += received_line_bytes * 16
        converter.configure(
            start_byte,
            received_line_bytes,
            self._width,
            self._height,
            pixel_format,
        )

    # ------------------------------------------------------------------
    # Test pattern helpers
    # ------------------------------------------------------------------

    def test_pattern(self, pattern=None):
        """Enable / disable the IMX274's built-in test pattern."""
        if pattern is None:
            self.set_register(0x303C, 0)
            self.set_register(0x377F, 0)
            self.set_register(0x3781, 0)
            self.set_register(0x370B, 0)
        else:
            self.set_register(0x303C, 0x11)
            self.set_register(0x370E, 0x01)
            self.set_register(0x377F, 0x01)
            self.set_register(0x3781, 0x01)
            self.set_register(0x370B, 0x11)
            self.set_register(0x303D, pattern)

    def test_pattern_update(self, pattern):
        self.set_register(0x303D, pattern)

    def synchronized_test_pattern_update(self, sequencer, pattern):
        """Schedule a test-pattern register write into the sequencer."""
        self.synchronized_set_register(sequencer, 0x303D, pattern)
        sequencer.enable()

    def synchronized_set_register(self, sequencer, register, value):
        """Encode a register write into the sequencer (no immediate I/O)."""
        self._i2c_expander.synchronized_configure(
            sequencer, self._i2c_expander_configuration.value
        )
        write_bytes = struct.pack(">H", register) + struct.pack(">B", value & 0xFF)
        self._i2c.encode_i2c_request(
            sequencer,
            CAM_I2C_ADDRESS,
            write_bytes,
            read_byte_count=0,
        )
