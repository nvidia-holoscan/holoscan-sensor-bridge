# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""I2C expander helper for HSB-Lite carriers.

The HSB-Lite I2C bus carries a peripheral that multiplexes the I2C
master across up to four downstream output channels. To talk to a
specific camera, the driver writes a one-byte mask to the expander's
I2C address (default ``0b01110000``) selecting which output is
enabled, then issues the per-camera I2C transactions on the same bus.

This module-side implementation takes the V1 ``I2cInterfaceV1``
handle directly — there's no implicit dependency on the legacy
``hololink`` package or the existing ``Serializer`` / ``Deserializer``
helpers.
"""

from enum import Enum


class I2C_Expander_Output_EN(Enum):
    """Output-enable masks for the LII2CExpander."""

    OUTPUT_1 = 0b0001  # for first camera
    OUTPUT_2 = 0b0010  # for second camera
    OUTPUT_3 = 0b0100
    OUTPUT_4 = 0b1000
    default = 0b0000


class LII2CExpander:
    """I2C expander that gates the bus toward one of four outputs.

    Constructed with a module ``I2cInterfaceV1`` handle pointing at
    the same bus the cameras live on; ``configure(output_en)`` selects
    which output the bus drives. ``synchronized_configure`` records
    the same selection into a ``SequencerInterfaceV1`` so it executes
    atomically with the rest of a hardware-triggered sequence.
    """

    I2C_EXPANDER_ADDRESS = 0b01110000

    def __init__(self, i2c):
        # i2c is a hololink_module.I2cInterfaceV1 reachable via
        # HololinkInterfaceV1.get_i2c(bus=..., address=I2C_EXPANDER_ADDRESS).
        self._i2c = i2c

    def configure(self, output_en=I2C_Expander_Output_EN.default.value):
        """Synchronously select which expander output is enabled."""
        write_bytes = bytes([output_en & 0xFF])
        read_byte_count = 0
        self._i2c.i2c_transaction(
            self.I2C_EXPANDER_ADDRESS,
            write_bytes,
            read_byte_count,
        )

    def synchronized_configure(self, sequencer, output_en):
        """Encode the selection into a sequencer for deferred execution."""
        write_bytes = bytes([output_en & 0xFF])
        read_byte_count = 0
        self._i2c.encode_i2c_request(
            sequencer,
            self.I2C_EXPANDER_ADDRESS,
            write_bytes,
            read_byte_count,
        )
