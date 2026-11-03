# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# OmniVision OX08D10 register tables for Aumovio SAT301DP17 camera module.
# Sensor: 3840x2160 HDR, RAW12, RGGB, 4-lane MIPI CSI-2, GMSL2 interface.
#
# ---------------------------------------------------------------------------
# GMSL SKILL WALL DOCUMENTATION
# Everything the add-sensor skill could NOT derive from the datasheet alone:
#
# 1. PIXEL FORMAT: RAW_12 (not RAW_10). The sensor outputs RAW_12 when
#    configured for GMSL HDR streaming. Register defaults are RAW_8/RAW_10 but
#    MIPI output groups must be forced to RAW_12 via 0x431C/0x431D/0x431F.
#    Cannot be inferred from the datasheet bit-depth field.
#
# 2. RECEIVED HEIGHT: 2176 rows (2160 active + 16 trailing non-image rows from
#    serializer). Using 2160 causes buffer overflow and dropped frames.
#    The datasheet describes "active array" as 2160 only.
#
# 3. 45FPS MODE: Only 5 registers needed (force RAW12 output groups). Sensor
#    powers up at 45fps by default. No PLL reprogramming required. The skill
#    generated a large PLL table from datasheet defaults that was wrong.
#
# 4. 30FPS MODE: Requires vendor-validated PLL (0x4837, 0x04xx block), frame
#    timing (VTS/HTS), voltage monitor thresholds (0x4583-0x458E), and ASIL
#    safety config (0x1035, 0x4Fxx). None derivable from public DS 2.03 defaults.
#    ALSO requires 8 named sequences from proprietary OX08DHsl.py vendor file:
#    seq_system_default_reg_set, seq_config_3840x2160_raw12_30fps,
#    seq_disable_embedded_data, seq_config_pwl12_blc4, seq_set_hdr_combine_table,
#    seq_set_fsync_internal, seq_read_device_info, seq_safety_config_flip.
#    Contact OmniVision for OX08DHsl.py.
#
# 5. GMSL ADDRESSES: MAX96716A deserializer (0x28) and MAX96719B serializer (0x40)
#    addresses come from the SAT301DP17 interface spec, not the sensor datasheet.
#
# 6. GMSL BRING-UP: Deserializer 6Gbps mode, link lock polling, serializer MIPI
#    PHY config (D-PHY 4-lane), sensor GPIO reset via serializer GPIO3, and
#    virtual channel mapping (0x4813) -- all require MAX96716A/MAX96719B
#    datasheets and board schematic. Not in any sensor datasheet.
#
# 7. NO I2C EXPANDER: Unlike IMX274/IMX681, GMSL cameras do NOT use the
#    li_i2c_expander. I2C tunnels through the GMSL link via MAX96719B.
#
# 8. SENSOR IDENTITY: Registers 0x300A/0x300B/0x300C, expected (0x58, 0x08, 0x44).
#    Must be validated after GMSL link is live before streaming.
# ---------------------------------------------------------------------------

from collections import namedtuple
from enum import Enum

import hololink

# Sentinel for delay entries (milliseconds). Uses string to distinguish from
# sensor register addresses, consistent with other hololink sensor drivers.
OX08D10_TABLE_WAIT_MS = "ox08d10-table-wait-ms"
OX08D10_TABLE_WRITE_MASKED = "ox08d10-table-write-masked"
OX08D10_WAIT_MS = 1
OX08D10_RESET_WAIT_MS = 10

# ---------------------------------------------------------------------------
# GMSL SerDes I2C addresses (SAT301DP17 Interface Spec, section 4)
# NOT in the OX08D10 sensor datasheet.
# ---------------------------------------------------------------------------
MAX96719_SER_ADDR = 0x40  # MAX96719B serializer on SAT301DP17 module
MAX96716A_DESER_ADDR = 0x28  # MAX96716A deserializer on Taurotech HSB board

# Legacy aliases used by existing hololink camera plumbing
MAX96919_SER_ADDR = MAX96719_SER_ADDR
MAX9295D_SER_ADDR = MAX96719_SER_ADDR

OX08D10_SENSOR_ADDR = 0x36  # 7-bit I2C address (tunneled via GMSL)


class GmslLink(Enum):
    LINK_A = 0
    LINK_B = 1


GMSL_LINK = GmslLink.LINK_A


def set_gmsl_link(link):
    if not isinstance(link, GmslLink):
        raise TypeError(f"Expected GmslLink, got {type(link).__name__}")
    global GMSL_LINK
    GMSL_LINK = link


# ---------------------------------------------------------------------------
# Mode enum and frame format
# ---------------------------------------------------------------------------


class Ox08d10_Mode(Enum):
    OX08D10_MODE_3840X2160_45FPS_RAW12 = 0
    OX08D10_MODE_3840X2160_30FPS_RAW12 = 1
    Unknown = 2


frame_format = namedtuple(
    "FrameFormat", ["width", "height", "framerate", "pixel_format", "num_lanes"]
)

ox08d10_frame_format = []
ox08d10_frame_format.insert(
    Ox08d10_Mode.OX08D10_MODE_3840X2160_45FPS_RAW12.value,
    # SKILL WALL: height=2176 (2160 active + 16 trailing rows from serializer)
    # SKILL WALL: pixel_format=RAW_12 (forced by output group registers, not DS default)
    frame_format(3840, 2176, 45, hololink.sensors.csi.PixelFormat.RAW_12, 4),
)
ox08d10_frame_format.insert(
    Ox08d10_Mode.OX08D10_MODE_3840X2160_30FPS_RAW12.value,
    frame_format(3840, 2176, 30, hololink.sensors.csi.PixelFormat.RAW_12, 4),
)


# ---------------------------------------------------------------------------
# 45fps mode -- force RAW12 output groups only.
# SKILL WALL: Sensor powers up at 45fps by default; only 5 registers needed.
# The skill incorrectly generated PLL tables from datasheet defaults.
# Source: vendor reference driver.
# ---------------------------------------------------------------------------
ox08d10_mode_3840x2160_45fps_raw12 = [
    (0x4319, 0x33),  # HDR combine defaults; keep both bitwidth groups enabled
    (0x431A, 0x22),  # MIPI/VFIFO bitwidth group0/group1: 12-bit
    (0x431C, 0x6C),  # MIPI image data type group0: RAW12 (DT=0x2C)
    (0x431D, 0x6C),  # MIPI image data type group1: RAW12
    (0x431F, 0x20),  # PWL0 enable, 12-bit output
    (OX08D10_TABLE_WAIT_MS, OX08D10_WAIT_MS),
]


# ---------------------------------------------------------------------------
# 30fps mode -- vendor-validated PLL + timing + voltage monitors + RAW12.
# SKILL WALL: PLL values (0x4837, 0x04xx), frame timing (VTS/HTS), voltage
# monitor thresholds (0x4583-0x458E), and ASIL config (0x1035/0x4Fxx) cannot
# be derived from the public datasheet. Values from vendor reference driver.
#
# NOTE: The complete 30fps init also requires 8 named sequences from the
# proprietary OX08DHsl.py file (system defaults, PWL tone map, HDR combine
# table, fsync internal, safety config). Those sequences are NOT included here
# because the file is not publicly available. This table is the partial static
# equivalent that handles PLL/timing/format. Contact OmniVision for OX08DHsl.py.
# ---------------------------------------------------------------------------
ox08d10_mode_3840x2160_30fps_raw12 = [
    (0x0100, 0x00),  # MODE_SELECT: software standby before PLL/MIPI reconfiguration
    # PLL / clock chain for 1100 Mbps MIPI PHY
    (
        0x4837,
        0x0E,
    ),  # MIPI_CTRL37: PHY byte clock period (0x0E = 14 * 2ns = 28ns -> 1100 MHz DDR)
    (0x0301, 0xC8),  # PLL1_CTRL1: PLL1 pre-divider and charge pump
    (0x0303, 0x01),  # PLL1_CTRL3: PLL1 output divider select
    (0x0304, 0x01),  # PLL1_CTRL4: PLL1 DAC clock divider
    (0x0305, 0x08),  # PLL1_CTRL5: PLL1 multiplier high byte
    (0x0307, 0x01),  # PLL1_CTRL7: PLL1 system divider
    (0x0362, 0xA8),  # PLL2_FILTER: PLL2 loop filter / baseband resistance
    (0x0408, 0xE0),  # PLL2_CTRL8: PLL2 tuning byte 0
    (0x0409, 0x80),  # PLL2_CTRL9: PLL2 tuning byte 1
    (0x040A, 0xDE),  # PLL2_CTRL10: PLL2 tuning byte 2
    (0x040B, 0x34),  # PLL2_CTRL11: PLL2 tuning byte 3
    (0x040C, 0x51),  # PLL2_CTRL12: PLL2 tuning byte 4
    (0x040D, 0xA3),  # PLL2_CTRL13: PLL2 tuning byte 5
    (0x040E, 0x08),  # PLL2_CTRL14: PLL2 tuning byte 6
    (0x0324, 0x01),  # PLL clock monitor expected SCLK count high byte
    (0x0325, 0x88),  # PLL clock monitor expected SCLK count low byte
    (0x0400, 0xEF),  # PLL_MON: expected SCLK period high byte
    (0x0401, 0xFF),  # PLL_MON: expected SCLK period low byte
    (0x0403, 0x85),  # PLL_MON: CRC checksum SCLK high byte
    (0x0404, 0x5D),  # PLL_MON: CRC checksum SCLK low byte
    (0x0405, 0x3A),  # PLL_MON: pulse width SCLK high byte
    (0x0406, 0xC8),  # PLL_MON: pulse width SCLK low byte
    (0x0407, 0x0D),  # PLL_MON: SCLK tolerance
    # Frame timing: HTS_DCG=0x0608 (1544), HTS_VS=0x03A4 (932), VTS=0x055E (1374) -> ~30.01 fps
    (0x380C, 0x06),  # HTS_DCG_H: DCG/LOFIC horizontal timing high byte
    (0x380D, 0x08),  # HTS_DCG_L: DCG/LOFIC horizontal timing low byte
    (0x388C, 0x03),  # HTS_VS_H: VS horizontal timing high byte
    (0x388D, 0xA4),  # HTS_VS_L: VS horizontal timing low byte
    (0x380E, 0x05),  # VTS_H: total rows per frame high byte
    (0x380F, 0x5E),  # VTS_L: total rows per frame low byte
    (0x35C6, 0x60),  # timing calibration for 30fps PHY
    # Timing generator CRC reference values (ASIL integrity check on timing config)
    (0x374F, 0x40),  # TIMING_GEN_CRC_REF byte 0
    (0x3750, 0xC3),  # TIMING_GEN_CRC_REF byte 1
    (0x3751, 0x60),  # TIMING_GEN_CRC_REF byte 2
    (0x3752, 0x4E),  # TIMING_GEN_CRC_REF byte 3
    (0x3012, 0x41),  # SC_PHY_CTRL: 4-lane D-PHY MIPI mode
    (0x4800, 0x04),  # MIPI_CTRL00: MIPI core enable, clock lane in HS mode
    # Supply voltage monitor thresholds (3 rails, high/low bounds each)
    (0x4583, 0x08),  # VMON rail 0 upper threshold high byte
    (0x4584, 0x47),  # VMON rail 0 upper threshold low byte
    (0x4585, 0x09),  # VMON rail 0 lower threshold high byte
    (0x4586, 0x55),  # VMON rail 0 lower threshold low byte
    (0x4587, 0x04),  # VMON rail 1 upper threshold high byte
    (0x4588, 0x84),  # VMON rail 1 upper threshold low byte
    (0x4589, 0x05),  # VMON rail 1 lower threshold high byte
    (0x458A, 0x17),  # VMON rail 1 lower threshold low byte
    (0x458B, 0x02),  # VMON rail 2 upper threshold high byte
    (0x458C, 0xC2),  # VMON rail 2 upper threshold low byte
    (0x458D, 0x03),  # VMON rail 2 lower threshold high byte
    (0x458E, 0x1C),  # VMON rail 2 lower threshold low byte
    # ASIL safety config
    (0x1035, 0x7F),  # ASIL_CTRL: enable all safety mechanisms (bits 6:0)
    (0x4F00, 0x00),  # SM_CTRL0: safety mechanism control byte 0
    (0x4F01, 0x00),  # SM_CTRL1: safety mechanism control byte 1
    (0x4F02, 0x00),  # SM_CTRL2: safety mechanism control byte 2
    (0x4F03, 0x00),  # SM_CTRL3: safety mechanism control byte 3
    (0x4F04, 0x2C),  # SM_CTRL4: safety mechanism control byte 4 (ERRB pin config)
    # Force RAW12 output groups (same 5 registers as 45fps mode)
    (0x4319, 0x33),  # HDR combine defaults; keep both bitwidth groups enabled
    (0x431A, 0x22),  # MIPI/VFIFO bitwidth group0/group1: 12-bit
    (0x431C, 0x6C),  # MIPI image data type group0: RAW12 (DT=0x2C)
    (0x431D, 0x6C),  # MIPI image data type group1: RAW12
    (0x431F, 0x20),  # PWL0 enable, 12-bit output
    (OX08D10_TABLE_WAIT_MS, OX08D10_WAIT_MS),
]


ox08d10_start = [
    (0x0100, 0x01),  # MODE_SELECT: streaming on
    (OX08D10_TABLE_WAIT_MS, OX08D10_WAIT_MS),
]

ox08d10_stop = [
    (0x0100, 0x00),  # MODE_SELECT: streaming off (software standby)
    (OX08D10_TABLE_WAIT_MS, OX08D10_WAIT_MS),
]


# ---------------------------------------------------------------------------
# GMSL bring-up sequences
#
# SKILL WALL: The entire GMSL section below is NOT derivable from the OX08D10
# sensor datasheet. It requires:
#   - MAX96716A deserializer datasheet + register map
#   - MAX96719B serializer datasheet + register map
#   - SAT301DP17 camera module interface spec and board schematic
#   - Vendor reference driver (MAX967XX.cpp / MAX96719Hsl.py)
#
# Functions return (i2c_addr, reg, value) 3-tuples for multi-device setup,
# distinct from the (reg, value) 2-tuples used for sensor register writes.
# ---------------------------------------------------------------------------


def get_max96716_control_ox08d10():
    """Configure MAX96716A deserializer control path for the selected GMSL link.

    The MAX96719B serializer is 6 Gbps GMSL2 only. Link speed MUST be set to
    6 Gbps or GMSL link will never lock (LOCKED bit stays 0).

    SKILL WALL: register meanings from MAX96716A datasheet + vendor HSL reference.
    """
    selected_link = 0x01 if GMSL_LINK == GmslLink.LINK_A else 0x02
    if GMSL_LINK == GmslLink.LINK_A:
        ctrl0_spd = 0x32  # RESET_ONESHOT(bit5) + bits1:0=10 -> 6Gbps link A
    else:
        ctrl0_spd = 0x23  # RESET_ONESHOT(bit5) + bit4=0(6Gbps) + bits1:0=11 link B

    return [
        # CTRL0 (0x0010): set 6Gbps. Previous value 0x33 left bit4=1 (3Gbps)
        # which prevents MAX96719B (6Gbps-only) from ever locking.
        (MAX96716A_DESER_ADDR, 0x0010, ctrl0_spd),
        (MAX96716A_DESER_ADDR, 0x0F00, selected_link),
        (MAX96716A_DESER_ADDR, 0x0012, 0x20),
    ]


def get_max96719_serializer_ox08d10():
    """Configure MAX96719B serializer, enable RCLKOUT, release sensor reset.

    SKILL WALL: D-PHY 4-lane adaptation of vendor seq_set_ser_phy_max96719().
    Vendor default is C-PHY 2-lane (0xA005[6:5]=01); changed to D-PHY 4-lane
    (0xA005[6:5]=11). GPIO6 = 25MHz RCLKOUT to sensor; GPIO3 = sensor reset.
    Requires MAX96719B datasheet + SAT301DP17 module schematic.
    """
    return [
        (MAX96719_SER_ADDR, 0x0001, 0x00),  # CC pass-through disabled
        (MAX96719_SER_ADDR, 0x350F, 0x05),
        (MAX96719_SER_ADDR, 0x3511, 0x46),
        (MAX96719_SER_ADDR, 0x358F, 0x05),
        (MAX96719_SER_ADDR, 0x3591, 0x46),
        (MAX96719_SER_ADDR, 0x3013, 0x00),
        (MAX96719_SER_ADDR, 0x0101, 0x00),
        (MAX96719_SER_ADDR, 0xA005, 0x61),  # D-PHY 4-lane + MIPI_RX_RESET
        (MAX96719_SER_ADDR, 0x2003, 0x20),  # video disabled during reset
        # Enable RCLKOUT (25MHz to sensor) and release sensor reset GPIO.
        (MAX96719_SER_ADDR, 0x0014, 0x01),
        (MAX96719_SER_ADDR, 0x3049, 0x30),
        (OX08D10_TABLE_WAIT_MS, 50),
        (MAX96719_SER_ADDR, 0x3049, 0x38),
        (OX08D10_TABLE_WAIT_MS, 500),  # 500ms sensor boot time after reset
        # Packet-detect timer then release MIPI reset.
        (MAX96719_SER_ADDR, 0xA005, 0x61),
        (MAX96719_SER_ADDR, 0xA113, 0x02),  # 33ms packet-detect timeout
        (OX08D10_TABLE_WAIT_MS, 50),
        (MAX96719_SER_ADDR, 0xA005, 0x60),  # release MIPI reset, D-PHY 4-lane
        (MAX96719_SER_ADDR, 0x2003, 0x21),  # enable video port
        (MAX96719_SER_ADDR, 0x0020, 0x9F),  # ERRB TX config (SerFinalizeInit)
    ]


def get_max96716_video_ox08d10(raw_data_type=0x2C):
    """Configure MAX96716A deserializer video pipe for one RAW12 stream.

    raw_data_type=0x2C is the MIPI CSI-2 data type code for RAW12.
    SKILL WALL: Pipe routing and tunnel mode require MAX96716A datasheet.
    """
    selected_link = 0x01 if GMSL_LINK == GmslLink.LINK_A else 0x02
    ctrl0_6gbps = 0x32 if GMSL_LINK == GmslLink.LINK_A else 0x23
    return [
        (MAX96716A_DESER_ADDR, 0x0F00, selected_link),
        (MAX96716A_DESER_ADDR, 0x0010, ctrl0_6gbps),
        (MAX96716A_DESER_ADDR, 0x0330, 0x04),
        (MAX96716A_DESER_ADDR, 0x0333, 0x4E),
        (MAX96716A_DESER_ADDR, 0x0334, 0xE4),
        (MAX96716A_DESER_ADDR, 0x040A, 0x00),
        (MAX96716A_DESER_ADDR, 0x044A, 0xD0),
        (MAX96716A_DESER_ADDR, 0x048A, 0xD0),
        (MAX96716A_DESER_ADDR, 0x04CA, 0x00),
        # Tunnel output enabled: MAX96719B forwards CSI-2 packets including FS/FE.
        (MAX96716A_DESER_ADDR, 0x0474, 0x09),
        (MAX96716A_DESER_ADDR, 0x04B4, 0x09),
        (MAX96716A_DESER_ADDR, 0x031D, 0x00),
        (MAX96716A_DESER_ADDR, 0x0320, 0x2F),
        (MAX96716A_DESER_ADDR, 0x0323, 0x2F),
        (MAX96716A_DESER_ADDR, 0x0326, 0x00),
        (
            (MAX96716A_DESER_ADDR, 0x0161, 0x10)
            if GMSL_LINK == GmslLink.LINK_A
            else (MAX96716A_DESER_ADDR, 0x0161, 0x34)
        ),
        (MAX96716A_DESER_ADDR, 0x0332, 0xF4),
        (MAX96716A_DESER_ADDR, 0x044B, 0x03),
        (MAX96716A_DESER_ADDR, 0x044D, raw_data_type | (0 << 6)),
        (MAX96716A_DESER_ADDR, 0x044E, raw_data_type | (0 << 6)),
        (MAX96716A_DESER_ADDR, 0x044F, raw_data_type | (2 << 6)),
        (MAX96716A_DESER_ADDR, 0x0450, raw_data_type | (2 << 6)),
        (MAX96716A_DESER_ADDR, 0x046D, 0x09),
    ]


def get_gmsl_sequence_ox08d10(gmsl_setup="full", raw_data_type=0x2C):
    """Return the GMSL bring-up register sequence for OX08D10.

    gmsl_setup selects which components are configured:
      "none"         - skip all GMSL setup (sensor I2C must already be live)
      "control"      - deserializer control path only
      "remote"       - serializer only
      "serializer"   - control + serializer
      "deserializer" - control + deserializer video pipe
      "full"         - all three (normal startup sequence)
    """
    control = get_max96716_control_ox08d10()
    serializer = get_max96719_serializer_ox08d10()
    deserializer = get_max96716_video_ox08d10(raw_data_type=raw_data_type)

    if gmsl_setup == "none":
        return []
    if gmsl_setup == "control":
        return control
    if gmsl_setup == "remote":
        return serializer
    if gmsl_setup == "serializer":
        return control + serializer
    if gmsl_setup == "deserializer":
        return control + deserializer
    if gmsl_setup == "full":
        return control + serializer + deserializer
    raise ValueError(f"Unsupported OX08D10 GMSL setup: {gmsl_setup}")


# ---------------------------------------------------------------------------
# Sensor initialization sequences for HDR4 PWL12 3840x2160 30fps mode.
# ---------------------------------------------------------------------------

# Software reset and 20ms stabilization wait.
ox08d10_software_reset = [
    (0x0107, 0x01),  # SOFTWARE_RST: trigger reset (self-clearing)
    (OX08D10_TABLE_WAIT_MS, 20),
]

# Force RAW12 MIPI output groups.
# Sensor default is RAW8/RAW10; these registers force RAW12 output.
ox08d10_raw12_output_groups = [
    (
        0x4319,
        0x33,
    ),  # FORMAT_REG_19: force 12-bit MIPI bitwidth for both groups, enable HDR combine
    (0x431A, 0x22),  # FORMAT_REG_1A: select format group 2 for MIPI and VFIFO channels
    (
        0x431C,
        0x6C,
    ),  # FORMAT_REG_1C: force MIPI data type 0 = RAW12 (bit6=force, bits[5:0]=0x2C)
    (
        0x431D,
        0x6C,
    ),  # FORMAT_REG_1D: force MIPI data type 1 = RAW12 (bit6=force, bits[5:0]=0x2C)
    (
        0x431F,
        0x20,
    ),  # FORMAT_REG_1F: 24-bit packing option 1 for HDR output; PWL disabled
]

# Disable embedded data output.
ox08d10_disable_ebd = [
    (0x4317, 0x08),  # EBD_SEQ_CTRL: disable EBD lines from all channels
]

# HDR combine table (0x5B80-0x5BBD).
ox08d10_hdr_combine_table = [
    # HDR combine control and weight table (section 4.6).
    # Combines HCG, LCG, LOFIC, and VS exposures into a single 24-bit linear HDR frame.
    (
        0x5B80,
        0x00,
    ),  # COMBINE_0_ISP_CTRL_00: HDR combine enable; 0x00 = combine mode, ratio method
    (0x5B93, 0x80),  # COMBINE_hard_thre: hard stability threshold for ratio comparison
    (0x5B96, 0x80),  # cmb_weight0_0: L/S combine weight table entry 0 byte 0
    (0x5B97, 0x80),  # cmb_weight0_1: L/S combine weight table entry 0 byte 1
    (0x5B98, 0x80),  # cmb_weight0_2: L/S combine weight table entry 0 byte 2
    (0x5B9A, 0x80),  # cmb_weight1_0: L/S combine weight table entry 1 byte 0
    (0x5B9B, 0x80),  # cmb_weight1_1: L/S combine weight table entry 1 byte 1
    (0x5B9C, 0x80),  # cmb_weight1_2: L/S combine weight table entry 1 byte 2
    (0x5B9D, 0x10),  # cmb_weight1_3: L/S combine weight table entry 1 byte 3
    (0x5B9E, 0x80),  # cmb_weight2_0: L/S combine weight table entry 2 byte 0
    (0x5B9F, 0x80),  # cmb_weight2_1: L/S combine weight table entry 2 byte 1
    (0x5BA0, 0x80),  # cmb_weight2_2: L/S combine weight table entry 2 byte 2
    (0x5BA1, 0x80),  # cmb_weight2_3: L/S combine weight table entry 2 byte 3
    (0x5BA2, 0x00),  # cmb_weight3_0: L/S combine weight table entry 3 byte 0
    (0x5BA3, 0x80),  # cmb_weight3_1: L/S combine weight table entry 3 byte 1
    (0x5BA4, 0x80),  # cmb_weight3_2: L/S combine weight table entry 3 byte 2
    (0x5BA5, 0x80),  # cmb_weight3_3: L/S combine weight table entry 3 byte 3
    (0x5BA6, 0x40),  # cmb_weight4_4: L/S combine weight table entry 4 byte 4
    (0x5BA7, 0x00),  # cmb_weight4_5: L/S combine weight table entry 4 byte 5
    (
        0x5BA8,
        0x02,
    ),  # cmb_thre_l0 high: knee point threshold 0 for long-exposure weight, high byte
    (
        0x5BA9,
        0x80,
    ),  # cmb_thre_l0 low: knee point threshold 0 for long-exposure weight, low byte
    (
        0x5BAA,
        0x03,
    ),  # cmb_thre_l1 high: knee point threshold 1 for long-exposure weight, high byte
    (
        0x5BAB,
        0x00,
    ),  # cmb_thre_l1 low: knee point threshold 1 for long-exposure weight, low byte
    (
        0x5BAC,
        0x03,
    ),  # cmb_thre_l2 high: knee point threshold 2 for long-exposure weight, high byte
    (
        0x5BAD,
        0x80,
    ),  # cmb_thre_l2 low: knee point threshold 2 for long-exposure weight, low byte
    (
        0x5BAE,
        0x02,
    ),  # cmb_thre_s0 high: knee point threshold 0 for short-exposure weight, high byte
    (
        0x5BAF,
        0x80,
    ),  # cmb_thre_s0 low: knee point threshold 0 for short-exposure weight, low byte
    (
        0x5BB0,
        0x0C,
    ),  # cmb_thre_s1 high: knee point threshold 1 for short-exposure weight, high byte
    (
        0x5BB1,
        0x00,
    ),  # cmb_thre_s1 low: knee point threshold 1 for short-exposure weight, low byte
    (
        0x5BB2,
        0x0E,
    ),  # cmb_thre_s2 high: knee point threshold 2 for short-exposure weight, high byte
    (
        0x5BB3,
        0x00,
    ),  # cmb_thre_s2 low: knee point threshold 2 for short-exposure weight, low byte
    (
        0x5BB4,
        0x0F,
    ),  # cmb_thre_s3 high: knee point threshold 3 for short-exposure weight, high byte
    (
        0x5BB5,
        0xFF,
    ),  # cmb_thre_s3 low: knee point threshold 3 for short-exposure weight, low byte
    (
        0x5BBD,
        0x33,
    ),  # COMBINE_ISP_CTRL_3D: HDR combine ratio error and stability control
]

# PWL12 BLC4 tone-map table (0x4027-0x5E84).
ox08d10_pwl12_blc4 = [
    # BLC target registers: black level correction target in ADU for each exposure.
    (0x4027, 0x43),  # BLK_LVL_TARGET_HCG_L: BLC target for HCG (long) exposure = 67 ADU
    (
        0x4029,
        0x43,
    ),  # BLK_LVL_TARGET_LCG_L: BLC target for LCG (medium) exposure = 67 ADU
    (
        0x402B,
        0x43,
    ),  # BLK_LVL_TARGET_LOFIC_L: BLC target for LOFIC (short) exposure = 67 ADU
    (
        0x402D,
        0x43,
    ),  # BLK_LVL_TARGET_VS_L: BLC target for VS (very short) exposure = 67 ADU
    # PWL compression LUT (section 5.2): piecewise linear tone mapping from 24-bit HDR to 12-bit output.
    # 0x5E00: mode control; 0x5E01-0x5E1F: X knee point dx exponents (slope per segment);
    # 0x5E20-0x5E84: Y knee point dy values (output values), three bytes per knee point.
    (0x5E00, 0x00),  # NON_RISC_00: PWL output mode = 24->12 bit compression
    # X knee point dx exponents: each value is the bit-shift exponent for the X-axis interval width
    (0x5E01, 0x06),  # NON_RISC_01: dx_exp[0] = 6 (X interval = 2^6 = 64)
    (0x5E02, 0x06),  # NON_RISC_02: dx_exp[1] = 6
    (0x5E03, 0x06),  # NON_RISC_03: dx_exp[2] = 6
    (0x5E04, 0x07),  # NON_RISC_04: dx_exp[3] = 7
    (0x5E05, 0x08),  # NON_RISC_05: dx_exp[4] = 8
    (0x5E06, 0x09),  # NON_RISC_06: dx_exp[5] = 9
    (0x5E07, 0x0A),  # NON_RISC_07: dx_exp[6] = 10
    (0x5E08, 0x0B),  # NON_RISC_08: dx_exp[7] = 11
    (0x5E09, 0x0C),  # NON_RISC_09: dx_exp[8] = 12
    (0x5E0A, 0x0D),  # NON_RISC_0A: dx_exp[9] = 13
    (0x5E0B, 0x0E),  # NON_RISC_0B: dx_exp[10] = 14
    (0x5E0C, 0x0F),  # NON_RISC_0C: dx_exp[11] = 15
    (0x5E0D, 0x10),  # NON_RISC_0D: dx_exp[12] = 16
    (0x5E0E, 0x11),  # NON_RISC_0E: dx_exp[13] = 17
    (0x5E0F, 0x12),  # NON_RISC_0F: dx_exp[14] = 18
    (0x5E10, 0x13),  # NON_RISC_10: dx_exp[15] = 19
    (0x5E11, 0x14),  # NON_RISC_11: dx_exp[16] = 20
    (0x5E12, 0x15),  # NON_RISC_12: dx_exp[17] = 21
    (0x5E13, 0x16),  # NON_RISC_13: dx_exp[18] = 22
    (0x5E14, 0x16),  # NON_RISC_14: dx_exp[19] = 22
    (0x5E15, 0x15),  # NON_RISC_15: dx_exp[20] = 21
    (0x5E16, 0x14),  # NON_RISC_16: dx_exp[21] = 20
    (0x5E17, 0x13),  # NON_RISC_17: dx_exp[22] = 19
    (0x5E18, 0x12),  # NON_RISC_18: dx_exp[23] = 18
    (0x5E19, 0x11),  # NON_RISC_19: dx_exp[24] = 17
    (0x5E1A, 0x10),  # NON_RISC_1A: dx_exp[25] = 16
    (0x5E1B, 0x0F),  # NON_RISC_1B: dx_exp[26] = 15
    (0x5E1C, 0x0E),  # NON_RISC_1C: dx_exp[27] = 14
    (0x5E1D, 0x0D),  # NON_RISC_1D: dx_exp[28] = 13
    (0x5E1E, 0x0C),  # NON_RISC_1E: dx_exp[29] = 12
    (0x5E1F, 0x00),  # NON_RISC_1F: dx_exp[30] = 0
    # Y knee point dy values: three bytes per knee point (24-bit output value)
    (0x5E20, 0x00),  # NON_RISC_20: dy_val[0] byte 2 (MSB)
    (0x5E21, 0x00),  # NON_RISC_21: dy_val[0] byte 1
    (0x5E22, 0x00),  # NON_RISC_22: dy_val[0] byte 0 (LSB)
    (0x5E23, 0x00),  # NON_RISC_23: dy_val[1] byte 2
    (0x5E24, 0x00),  # NON_RISC_24: dy_val[1] byte 1
    (0x5E25, 0x00),  # NON_RISC_25: dy_val[1] byte 0
    (0x5E26, 0x00),  # NON_RISC_26: dy_val[2] byte 2
    (0x5E27, 0x40),  # NON_RISC_27: dy_val[2] byte 1
    (0x5E28, 0x00),  # NON_RISC_28: dy_val[2] byte 0
    (0x5E29, 0x00),  # NON_RISC_29: dy_val[3] byte 2
    (0x5E2A, 0x2D),  # NON_RISC_2A: dy_val[3] byte 1
    (0x5E2B, 0x00),  # NON_RISC_2B: dy_val[3] byte 0
    (0x5E2C, 0x00),  # NON_RISC_2C: dy_val[4] byte 2
    (0x5E2D, 0x4E),  # NON_RISC_2D: dy_val[4] byte 1
    (0x5E2E, 0x00),  # NON_RISC_2E: dy_val[4] byte 0
    (0x5E2F, 0x00),  # NON_RISC_2F: dy_val[5] byte 2
    (0x5E30, 0x86),  # NON_RISC_30: dy_val[5] byte 1
    (0x5E31, 0x00),  # NON_RISC_31: dy_val[5] byte 0
    (0x5E32, 0x00),  # NON_RISC_32: dy_val[6] byte 2
    (0x5E33, 0xE4),  # NON_RISC_33: dy_val[6] byte 1
    (0x5E34, 0x00),  # NON_RISC_34: dy_val[6] byte 0
    (0x5E35, 0x01),  # NON_RISC_35: dy_val[7] byte 2
    (0x5E36, 0x88),  # NON_RISC_36: dy_val[7] byte 1
    (0x5E37, 0x00),  # NON_RISC_37: dy_val[7] byte 0
    (0x5E38, 0x02),  # NON_RISC_38: dy_val[8] byte 2
    (0x5E39, 0xA0),  # NON_RISC_39: dy_val[8] byte 1
    (0x5E3A, 0x00),  # NON_RISC_3A: dy_val[8] byte 0
    (0x5E3B, 0x00),  # NON_RISC_3B: dy_val[9] byte 2
    (0x5E3C, 0x90),  # NON_RISC_3C: dy_val[9] byte 1
    (0x5E3D, 0x00),  # NON_RISC_3D: dy_val[9] byte 0
    (0x5E3E, 0x00),  # NON_RISC_3E: dy_val[10] byte 2
    (0x5E3F, 0x63),  # NON_RISC_3F: dy_val[10] byte 1
    (0x5E40, 0x00),  # NON_RISC_40: dy_val[10] byte 0
    (0x5E41, 0x00),  # NON_RISC_41: dy_val[11] byte 2
    (0x5E42, 0xAA),  # NON_RISC_42: dy_val[11] byte 1
    (0x5E43, 0x00),  # NON_RISC_43: dy_val[11] byte 0
    (0x5E44, 0x00),  # NON_RISC_44: dy_val[12] byte 2
    (0x5E45, 0x79),  # NON_RISC_45: dy_val[12] byte 1
    (0x5E46, 0x00),  # NON_RISC_46: dy_val[12] byte 0
    (0x5E47, 0x00),  # NON_RISC_47: dy_val[13] byte 2
    (0x5E48, 0x4E),  # NON_RISC_48: dy_val[13] byte 1
    (0x5E49, 0x00),  # NON_RISC_49: dy_val[13] byte 0
    (0x5E4A, 0x00),  # NON_RISC_4A: dy_val[14] byte 2
    (0x5E4B, 0x86),  # NON_RISC_4B: dy_val[14] byte 1
    (0x5E4C, 0x00),  # NON_RISC_4C: dy_val[14] byte 0
    (0x5E4D, 0x00),  # NON_RISC_4D: dy_val[15] byte 2
    (0x5E4E, 0x62),  # NON_RISC_4E: dy_val[15] byte 1
    (0x5E4F, 0x00),  # NON_RISC_4F: dy_val[15] byte 0
    (0x5E50, 0x00),  # NON_RISC_50: dy_val[16] byte 2
    (0x5E51, 0xA8),  # NON_RISC_51: dy_val[16] byte 1
    (0x5E52, 0x00),  # NON_RISC_52: dy_val[16] byte 0
    (0x5E53, 0x01),  # NON_RISC_53: dy_val[17] byte 2
    (0x5E54, 0x21),  # NON_RISC_54: dy_val[17] byte 1
    (0x5E55, 0x00),  # NON_RISC_55: dy_val[17] byte 0
    (0x5E56, 0x00),  # NON_RISC_56: dy_val[18] byte 2
    (0x5E57, 0x91),  # NON_RISC_57: dy_val[18] byte 1
    (0x5E58, 0x00),  # NON_RISC_58: dy_val[18] byte 0
    (0x5E59, 0x00),  # NON_RISC_59: dy_val[19] byte 2
    (0x5E5A, 0xF9),  # NON_RISC_5A: dy_val[19] byte 1
    (0x5E5B, 0x00),  # NON_RISC_5B: dy_val[19] byte 0
    (0x5E5C, 0x01),  # NON_RISC_5C: dy_val[20] byte 2
    (0x5E5D, 0x8A),  # NON_RISC_5D: dy_val[20] byte 1
    (0x5E5E, 0x00),  # NON_RISC_5E: dy_val[20] byte 0
    (0x5E5F, 0x00),  # NON_RISC_5F: dy_val[21] byte 2
    (0x5E60, 0xC5),  # NON_RISC_60: dy_val[21] byte 1
    (0x5E61, 0x00),  # NON_RISC_61: dy_val[21] byte 0
    (0x5E62, 0x00),  # NON_RISC_62: dy_val[22] byte 2
    (0x5E63, 0x63),  # NON_RISC_63: dy_val[22] byte 1
    (0x5E64, 0x00),  # NON_RISC_64: dy_val[22] byte 0
    (0x5E65, 0x00),  # NON_RISC_65: dy_val[23] byte 2
    (0x5E66, 0x31),  # NON_RISC_66: dy_val[23] byte 1
    (0x5E67, 0x00),  # NON_RISC_67: dy_val[23] byte 0
    (0x5E68, 0x00),  # NON_RISC_68: dy_val[24] byte 2
    (0x5E69, 0x18),  # NON_RISC_69: dy_val[24] byte 1
    (0x5E6A, 0x00),  # NON_RISC_6A: dy_val[24] byte 0
    (0x5E6B, 0x00),  # NON_RISC_6B: dy_val[25] byte 2
    (0x5E6C, 0x0D),  # NON_RISC_6C: dy_val[25] byte 1
    (0x5E6D, 0x00),  # NON_RISC_6D: dy_val[25] byte 0
    (0x5E6E, 0x00),  # NON_RISC_6E: dy_val[26] byte 2
    (0x5E6F, 0x06),  # NON_RISC_6F: dy_val[26] byte 1
    (0x5E70, 0x00),  # NON_RISC_70: dy_val[26] byte 0
    (0x5E71, 0x00),  # NON_RISC_71: dy_val[27] byte 2
    (0x5E72, 0x03),  # NON_RISC_72: dy_val[27] byte 1
    (0x5E73, 0x00),  # NON_RISC_73: dy_val[27] byte 0
    (0x5E74, 0x00),  # NON_RISC_74: dy_val[28] byte 2
    (0x5E75, 0x01),  # NON_RISC_75: dy_val[28] byte 1
    (0x5E76, 0x00),  # NON_RISC_76: dy_val[28] byte 0
    (0x5E77, 0x00),  # NON_RISC_77: dy_val[29] byte 2
    (0x5E78, 0x01),  # NON_RISC_78: dy_val[29] byte 1
    (0x5E79, 0x00),  # NON_RISC_79: dy_val[29] byte 0
    (0x5E7A, 0x00),  # NON_RISC_7A: dy_val[30] byte 2
    (0x5E7B, 0x00),  # NON_RISC_7B: dy_val[30] byte 1
    (0x5E7C, 0x00),  # NON_RISC_7C: dy_val[30] byte 0
    (0x5E7D, 0x00),  # NON_RISC_7D: dy_val[31] byte 2
    (0x5E7E, 0x00),  # NON_RISC_7E: dy_val[31] byte 1
    (0x5E7F, 0x00),  # NON_RISC_7F: dy_val[31] byte 0
    (0x5E80, 0x00),  # NON_RISC_80: dy_val[32] byte 2
    (0x5E81, 0x00),  # NON_RISC_81: dy_val[32] byte 1
    (0x5E82, 0x00),  # NON_RISC_82: dy_val[32] byte 0
    (0x5E83, 0x00),  # NON_RISC_83: dy_val[33] byte 2
    (0x5E84, 0x00),  # NON_RISC_84: dy_val[33] byte 1
]

# Internal frame sync configuration (master free-run mode).
ox08d10_fsync_internal = [
    (0x3015, 0x0A),  # SC_CTRL_15: FSYNC pad output enable
    (
        OX08D10_TABLE_WRITE_MASKED,
        (0x3009, 0x02, 0x02),
    ),  # SC_CTRL_9: FSYNC pad input enable (bit 1)
    (0x3822, 0x04),  # Y_INC_ISP_1: ISP frame increment config
    (0x3823, 0x50),  # timing control (frame increment high nibble)
    (0x383E, 0x81),  # TIMING_CTRL_3E: internal FSYNC master enable
    (0x3881, 0x30),  # FSYNC period high byte [23:16]
    (0x3882, 0x05),  # FSYNC period mid byte [15:8]
    (0x3883, 0x35),  # FSYNC period low byte [7:0]
    (0x388E, 0x01),  # FSYNC sync pulse width high byte
    (0x388F, 0x00),  # FSYNC sync pulse width low byte
    (0x3892, 0x44),  # FSYNC output pin mux config
    (0x3826, 0x00),  # TC_R_INIT_MAN_H: timing counter init high byte
    (0x3827, 0x29),  # TC_R_INIT_MAN_L: timing counter init low byte
    (0x3836, 0x00),  # THRESHOLD_VTS_SUB_H: sub-frame VTS threshold high
    (0x3837, 0x0A),  # THRESHOLD_VTS_SUB_L: sub-frame VTS threshold low
]

# Mirror/flip orientation registers for GMSL module mounting.
ox08d10_flip_adjust = [
    (
        OX08D10_TABLE_WRITE_MASKED,
        (0x3820, 0x44, 0x64),
    ),  # IMAGE_FORMAT: V-flip + BLC V-flip enable (bits 6,2)
    (
        OX08D10_TABLE_WRITE_MASKED,
        (0x4018, 0x60, 0x60),
    ),  # MIPI output: R/B channel reversal for flip (bits 6,5)
    (
        OX08D10_TABLE_WRITE_MASKED,
        (0x2800, 0x08, 0x18),
    ),  # XCHK: cross-check flip mirror mask (bit 3)
    (
        OX08D10_TABLE_WRITE_MASKED,
        (0x2803, 0x01, 0x01),
    ),  # XCHK_DFD: cross-check DFD flip enable (bit 0)
    (
        OX08D10_TABLE_WRITE_MASKED,
        (0x4640, 0x42, 0x43),
    ),  # UID: flip/mirror flags update (bits 1,6)
    (
        OX08D10_TABLE_WRITE_MASKED,
        (0x6909, 0x01, 0x01),
    ),  # CFA: maintain CFA pattern on mirror (bit 0)
]

# Safety configuration for flip mode.
ox08d10_safety_config_flip = [
    (
        0x4613,
        0x42,
    ),  # VFIFO_DTR_CRC_REF_A[7:0]: VFIFO CRC reference byte 0 for flip mode
    (0x4614, 0x9F),  # VFIFO_DTR_CRC_REF_A[15:8]: VFIFO CRC reference byte 1
    (0x4615, 0x06),  # VFIFO_DTR_CRC_REF_A[23:16]: VFIFO CRC reference byte 2
    (0x4616, 0xE6),  # VFIFO_DTR_CRC_REF_A[31:24]: VFIFO CRC reference byte 3
    (0x507A, 0x7D),  # STAT_CRC_GLD0: statistics CRC golden reference byte 0
    (0x507B, 0x09),  # STAT_CRC_GLD1: statistics CRC golden reference byte 1
]

# Clear ISP channel-select registers after flip safety config.
ox08d10_isp_clear = [
    (0x5288, 0x00),  # ISP HCG channel select: clear
    (0x5488, 0x00),  # ISP LCG channel select: clear
    (0x5688, 0x00),  # ISP LOFIC channel select: clear
    (0x5888, 0x00),  # ISP VS channel select: clear
]

# System defaults: pixel, analog, ISP, and calibration registers.
ox08d10_system_defaults = [
    # Temperature sensor calibration coefficients (section 3.8).
    # 0x3017 gates write access to calibration registers; written 0xF8 to enable,
    # then 0xF0 to restore normal operation after coefficients are loaded.
    (0x3017, 0xF8),  # SC_CTRL_17: enable temperature calibration write mode
    (0x4D17, 0x0A),  # TPM_REG_17: sensor 0 calibration slope high byte
    (0x4D18, 0x00),  # TPM_REG_18: sensor 0 calibration slope low byte
    (0x4D22, 0x0A),  # TPM_REG_22: sensor 1 calibration slope high byte
    (0x4D23, 0x00),  # TPM_REG_23: sensor 1 calibration slope low byte
    (0x4D26, 0x0A),  # TPM_REG_26: average calibration slope high byte
    (0x4D27, 0x00),  # TPM_REG_27: average calibration slope low byte
    (0x4D19, 0xAF),  # TPM_REG_19: sensor 0 calibration offset high byte
    (0x4D1A, 0x00),  # TPM_REG_1A: sensor 0 calibration offset low byte
    (0x4D24, 0xAF),  # TPM_REG_24: sensor 1 calibration offset high byte
    (0x4D25, 0x00),  # TPM_REG_25: sensor 1 calibration offset low byte
    (0x4D28, 0xAF),  # TPM_REG_28: average calibration offset high byte
    (0x4D29, 0x00),  # TPM_REG_29: average calibration offset low byte
    (0x4D1B, 0x01),  # TPM_REG_1B: sensor 0 calibration coefficient high byte
    (0x4D1C, 0xB8),  # TPM_REG_1C: sensor 0 calibration coefficient low byte
    (0x3017, 0xF0),  # SC_CTRL_17: exit temperature calibration write mode
    (0x4D36, 0x26),  # TPM_REG_36: calibration update trigger
    # Group hold sequences for embedded row configuration (section 5.5).
    # 0x3208 GROUP_ACCESS: upper nibble = command (0=hold start, 1=hold end, A=delay launch),
    #                       lower nibble = group number (0-10).
    (0x3208, 0x04),  # GROUP_ACCESS: group 4 hold start
    (0x462A, 0x04),  # EBD channel select (stored in group 4 SRAM)
    (0x300A, 0x03),  # EBD row config (stored in group 4 SRAM)
    (0x302A, 0x01),  # EBD row config (stored in group 4 SRAM)
    (0x3208, 0x14),  # GROUP_ACCESS: group 4 hold end
    (0x3208, 0x05),  # GROUP_ACCESS: group 5 hold start
    (0x462A, 0x04),  # EBD channel select (stored in group 5 SRAM)
    (0x300A, 0x03),  # EBD row config (stored in group 5 SRAM)
    (0x302A, 0x01),  # EBD row config (stored in group 5 SRAM)
    (0x3208, 0x15),  # GROUP_ACCESS: group 5 hold end
    (0x3208, 0x02),  # GROUP_ACCESS: group 2 hold start
    (
        0x3507,
        0x00,
    ),  # EXP_MANUAL: manual exposure override disable (stored in group 2 SRAM)
    (0x3208, 0x12),  # GROUP_ACCESS: group 2 hold end
    (0x3208, 0xA2),  # GROUP_ACCESS: delay launch group 2 (0xA = delay launch command)
    (0x3208, 0x0A),  # GROUP_ACCESS: group 10 hold start (lower nibble 0xA = group 10)
    (
        0x3507,
        0x00,
    ),  # EXP_MANUAL: manual exposure override disable (stored in group 10 SRAM)
    (0x3208, 0x1A),  # GROUP_ACCESS: group 10 hold end
    # Pixel and analog settings
    (0x0362, 0xA8),  # PLL2_FILTER: PLL2 loop filter / baseband resistance
    (0x4505, 0x30),  # ISP_CTRL5: ISP channel control
    (0x3B40, 0x05),  # ANALOG_CTRL_40: pixel row checker threshold
    (0x3B41, 0x40),  # ANALOG_CTRL_41: pixel row checker threshold
    (0x3B42, 0x01),  # ANALOG_CTRL_42: pixel row checker config
    (0x3B43, 0x10),  # ANALOG_CTRL_43: pixel row checker config
    (0x3B45, 0x80),  # ANALOG_CTRL_45: pixel row checker config
    (0x3B47, 0x80),  # ANALOG_CTRL_47: pixel row checker config
    (0x3B48, 0x0E),  # ANALOG_CTRL_48: analog timing control
    (0x3B49, 0x09),  # ANALOG_CTRL_49: analog timing control
    (0x3B4A, 0x0C),  # ANALOG_CTRL_4A: analog timing control
    (0x3B4B, 0xA5),  # ANALOG_CTRL_4B: analog timing control
    (0x3B4D, 0x44),  # ANALOG_CTRL_4D: analog bias setting
    (0x3B4E, 0x44),  # ANALOG_CTRL_4E: analog bias setting
    (0x3704, 0x00),  # ANALOG_CTRL: analog timing setting
    (0x3730, 0x01),  # ANALOG_CTRL: analog ramp control
    (0x3739, 0x01),  # ANALOG_CTRL: analog ramp control
    (0x374F, 0x4C),  # TIMING_GEN_CRC_REF byte 0 (45fps default)
    (0x3750, 0x35),  # TIMING_GEN_CRC_REF byte 1 (45fps default)
    (0x3751, 0x42),  # TIMING_GEN_CRC_REF byte 2 (45fps default)
    (0x3752, 0x19),  # TIMING_GEN_CRC_REF byte 3 (45fps default)
    (0x3760, 0x01),  # ANALOG_CTRL: analog column amplifier control
    (0x377A, 0x01),  # ANALOG_CTRL: analog settling time
    (0x377B, 0x01),  # ANALOG_CTRL: analog settling time
    (0x377C, 0x01),  # ANALOG_CTRL: analog settling time
    (0x2803, 0x01),  # XCHK_DFD: cross-check DFD flip enable
    (0x3101, 0xB2),  # SC_CTRL: sensor system control
    (0x3400, 0x0C),  # MWB_GAIN_CTRL: manual white balance gain control
    (0x3406, 0x01),  # MWB_R_GAIN_H: manual WB R-channel gain high byte
    (0x3407, 0x01),  # MWB_R_GAIN_L: manual WB R-channel gain low byte
    (0x3408, 0x02),  # MWB_G_GAIN_H: manual WB G-channel gain high byte
    (0x3409, 0x0C),  # MWB_G_GAIN_L: manual WB G-channel gain low byte
    (0x340B, 0x0B),  # MWB_B_GAIN: manual WB B-channel gain
    (0x3426, 0x10),  # ANALOG_CTRL: analog control setting
    (0x3429, 0x01),  # ANALOG_CTRL: analog control setting
    (0x342B, 0x00),  # ANALOG_CTRL: analog control setting
    (0x3589, 0x60),  # LOFIC analog control
    (0x354A, 0x01),  # LOFIC gain control high byte
    (0x354B, 0xA0),  # LOFIC gain control low byte
    (0x35C9, 0x60),  # VS analog control
    (0x3611, 0x80),  # ANALOG_CTRL: column bias setting
    (0x3616, 0x30),  # ANALOG_CTRL: column bias setting
    (0x361D, 0x0B),  # ANALOG_CTRL: ADC reference setting
    (0x3620, 0xC0),  # ANALOG_CTRL: ADC control
    (0x3621, 0x02),  # ANALOG_CTRL: ADC control
    (0x3627, 0x3C),  # ANALOG_CTRL: comparator control
    (0x3634, 0x3C),  # ANALOG_CTRL: ramp generator control
    (0x3636, 0x52),  # ANALOG_CTRL: ramp generator control
    (0x3637, 0xCA),  # ANALOG_CTRL: ramp generator control
    (0x363B, 0xE4),  # ANALOG_CTRL: analog timing
    (0x3673, 0x11),  # ANALOG_CTRL: column timing 0
    (0x3674, 0x0A),  # ANALOG_CTRL: column timing 1
    (0x3675, 0x15),  # ANALOG_CTRL: column timing 2
    (0x3676, 0x17),  # ANALOG_CTRL: column timing 3
    (0x3677, 0x10),  # ANALOG_CTRL: column timing 4
    (0x3678, 0x09),  # ANALOG_CTRL: column timing 5
    (0x3679, 0x14),  # ANALOG_CTRL: column timing 6
    (0x367A, 0x16),  # ANALOG_CTRL: column timing 7
    (0x367C, 0x09),  # ANALOG_CTRL: column timing 8
    (0x3688, 0x10),  # ANALOG_CTRL: ADC column control 0
    (0x3689, 0x12),  # ANALOG_CTRL: ADC column control 1
    (0x368F, 0x12),  # ANALOG_CTRL: ADC column control 2
    (0x3690, 0x13),  # ANALOG_CTRL: ADC column control 3
    (0x3691, 0x0C),  # ANALOG_CTRL: ADC column control 4
    (0x3692, 0x36),  # ANALOG_CTRL: ADC column control 5
    (0x380E, 0x06),  # VTS_H: total rows per frame high byte (45fps default)
    (0x380F, 0xA0),  # VTS_L: total rows per frame low byte (45fps default)
    (0x3B81, 0x0F),  # ANALOG_CTRL_81: analog row checker threshold high byte
    (0x3B82, 0x00),  # ANALOG_CTRL_82: analog row checker threshold low byte
    (0x3B84, 0x0F),  # ANALOG_CTRL_84: analog row checker threshold high byte
    (0x3B85, 0x00),  # ANALOG_CTRL_85: analog row checker threshold low byte
    (0x430D, 0x93),  # MIPI_CTRL_0D: MIPI data lane timing control
    (0x4327, 0xF0),  # MIPI_CTRL_27: MIPI LP/HS transition timing
    (0x4D09, 0x9F),  # TPM_REG_9: temperature monitor control
    (0x4D16, 0xFF),  # TPM_REG_16: temperature calibration register
    (0x4D20, 0xFF),  # TPM_REG_20: temperature calibration register
    (0x4D21, 0xFF),  # TPM_REG_21: temperature calibration register
    (0x4D37, 0x00),  # TPM_REG_37: temperature monitor config
    (0x4022, 0x80),  # BLC_CTRL_22: black level correction control
    (0x408A, 0x01),  # PLL_MONITOR high byte
    (0x408B, 0x40),  # PLL_MONITOR low byte
    # ISP calibration LUT coefficients for chromatic aberration and denoise (CA+DNS)
    (0x6A04, 0x08),
    (0x6A05, 0x02),
    (0x6A07, 0x16),
    (0x6A08, 0xA7),
    (0x6A0A, 0x1A),
    (0x6A0B, 0x1B),
    (0x6A0D, 0x1C),
    (0x6A0E, 0x08),
    (0x6A10, 0x23),
    (0x6A11, 0x13),
    (0x6A13, 0x25),
    (0x6A14, 0xA8),
    (0x6A16, 0x2A),
    (0x6A17, 0x02),
    (0x6A19, 0x2C),
    (0x6A1A, 0x08),
    (0x6A1C, 0x2E),
    (0x6A1D, 0x26),
    (0x6A1F, 0x36),
    (0x6A20, 0xF8),
    (0x6A22, 0x38),
    (0x6A23, 0x0A),
    (0x6A25, 0x3F),
    (0x6A26, 0x15),
    (0x6A28, 0x41),
    (0x6A29, 0x0B),
    (0x6A2B, 0x43),
    (0x6A2C, 0x06),
    (0x6A2E, 0x45),
    (0x6A2F, 0x02),
    (0x6A31, 0x4B),
    (0x6A32, 0x02),
    (0x6A34, 0x4D),
    (0x6A35, 0x73),
    (0x6A37, 0x4F),
    (0x6A38, 0x02),
    (0x6A3A, 0x51),
    (0x6A3B, 0x0C),
    (0x6A3D, 0x53),
    (0x6A3E, 0x2F),
    (0x6A40, 0x65),
    (0x6A41, 0x03),
    (0x6A43, 0x67),
    (0x6A44, 0x2B),
    (0x6A46, 0x6D),
    (0x6A47, 0x02),
    (0x6A49, 0x76),
    (0x6A4A, 0x04),
    (0x6A4C, 0x78),
    (0x6A4D, 0x03),
    (0x6A4F, 0x7A),
    (0x6A50, 0x03),
    (0x6A52, 0x7C),
    (0x6A53, 0x03),
    (0x6A55, 0x7E),
    (0x6A56, 0x54),
    (0x6A58, 0x80),
    (0x6A59, 0x03),
    (0x6A5B, 0x82),
    (0x6A5C, 0x03),
    (0x6A5E, 0x84),
    (0x6A61, 0x86),
    (0x6A62, 0x1E),
    (0x6A64, 0x88),
    (0x6A67, 0x8B),
    (0x6A68, 0x2F),
    (0x6A6A, 0x8D),
    (0x6A6B, 0x06),
    (0x6A6D, 0x94),
    (0x6A6E, 0x06),
    (0x6A70, 0x98),
    (0x6A71, 0x21),
    (0x6A73, 0x9A),
    (0x6A74, 0x30),
    (0x6A76, 0x9C),
    (0x6A77, 0x48),
    (0x6A79, 0xA3),
    (0x6A7A, 0x1B),
    (0x6A7C, 0xB6),
    (0x6A7D, 0x02),
    (0x6A7F, 0xBA),
    (0x6A80, 0x12),
    (0x6A82, 0xC2),
    (0x6A83, 0x03),
    (0x6A85, 0xC6),
    (0x6A86, 0x04),
    (0x6A88, 0xCA),
    (0x6A89, 0x03),
    (0x6A8B, 0xCF),
    (0x6A8C, 0x03),
    (0x6A8E, 0xD1),
    (0x6A8F, 0x14),
    (0x6A91, 0xD8),
    (0x6A92, 0x0D),
    (0x6A94, 0xDD),
    (0x6A95, 0x5B),
    (0x6A97, 0xDF),
    (0x6A98, 0x03),
    (0x6A9A, 0xE8),
    (0x6A9B, 0x2D),
    (0x6A9D, 0xEF),
    (0x6A9E, 0x01),
    (0x6AA0, 0xF3),
    (0x6AA1, 0x5F),
    (0x6AA3, 0xF5),
    (0x6AA4, 0x05),
    (0x6AA6, 0xFA),
    (0x6AA7, 0x21),
    (0x6AA9, 0xFE),
    (0x6AAA, 0x33),
    (0x6AAC, 0x00),
    (0x6AAD, 0x00),
    (0x6AAF, 0x00),
    (0x6AB0, 0x00),
    (0x6AB2, 0x00),
    (0x6AB3, 0x00),
    (0x6AB5, 0x00),
    (0x6AB6, 0x00),
    (0x6AB8, 0x00),
    (0x6AB9, 0x00),
    (0x6ABB, 0x00),
    (0x6ABC, 0x00),
    (0x6ABE, 0x00),
    (0x6ABF, 0x00),
    (0x6AC1, 0x00),
    (0x6AC2, 0x00),
    (0x6AC4, 0x00),
    (0x6AC5, 0x00),
    # 6b** block: ISP calibration LUT coefficients (continued)
    (0x6B1D, 0x37),
    (0x6B1E, 0x42),
    (0x6B1F, 0x64),
    (0x6B20, 0x68),
    (0x6B21, 0x7D),
    (0x6B22, 0x8C),
    (0x6B27, 0x05),
    (0x6B28, 0x6C),
    (0x6B29, 0x89),
    (0x6B2A, 0x8E),
    (0x6B2D, 0x37),
    (0x6B2E, 0x44),
    (0x6B2F, 0x64),
    (0x6B30, 0x68),
    (0x6B31, 0x7B),
    (0x6B32, 0x9B),
    (0x6B35, 0x64),
    (0x6B36, 0x68),
    (0x6B3B, 0x37),
    (0x6B3C, 0x44),
    (0x6B3D, 0x64),
    (0x6B3E, 0x6B),
    (0x6B3F, 0x79),
    (0x6B40, 0x95),
    (0x6B41, 0xCC),
    (0x6B42, 0xCE),
    (0x6B45, 0xC3),
    (0x6B46, 0xC5),
    (0x6B49, 0x77),
    (0x6B4A, 0x87),
    (0x6B4B, 0x93),
    (0x6B4C, 0xA1),
    (0x6B4D, 0x00),
    (0x6B4E, 0x00),
    (0x6B59, 0x05),
    (0x6B5A, 0x6C),
    (0x6B5B, 0x89),
    (0x6B5C, 0x8E),
    (0x6B5F, 0x89),
    (0x6B60, 0x99),
    (0x6B65, 0x93),
    (0x6B66, 0x87),
    (0x6B68, 0xA1),
    (0x6B6D, 0x19),
    (0x6B6E, 0x20),
    (0x6B6F, 0x3B),
    (0x6B71, 0x89),
    (0x6B72, 0x99),
    (0x6B73, 0xE4),
    (0x6B74, 0xEE),
    (0x6B78, 0x1B),
    (0x6B79, 0x3A),
    (0x6B7A, 0xFC),
    (0x6B7D, 0x89),
    (0x6B7E, 0x8E),
    (0x6B7F, 0xBE),
    (0x6B80, 0xD4),
    (0x6B87, 0x2B),
    (0x6B88, 0x2F),
    (0x6B89, 0x3E),
    (0x6B8B, 0x70),
    (0x6B8C, 0x8C),
    (0x6B8D, 0x97),
    (0x6B8E, 0x9F),
    (0x6B8F, 0xE7),
    (0x6B90, 0xEC),
    (0x6B92, 0x15),
    (0x6B93, 0x66),
    (0x6B94, 0xA2),
    (0x6B95, 0xBE),
    (0x6B96, 0xD9),
    (0x6B98, 0x18),
    (0x6B99, 0x1D),
    (0x6B9A, 0x27),
    (0x6B9B, 0x32),
    (0x6B9C, 0x39),
    (0x6B9E, 0x63),
    (0x6B9F, 0x68),
    (0x6BA0, 0x85),
    (0x6BA1, 0x90),
    (0x6BA2, 0x9D),
    (0x6BA3, 0xBE),
    (0x6BA4, 0xE0),
    (0x6BA5, 0xEB),
    (0x6BA6, 0xFB),
    (0x6BA9, 0x89),
    (0x6BAA, 0x8E),
    (0x6BAB, 0xBE),
    (0x6BAC, 0xCB),
    (0x6BAF, 0xC7),
    (0x6BB0, 0xC9),
    (0x6BB4, 0xF7),
    (0x6BB7, 0x18),
    (0x6BB8, 0x20),
    (0x6BB9, 0x27),
    (0x6BBA, 0x35),
    (0x6BBB, 0x39),
    (0x6BBD, 0x89),
    (0x6BBE, 0x93),
    (0x6BBF, 0xA4),
    (0x6BC0, 0xBD),
    (0x6BC1, 0xE0),
    (0x6BC2, 0xF0),
    (0x6BC3, 0xBE),
    (0x6BC4, 0xDC),
    (0x6BC7, 0x22),
    (0x6BC8, 0x19),
    (0x6BC9, 0x31),
    (0x6BCA, 0x27),
    (0x6BCC, 0x3A),
    (0x6BCD, 0x93),
    (0x6BCE, 0x8A),
    (0x6BCF, 0xBD),
    (0x6BD0, 0xA5),
    (0x6BD1, 0xEA),
    (0x6BD2, 0xE1),
    (0x6BD9, 0x90),
    (0x6BDA, 0x9D),
    (0x6BDB, 0x64),
    (0x6BDC, 0x85),
    (0x6BDD, 0x27),
    (0x6BDE, 0x3A),
    (0x6BDF, 0x18),
    (0x6BE0, 0x26),
    (0x6BE1, 0x0B),
    (0x6BE2, 0x17),
    (0x6BE3, 0xD4),
    (0x6BE4, 0xDE),
    (0x6BEB, 0x3B),
    (0x6BEC, 0x64),
    (0x6BED, 0xDE),
    (0x6BEE, 0xF8),
    (0x6BEF, 0x85),
    (0x6BF0, 0x0A),
    (0x6BF1, 0x9D),
    (0x6BF2, 0x89),
    (0x6BF3, 0xF8),
    (0x6BF4, 0xA4),
    (0x6BF7, 0xA0),
    (0x6BF8, 0xA2),
    (0x6BF9, 0xF6),
    (0x6BFA, 0xF7),
    (0x6BFC, 0x09),
    (0x6BFD, 0x58),
    (0x6BFE, 0xFF),
    (0x6BFF, 0xBE),
    # 6c** block: ISP calibration LUT coefficients (continued)
    (0x6C00, 0xD4),
    (0x6C01, 0xD0),
    (0x6C02, 0xD2),
    (0x6C03, 0xF7),
    (0x6C04, 0xF9),
    (0x6C09, 0x54),
    (0x6C0A, 0x62),
    (0x6C0B, 0xBE),
    (0x6C0C, 0xC1),
    (0x6C11, 0x0B),
    (0x6C12, 0x0D),
    (0x6C13, 0x10),
    (0x6C14, 0x12),
    (0x6C15, 0x54),
    (0x6C16, 0x55),
    (0x6C17, 0x5B),
    (0x6C18, 0x5C),
    (0x6C19, 0x5D),
    (0x6C1A, 0x5E),
    (0x6C1B, 0x60),
    (0x6C1C, 0x63),
    (0x6C1D, 0x6E),
    (0x6C1E, 0x70),
    (0x6C1F, 0x73),
    (0x6C20, 0x75),
    (0x6C21, 0x2B),
    (0x6C22, 0x2D),
    (0x6C25, 0x7F),
    (0x6C26, 0x83),
    (0x6C27, 0xE9),
    (0x6C28, 0xF2),
    (0x6C2E, 0x0D),
    (0x6C32, 0xF4),
    (0x6C3D, 0x24),
    (0x6C3E, 0x3C),
    (0x6C47, 0x29),
    (0x6C48, 0x85),
    (0x6C49, 0xE5),
    (0x6C4A, 0xF7),
    (0x6C4B, 0x2B),
    (0x6C4C, 0x3B),
    (0x6C4D, 0x40),
    (0x6C4F, 0x7F),
    (0x6C50, 0x81),
    (0x6C51, 0xE6),
    (0x6C52, 0xF7),
    (0x6C53, 0x2B),
    (0x6C54, 0x3B),
    (0x6C55, 0x40),
    (0x6C57, 0x7F),
    (0x6C58, 0x81),
    (0x6C59, 0xE6),
    (0x6C5A, 0xF7),
    (0x6C5C, 0xA2),
    (0x6C5D, 0xBB),
    (0x6C5E, 0xFD),
    (0x6C63, 0xA7),
    (0x6C64, 0xA9),
    (0x6C65, 0xAB),
    (0x6C66, 0xAD),
    (0x6C67, 0xAF),
    (0x6C68, 0xB1),
    (0x6C69, 0xB3),
    (0x6C6A, 0xB5),
    (0x6C6B, 0xB7),
    (0x6C6C, 0xB9),
    (0x6C6D, 0xA7),
    (0x6C6E, 0xAC),
    (0x6C70, 0x09),
    (0x6C71, 0x24),
    (0x6C73, 0x0F),
    (0x6C74, 0x15),
    (0x6C77, 0x0B),
    (0x6C79, 0x21),
    (0x6C7A, 0x08),
    (0x6C7C, 0x23),
    (0x6C7D, 0x0A),
    (0x6C7F, 0x2A),
    (0x6C80, 0x0F),
    (0x6C82, 0x31),
    (0x6C85, 0x33),
    (0x6C86, 0x21),
    (0x6C88, 0x37),
    (0x6C8B, 0x3F),
    (0x6C8E, 0x41),
    (0x6C91, 0x43),
    (0x6C94, 0x45),
    (0x6C97, 0x48),
    (0x6C98, 0x07),
    (0x6C9B, 0x20),
    (0x6C9D, 0x54),
    (0x6C9E, 0x2F),
    (0x6CA0, 0x59),
    (0x6CA1, 0x48),
    (0x6CA4, 0x8C),
    (0x6CA6, 0x5D),
    (0x6CA7, 0x69),
    (0x6CAA, 0x30),
    (0x6CAC, 0x65),
    (0x6CAD, 0x21),
    (0x6CAF, 0x6C),
    (0x6CB0, 0x50),
    (0x6CB2, 0x6E),
    (0x6CB3, 0x52),
    (0x6CB5, 0x75),
    (0x6CB6, 0x19),
    (0x6CB8, 0x77),
    (0x6CB9, 0x0B),
    (0x6CBB, 0x88),
    (0x6CBC, 0x10),
    (0x6CBE, 0x8A),
    (0x6CBF, 0x3E),
    (0x6CC1, 0x94),
    (0x6CC2, 0x1F),
    (0x6CC4, 0x96),
    (0x6CC5, 0x0F),
    (0x6CC7, 0x9C),
    (0x6CC8, 0x14),
    (0x6CCA, 0x9E),
    (0x6CCB, 0x3C),
    (0x6CCD, 0xA0),
    (0x6CCE, 0x59),
    (0x6CD0, 0xA4),
    (0x6CD1, 0x22),
    (0x6CD3, 0xAE),
    (0x6CD4, 0x1C),
    (0x6CD6, 0xB5),
    (0x6CD7, 0x08),
    (0x6CD9, 0xB7),
    (0x6CDA, 0x14),
    (0x6CDC, 0xB9),
    (0x6CDD, 0x2A),
    (0x6CE0, 0x1C),
    (0x6CE2, 0xBE),
    (0x6CE3, 0x08),
    (0x6CE5, 0xC0),
    (0x6CE6, 0x15),
    (0x6CE8, 0xC9),
    (0x6CE9, 0x12),
    (0x6CEB, 0xCB),
    (0x6CEC, 0x08),
    (0x6CEE, 0xD0),
    (0x6CEF, 0x06),
    (0x6CF1, 0xD2),
    (0x6CF2, 0x1A),
    (0x6CF4, 0xD4),
    (0x6CF5, 0x0E),
    (0x6CF7, 0xD6),
    (0x6CF8, 0x09),
    (0x6CFA, 0xE3),
    (0x6CFB, 0x0A),
    (0x6CFD, 0xE5),
    (0x6CFE, 0x24),
    # 6d** block: ISP calibration LUT coefficients (continued)
    (0x6D00, 0xED),
    (0x6D01, 0x34),
    (0x6D03, 0xF4),
    (0x6D04, 0x30),
    (0x6D06, 0xF9),
    (0x6D07, 0x59),
    (0x6DBC, 0x00),
    (0x6DBF, 0x0A),
    (0x6DC0, 0x24),
    (0x6DC1, 0xC1),
    (0x6DC2, 0xD3),
    (0x6DC3, 0x3E),
    (0x6DC4, 0x50),
    (0x6DC5, 0x74),
    (0x6DC6, 0x87),
    (0x6DC7, 0x0A),
    (0x6DC8, 0x12),
    (0x6DC9, 0xC1),
    (0x6DCA, 0xCF),
    (0x6DCB, 0x0A),
    (0x6DCC, 0x1A),
    (0x6DCD, 0xC1),
    (0x6DCE, 0xD1),
    (0x6DCF, 0x0A),
    (0x6DD0, 0x10),
    (0x6DD1, 0xC1),
    (0x6DD2, 0xCA),
    (0x6DD9, 0x3E),
    (0x6DDA, 0x42),
    (0x6DDB, 0x74),
    (0x6DDC, 0x78),
    (0x6DDD, 0x3E),
    (0x6DDE, 0x44),
    (0x6DDF, 0x74),
    (0x6DE0, 0x7C),
    (0x6DE1, 0x3E),
    (0x6DE2, 0x40),
    (0x6DE3, 0x74),
    (0x6DE4, 0x76),
    (0x6DE9, 0x32),
    (0x6DEA, 0x35),
    (0x6DEB, 0x6D),
    (0x6DEC, 0x70),
    (0x6DED, 0xE4),
    (0x6DEE, 0xE6),
    (0x6DEF, 0xF8),
    (0x6DF0, 0xFA),
    (0x6DF1, 0x52),
    (0x6DF2, 0x56),
    (0x6DF3, 0x5C),
    (0x6DF4, 0x5F),
    (0x6DF5, 0x97),
    (0x6DF6, 0xA3),
    (0x6DF7, 0xB4),
    (0x6DF8, 0xBB),
    (0x6DF9, 0x2B),
    (0x6DFA, 0x30),
    (0x6DFB, 0x66),
    (0x6DFC, 0x6B),
    (0x6DFD, 0xDC),
    (0x6DFE, 0xE2),
    (0x6DFF, 0xEE),
    # 6e** block: ISP calibration LUT coefficients (continued)
    (0x6E00, 0xF3),
    (0x6E01, 0x47),
    (0x6E02, 0x49),
    (0x6E03, 0x58),
    (0x6E04, 0x5A),
    (0x6E05, 0x8B),
    (0x6E06, 0x93),
    (0x6E07, 0xA5),
    (0x6E08, 0xAD),
    (0x6E09, 0x5F),
    (0x6E0A, 0x36),
    (0x6E0B, 0xBB),
    (0x6E0C, 0x71),
    (0x6E0F, 0x36),
    (0x6E10, 0x5F),
    (0x6E11, 0x71),
    (0x6E12, 0xBB),
    (0x6E15, 0x5F),
    (0x6E16, 0x36),
    (0x6E17, 0xBB),
    (0x6E18, 0x71),
    (0x6E1B, 0x36),
    (0x6E1C, 0x5F),
    (0x6E1D, 0x71),
    (0x6E1E, 0xBB),
    (0x6E4D, 0x74),
    (0x6E4E, 0x95),
    (0x6E4F, 0xA1),
    (0x6E50, 0xAF),
    (0x6E51, 0xC3),
    (0x6E52, 0xD7),
    (0x6E53, 0xEC),
    (0x6E54, 0xF5),
    (0x6E5B, 0xBB),
    (0x6E5C, 0xC8),
    (0x6E6B, 0x08),
    (0x6E6C, 0x71),
    (0x6E6D, 0x36),
    (0x6E6E, 0x5E),
    (0x6E6F, 0x89),
    (0x6E70, 0xBB),
    (0x6E71, 0x0E),
    (0x6E72, 0x22),
    (0x6E73, 0xBF),
    (0x6E74, 0xCC),
    (0x6E77, 0x0E),
    (0x6E78, 0x22),
    (0x6E79, 0xBF),
    (0x6E7A, 0xCC),
    (0x6E7D, 0x29),
    (0x6E7F, 0xD5),
    (0x6E80, 0x70),
    (0x6E81, 0x29),
    (0x6E82, 0x34),
    (0x6E83, 0x64),
    (0x6E84, 0x70),
    (0x6E89, 0x36),
    (0x6E8A, 0x38),
    (0x6E8B, 0x74),
    (0x6E8C, 0x82),
    (0x6E8F, 0x36),
    (0x6E90, 0x38),
    (0x6E91, 0x74),
    (0x6E92, 0x82),
    (0x6E95, 0x46),
    (0x6E96, 0x5E),
    (0x6E97, 0x89),
    (0x6E98, 0xBB),
    (0x6E9B, 0x5C),
    (0x6E9C, 0x5E),
    (0x6E9D, 0x6D),
    (0x6E9E, 0x6F),
    (0x6E9F, 0x9B),
    (0x6EA0, 0xA1),
    (0x6EA1, 0xF8),
    (0x6EA2, 0xFA),
    (0x6EA3, 0x32),
    (0x6EA4, 0x34),
    (0x6EA5, 0x53),
    (0x6EA6, 0x55),
    (0x6EA7, 0xB6),
    (0x6EA8, 0xBA),
    (0x6EA9, 0xE4),
    (0x6EAA, 0xE6),
    (0x6EC0, 0x60),
    (0x6EC3, 0x72),
    (0x6EC4, 0xBB),
    (0x6EC5, 0x36),
    (0x6EC6, 0x5E),
    (0x6EC8, 0x72),
    (0x6ECB, 0xBB),
    (0x6ED3, 0x62),
    (0x6ED4, 0x9D),
    (0x6ED5, 0x9F),
    (0x6ED6, 0xB8),
    (0x6ED7, 0xBD),
    (0x6ED8, 0x20),
    (0x6ED9, 0xE8),
    (0x6EDA, 0xE0),
    (0x6EEC, 0x09),
    (0x6EED, 0x68),
    (0x6EEF, 0x16),
    (0x6EF0, 0x17),
    (0x6EF2, 0x26),
    (0x6EF3, 0x24),
    (0x6EF5, 0x28),
    (0x6EF6, 0x6C),
    (0x6EF8, 0x2A),
    (0x6EF9, 0x44),
    (0x6EFB, 0x30),
    (0x6EFC, 0x16),
    (0x6EFE, 0x32),
    (0x6EFF, 0x1A),
    # 6f** block: ISP calibration LUT coefficients (continued)
    (0x6F01, 0x34),
    (0x6F02, 0x6C),
    (0x6F04, 0x38),
    (0x6F05, 0x62),
    (0x6F07, 0x3E),
    (0x6F08, 0x40),
    (0x6F0A, 0x40),
    (0x6F0B, 0x2A),
    (0x6F0D, 0x42),
    (0x6F0E, 0x6C),
    (0x6F10, 0x50),
    (0x6F11, 0x2D),
    (0x6F13, 0x56),
    (0x6F14, 0x52),
    (0x6F16, 0x5B),
    (0x6F17, 0x40),
    (0x6F19, 0x5D),
    (0x6F1A, 0x1C),
    (0x6F1C, 0x6D),
    (0x6F1D, 0x41),
    (0x6F1F, 0x89),
    (0x6F20, 0xAB),
    (0x6F22, 0x8B),
    (0x6F23, 0x48),
    (0x6F25, 0xB1),
    (0x6F26, 0x2F),
    (0x6F28, 0xC0),
    (0x6F29, 0x5F),
    (0x6F2B, 0xCD),
    (0x6F2C, 0x15),
    (0x6F2E, 0xD2),
    (0x6F2F, 0x23),
    (0x6F31, 0xD8),
    (0x6F32, 0x2A),
    (0x6F34, 0xDA),
    (0x6F35, 0x55),
    (0x6F45, 0x27),
    (0x6F46, 0x29),
    (0x6F47, 0x33),
    (0x6F48, 0x35),
    (0x6F49, 0x8A),
    (0x6F4A, 0xB0),
    (0x6F4B, 0xD9),
    (0x6F4C, 0xF2),
    (0x6F4D, 0x08),
    (0x6F4E, 0x0A),
    (0x6F4F, 0x41),
    (0x6F50, 0x43),
    (0x6F51, 0x5A),
    (0x6F52, 0x6C),
    (0x6F53, 0xB2),
    (0x6F54, 0xC1),
    (0x6F55, 0x31),
    (0x6F56, 0x47),
    (0x6F59, 0x17),
    (0x6F5A, 0x18),
    (0x6F5B, 0x2C),
    (0x6F5C, 0x2E),
    (0x6F5D, 0x36),
    (0x6F5E, 0x37),
    (0x6F5F, 0x4E),
    (0x6F60, 0x4F),
    (0x6F61, 0x5E),
    (0x6F62, 0x60),
    (0x6F63, 0x8C),
    (0x6F64, 0x8E),
    (0x6F65, 0xCE),
    (0x6F66, 0xD1),
    (0x6F67, 0xDB),
    (0x6F68, 0xDC),
    (0x6F69, 0x43),
    (0x6F6A, 0x4E),
    (0x6F6D, 0x13),
    (0x6F6E, 0x15),
    (0x6F6F, 0x49),
    (0x6F70, 0x4B),
    (0x6F71, 0x86),
    (0x6F72, 0x88),
    (0x6F73, 0xCA),
    (0x6F74, 0xCC),
    (0x6F75, 0x21),
    (0x6F76, 0x25),
    (0x6F77, 0x2B),
    (0x6F78, 0x2F),
    (0x6F79, 0x39),
    (0x6F7A, 0x3D),
    (0x6F7B, 0x51),
    (0x6F7C, 0x55),
    (0x6F7D, 0x82),
    (0x6F7E, 0x86),
    (0x6F7F, 0x99),
    (0x6F80, 0x9D),
    (0x6F81, 0xD3),
    (0x6F82, 0xD7),
    (0x6F83, 0xF0),
    (0x6F84, 0xF4),
    (0x6F85, 0x0D),
    (0x6F86, 0x13),
    (0x6F87, 0x43),
    (0x6F88, 0x49),
    (0x6F89, 0x80),
    (0x6F8A, 0x86),
    (0x6F8B, 0xC4),
    (0x6F8C, 0xCA),
    (0x6F91, 0x3F),
    (0x6F92, 0x7B),
    (0x6F9D, 0x57),
    (0x6F9E, 0x5C),
    (0x6F9F, 0xEC),
    (0x6FA0, 0xF2),
    # HDR combine extended control (0x5DAx range, between combine and PWL blocks)
    (0x5DA1, 0x01),  # COMBINE_0_ISP_CTRL_A1: combine blend control
    (0x5DA3, 0x01),  # COMBINE_0_ISP_CTRL_A3: combine blend control
    (0x5DA5, 0x01),  # COMBINE_0_ISP_CTRL_A5: combine blend control
    (0x5DA7, 0x01),  # COMBINE_0_ISP_CTRL_A7: combine blend control
    (0x5DAD, 0xCA),  # COMBINE_0_ISP_CTRL_AD: combine weight high byte
    (0x5DAE, 0xF5),  # COMBINE_0_ISP_CTRL_AE: combine weight low byte
    # PWL channel 0 (NON_RISC_xx, 0x5Exx): initial base values; overridden by ox08d10_pwl12_blc4.
    # X knee point dx exponents (0x5E01-0x5E1F): segment interval bit-shift exponents.
    # Y knee point dy values (0x5E20-0x5E84): output values, three bytes per knee point.
    (0x5E01, 0x08),  # NON_RISC_01: dx_exp[0] = 8
    (0x5E02, 0x09),  # NON_RISC_02: dx_exp[1] = 9
    (0x5E04, 0x0B),  # NON_RISC_04: dx_exp[3] = 11
    (0x5E05, 0x0C),  # NON_RISC_05: dx_exp[4] = 12
    (0x5E06, 0x0C),  # NON_RISC_06: dx_exp[5] = 12
    (0x5E07, 0x0C),  # NON_RISC_07: dx_exp[6] = 12
    (0x5E0C, 0x0D),  # NON_RISC_0C: dx_exp[11] = 13
    (0x5E0D, 0x0D),  # NON_RISC_0D: dx_exp[12] = 13
    (0x5E0E, 0x0D),  # NON_RISC_0E: dx_exp[13] = 13
    (0x5E0F, 0x0E),  # NON_RISC_0F: dx_exp[14] = 14
    (0x5E10, 0x0E),  # NON_RISC_10: dx_exp[15] = 14
    (0x5E11, 0x0E),  # NON_RISC_11: dx_exp[16] = 14
    (0x5E12, 0x0E),  # NON_RISC_12: dx_exp[17] = 14
    (0x5E13, 0x0F),  # NON_RISC_13: dx_exp[18] = 15
    (0x5E14, 0x0F),  # NON_RISC_14: dx_exp[19] = 15
    (0x5E15, 0x10),  # NON_RISC_15: dx_exp[20] = 16
    (0x5E16, 0x11),  # NON_RISC_16: dx_exp[21] = 17
    (0x5E17, 0x11),  # NON_RISC_17: dx_exp[22] = 17
    (0x5E18, 0x12),  # NON_RISC_18: dx_exp[23] = 18
    (0x5E19, 0x12),  # NON_RISC_19: dx_exp[24] = 18
    (0x5E1B, 0x13),  # NON_RISC_1B: dx_exp[26] = 19
    (0x5E1E, 0x16),  # NON_RISC_1E: dx_exp[29] = 22
    (0x5E1F, 0x16),  # NON_RISC_1F: dx_exp[30] = 22
    (0x5E20, 0x16),  # NON_RISC_20: dy_val[0] byte 2
    (0x5E21, 0x08),  # NON_RISC_21: dy_val[0] byte 1
    (0x5E23, 0x01),  # NON_RISC_23: dy_val[1] byte 2
    (0x5E26, 0x00),  # NON_RISC_26: dy_val[2] byte 2
    (0x5E27, 0x40),  # NON_RISC_27: dy_val[2] byte 1
    (0x5E29, 0x00),  # NON_RISC_29: dy_val[3] byte 2
    (0x5E2A, 0x40),  # NON_RISC_2A: dy_val[3] byte 1
    (0x5E2D, 0x40),  # NON_RISC_2D: dy_val[4] byte 1
    (0x5E30, 0x40),  # NON_RISC_30: dy_val[5] byte 1
    (0x5E33, 0x40),  # NON_RISC_33: dy_val[6] byte 1
    (0x5E36, 0x40),  # NON_RISC_36: dy_val[7] byte 1
    (0x5E39, 0x40),  # NON_RISC_39: dy_val[8] byte 1
    (0x5E3C, 0x40),  # NON_RISC_3C: dy_val[9] byte 1
    (0x5E5A, 0x80),  # NON_RISC_5A: dy_val[19] byte 1
    (0x5E5D, 0x80),  # NON_RISC_5D: dy_val[20] byte 1
    (0x5E5F, 0x01),  # NON_RISC_5F: dy_val[21] byte 2
    (0x5E60, 0x00),  # NON_RISC_60: dy_val[21] byte 1
    (0x5E62, 0x01),  # NON_RISC_62: dy_val[22] byte 2
    (0x5E63, 0x00),  # NON_RISC_63: dy_val[22] byte 1
    (0x5E65, 0x01),  # NON_RISC_65: dy_val[23] byte 2
    (0x5E66, 0x00),  # NON_RISC_66: dy_val[23] byte 1
    (0x5E68, 0x01),  # NON_RISC_68: dy_val[24] byte 2
    (0x5E69, 0x00),  # NON_RISC_69: dy_val[24] byte 1
    (0x5E6B, 0x01),  # NON_RISC_6B: dy_val[25] byte 2
    (0x5E6C, 0x00),  # NON_RISC_6C: dy_val[25] byte 1
    (0x5E6E, 0x01),  # NON_RISC_6E: dy_val[26] byte 2
    (0x5E6F, 0x00),  # NON_RISC_6F: dy_val[26] byte 1
    (0x5E71, 0x01),  # NON_RISC_71: dy_val[27] byte 2
    (0x5E72, 0x00),  # NON_RISC_72: dy_val[27] byte 1
    (0x5E75, 0x80),  # NON_RISC_75: dy_val[28] byte 1
    (0x5E78, 0x80),  # NON_RISC_78: dy_val[29] byte 1
    (0x5E7B, 0x80),  # NON_RISC_7B: dy_val[30] byte 1
    (0x5E7E, 0x80),  # NON_RISC_7E: dy_val[31] byte 1
    (0x5E81, 0x80),  # NON_RISC_81: dy_val[32] byte 1
    (0x5E84, 0x3F),  # NON_RISC_84: dy_val[33] byte 1
    # PWL channel 1 (0x5Fxx): same NON_RISC structure as 0x5Exx for second PWL channel
    (0x5F01, 0x08),  # NON_RISC_01 ch1: dx_exp[0] = 8
    (0x5F02, 0x09),  # NON_RISC_02 ch1: dx_exp[1] = 9
    (0x5F04, 0x0B),  # NON_RISC_04 ch1: dx_exp[3] = 11
    (0x5F05, 0x0C),  # NON_RISC_05 ch1: dx_exp[4] = 12
    (0x5F06, 0x0C),  # NON_RISC_06 ch1: dx_exp[5] = 12
    (0x5F07, 0x0C),  # NON_RISC_07 ch1: dx_exp[6] = 12
    (0x5F0C, 0x0D),  # NON_RISC_0C ch1: dx_exp[11] = 13
    (0x5F0D, 0x0D),  # NON_RISC_0D ch1: dx_exp[12] = 13
    (0x5F0E, 0x0D),  # NON_RISC_0E ch1: dx_exp[13] = 13
    (0x5F0F, 0x0E),  # NON_RISC_0F ch1: dx_exp[14] = 14
    (0x5F10, 0x0E),  # NON_RISC_10 ch1: dx_exp[15] = 14
    (0x5F11, 0x0E),  # NON_RISC_11 ch1: dx_exp[16] = 14
    (0x5F12, 0x0E),  # NON_RISC_12 ch1: dx_exp[17] = 14
    (0x5F13, 0x0F),  # NON_RISC_13 ch1: dx_exp[18] = 15
    (0x5F14, 0x0F),  # NON_RISC_14 ch1: dx_exp[19] = 15
    (0x5F15, 0x10),  # NON_RISC_15 ch1: dx_exp[20] = 16
    (0x5F16, 0x11),  # NON_RISC_16 ch1: dx_exp[21] = 17
    (0x5F17, 0x11),  # NON_RISC_17 ch1: dx_exp[22] = 17
    (0x5F18, 0x12),  # NON_RISC_18 ch1: dx_exp[23] = 18
    (0x5F19, 0x12),  # NON_RISC_19 ch1: dx_exp[24] = 18
    (0x5F1B, 0x13),  # NON_RISC_1B ch1: dx_exp[26] = 19
    (0x5F1E, 0x16),  # NON_RISC_1E ch1: dx_exp[29] = 22
    (0x5F1F, 0x16),  # NON_RISC_1F ch1: dx_exp[30] = 22
    (0x5F20, 0x16),  # NON_RISC_20 ch1: dy_val[0] byte 2
    (0x5F21, 0x08),  # NON_RISC_21 ch1: dy_val[0] byte 1
    (0x360A, 0x03),  # EXT_SPARE_REG10: external spare register 10
    (0x360B, 0x14),  # EXT_SPARE_REG11: external spare register 11
    (0x322D, 0xE0),  # ROC_CRC_REF_3: read-out channel CRC reference bits [31:24]
    (0x322E, 0xE4),  # ROC_CRC_REF_2: read-out channel CRC reference bits [23:16]
    (0x322F, 0xC8),  # ROC_CRC_REF_1: read-out channel CRC reference bits [15:8]
    (0x3230, 0x6D),  # ROC_CRC_REF_0: read-out channel CRC reference bits [7:0]
    # Tail: group-hold release, HDR combine table update, ISP enable
    (0x0104, 0x00),  # GROUP_ACCESS: end group hold (release deferred register writes)
    (
        0x3B81,
        0x2D,
    ),  # PIXASIL_ROI_FAILED_NR_TH_H2: pixel ASIL ROI failed count threshold high
    (
        0x3B84,
        0x2D,
    ),  # PIXASIL_ROI_FAILED_NR_TH_L: pixel ASIL ROI failed count threshold low
    # Updated HDR combine thresholds for channel 0
    (
        0x5BBD,
        0x33,
    ),  # COMBINE_0_ISP_CTRL_3D: HDR combine ratio error and stability control
    (
        0x5BAE,
        0x02,
    ),  # cmb_thre_s0 high: knee point threshold 0 for short-exposure weight, high byte
    (
        0x5BAF,
        0x80,
    ),  # cmb_thre_s0 low: knee point threshold 0 for short-exposure weight, low byte
    (
        0x5BB0,
        0x0C,
    ),  # cmb_thre_s1 high: knee point threshold 1 for short-exposure weight, high byte
    (
        0x5BB1,
        0x00,
    ),  # cmb_thre_s1 low: knee point threshold 1 for short-exposure weight, low byte
    (
        0x5BB2,
        0x0E,
    ),  # cmb_thre_s2 high: knee point threshold 2 for short-exposure weight, high byte
    (
        0x5BB3,
        0x00,
    ),  # cmb_thre_s2 low: knee point threshold 2 for short-exposure weight, low byte
    (
        0x5BB4,
        0x0F,
    ),  # cmb_thre_s3 high: knee point threshold 3 for short-exposure weight, high byte
    (
        0x5BB5,
        0xFF,
    ),  # cmb_thre_s3 low: knee point threshold 3 for short-exposure weight, low byte
    # HDR combine control for channels 1 and 2
    (0x5C3D, 0x0B),  # COMBINE_1_ISP_CTRL_3D: HDR combine channel 1 ratio control
    (0x5C21, 0x40),  # COMBINE_1_ISP_CTRL_21: HDR combine channel 1 weight
    (0x5CBD, 0x09),  # COMBINE_2_ISP_CTRL_3D: HDR combine channel 2 ratio control
    (0x5CA1, 0x40),  # COMBINE_2_ISP_CTRL_21: HDR combine channel 2 weight
    (0x5BBE, 0xA0),  # COMBINE_0_ISP_CTRL_41: combine threshold shift and input option
    (0x5BCA, 0x00),  # COMBINE_0_ISP_CTRL_4A: long exposure lower threshold
    (0x5BCB, 0x40),  # COMBINE_0_ISP_CTRL_4B: combine lower threshold
    # ISP calibration LUT (0x63xx range)
    (0x639B, 0x22),  # ISP calibration LUT entry at 0x639B
    (0x639D, 0x2C),  # ISP calibration LUT entry at 0x639D
    (0x639F, 0x34),  # ISP calibration LUT entry at 0x639F
    (0x63A1, 0x3A),  # ISP calibration LUT entry at 0x63A1
    (0x63A3, 0x40),  # ISP calibration LUT entry at 0x63A3
    (0x63A5, 0x46),  # ISP calibration LUT entry at 0x63A5
    (0x63A7, 0x48),  # ISP calibration LUT entry at 0x63A7
    (0x63A9, 0x4A),  # ISP calibration LUT entry at 0x63A9
    (0x63CC, 0x14),  # ISP calibration LUT entry at 0x63CC
    (0x63CD, 0x14),  # ISP calibration LUT entry at 0x63CD
    (0x63CE, 0x14),  # ISP calibration LUT entry at 0x63CE
    (0x63CF, 0x14),  # ISP calibration LUT entry at 0x63CF
    (0x63D0, 0x14),  # ISP calibration LUT entry at 0x63D0
    (0x63D1, 0x14),  # ISP calibration LUT entry at 0x63D1
    (0x63D2, 0x14),  # ISP calibration LUT entry at 0x63D2
    (0x63D3, 0x14),  # ISP calibration LUT entry at 0x63D3
    (0x63D4, 0x02),  # ISP calibration LUT entry at 0x63D4
    (0x63DC, 0x0C),  # ISP calibration LUT entry at 0x63DC
    (0x63DD, 0x0A),  # ISP calibration LUT entry at 0x63DD
    # ISP calibration LUT (0x66xx range)
    (0x66CD, 0x01),  # ISP calibration LUT entry at 0x66CD
    (0x66CE, 0x01),  # ISP calibration LUT entry at 0x66CE
    (0x66CF, 0x01),  # ISP calibration LUT entry at 0x66CF
    (0x66D0, 0x01),  # ISP calibration LUT entry at 0x66D0
    (0x66D1, 0x01),  # ISP calibration LUT entry at 0x66D1
    (0x66D2, 0x11),  # ISP calibration LUT entry at 0x66D2
    (0x66D3, 0x11),  # ISP calibration LUT entry at 0x66D3
    (0x66D5, 0x01),  # ISP calibration LUT entry at 0x66D5
    (0x66D6, 0x01),  # ISP calibration LUT entry at 0x66D6
    (0x66D7, 0x01),  # ISP calibration LUT entry at 0x66D7
    (0x66D8, 0x01),  # ISP calibration LUT entry at 0x66D8
    (0x66D9, 0x11),  # ISP calibration LUT entry at 0x66D9
    (0x66DA, 0x11),  # ISP calibration LUT entry at 0x66DA
    (0x66DB, 0x11),  # ISP calibration LUT entry at 0x66DB
    (0x66DC, 0x09),  # ISP calibration LUT entry at 0x66DC
    (0x66DD, 0x09),  # ISP calibration LUT entry at 0x66DD
    (0x66DE, 0x09),  # ISP calibration LUT entry at 0x66DE
    (0x66DF, 0x09),  # ISP calibration LUT entry at 0x66DF
    (0x66E0, 0x09),  # ISP calibration LUT entry at 0x66E0
    (0x66E1, 0x09),  # ISP calibration LUT entry at 0x66E1
    (0x66E2, 0x09),  # ISP calibration LUT entry at 0x66E2
    (0x66E3, 0x09),  # ISP calibration LUT entry at 0x66E3
    (0x66E4, 0x09),  # ISP calibration LUT entry at 0x66E4
    (0x66E5, 0x09),  # ISP calibration LUT entry at 0x66E5
    (0x66E6, 0x09),  # ISP calibration LUT entry at 0x66E6
    (0x66E7, 0x09),  # ISP calibration LUT entry at 0x66E7
    (0x66E8, 0x09),  # ISP calibration LUT entry at 0x66E8
    (0x66E9, 0x09),  # ISP calibration LUT entry at 0x66E9
    (0x66EA, 0x09),  # ISP calibration LUT entry at 0x66EA
    (0x66EB, 0x09),  # ISP calibration LUT entry at 0x66EB
    (0x480B, 0x02),  # MIPI_LANE_SWAP_0: MIPI lane swap configuration
    (0x480F, 0x13),  # MIPI_LANE_SWAP_1: MIPI lane swap configuration
    # ISP pipeline enable for each HDR exposure channel
    (
        0x5004,
        0x1E,
    ),  # ISP_CTRL_04: HCG channel enable (DPC, OTP-DPC, WB gain, pre-ISP, SCIP)
    (0x5005, 0x1E),  # ISP_CTRL_05: LCG channel enable (same as HCG)
    (0x5006, 0xFE),  # ISP_CTRL_06: LOFIC channel enable (all ISP features)
    (0x5007, 0xFE),  # ISP_CTRL_07: VS channel enable (all ISP features)
    # TPG (test pattern generator) disable: 0x00 = normal image output
    (0x5240, 0x00),  # TPG_CTRL_HCG: disable test pattern for HCG channel
    (0x5440, 0x00),  # TPG_CTRL_LCG: disable test pattern for LCG channel
    (0x5640, 0x00),  # TPG_CTRL_LOFIC: disable test pattern for LOFIC channel
    (0x5840, 0x00),  # TPG_CTRL_VS: disable test pattern for VS channel
]
