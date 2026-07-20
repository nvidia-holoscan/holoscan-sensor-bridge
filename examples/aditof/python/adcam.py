#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved.
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

import ctypes  # Added for FW Update as this needs structure format handling similar to C/C++

# More details about the chip and eval kit can be found on https://www.analog.com/en/products/adtf3175.html
import logging
import math
import struct
import time
from collections import namedtuple

import hololink as hololink_module

# Firmware update related changes
# Constants from Adsd3500.cpp for Firmware update
FLASH_PAGE_SIZE = 256
WRITE_MASTER_FIRMWARE_COMMAND = 0x04
WRITE_SLAVE_FIRMWARE_COMMAND = 0x2A
GET_MASTER_FIRMWARE_COMMAND = 0x01
GET_SLAVE_FIRMWARE_COMMAND = 0x04
ADSD3500_CMD_GET_STATUS = 0x0020
RESET_ADSD3500_CMD = 0x00240000
GET_MASTER_CHIP_ID_CMD = 0x0112
GET_SLAVE_CHIP_ID_CMD = 0x0116
SET_SWITCH_TO_BURST_MODE = 0x0019
STREAM_ON_CMD = 0x00AD
STREAM_ON_VAL = 0x00C5
STREAM_OFF_CMD = 0x000C
STREAM_OFF_VAL = 0x0002
ENABLE_VAL = 0x0001
SET_FRAME_RATE_REG = 0x0022
MIPI_CLK_CONTINUOUS_CMD = 0x00A9
MIPI_OUTPUT_SPEED_CMD = 0x0031
ADSD3500_CMD_GET_CHIP_INFO = 0x0032
DESKEW_ENABLE_CMD = 0x00AB
GET_IMAGER_ERROR_CMD = 0x0038
GET_MIPI_CLK_CONTINUOUS_CMD = 0x00AA
MIPI_SPEED_1_5_GBPS = 0x0003
MIPI_SPEED_1GBPS = 0x0004
LOW_BYTE_MASK = 0x00FF
GET_DUAL_ADSD3500_ENABLED_CMD = 0x005A
ADI_STATUS_FIRMWARE_UPDATE = 0x000E
ADI_STATUS_SECOND_FIRMWARE_FLASH_UPDATE = 0x0027
ADI_ROM_CFG_CRC_SEED_VALUE = 0xFFFFFFFF
ADI_ROM_CFG_CRC_POLYNOMIAL = 0x04C11DB7
ADI_DUAL_FW_SLOT_SIZE = 0x20000  # 128 KB per slot
ADI_CHUNK_HEADER_SIZE = 20  # ADI chunk header size in bytes
FW_MIN_VERSION = (8, 1, 0, 0)  # Minimum supported firmware version

# ---------------------------------------------------------------------------
# Imager type codes returned by register 0x0032 (ADSD3500_CMD_GET_CHIP_INFO)
# resp[0] (bits [15:8]) = Imager Type, resp[1] (bits [7:0]) = CCB Version
# ---------------------------------------------------------------------------
ADCAM_IMAGER_TYPE_ADSD3100 = 1
ADCAM_IMAGER_TYPE_ADTF3066 = 2

# Converted from uint32_t const crc32_table[256]. Needed for Firmware update
crc32_table = (
    0,
    79764919,
    159529838,
    222504665,
    319059676,
    398814059,
    445009330,
    507990021,
    638119352,
    583659535,
    797628118,
    726387553,
    890018660,
    835552979,
    1015980042,
    944750013,
    1276238704,
    1221641927,
    1167319070,
    1095957929,
    1595256236,
    1540665371,
    1452775106,
    1381403509,
    1780037320,
    1859660671,
    1671105958,
    1733955601,
    2031960084,
    2111593891,
    1889500026,
    1952343757,
    2552477408,
    2632100695,
    2443283854,
    2506133561,
    2334638140,
    2414271883,
    2191915858,
    2254759653,
    3190512472,
    3135915759,
    3081330742,
    3009969537,
    2905550212,
    2850959411,
    2762807018,
    2691435357,
    3560074640,
    3505614887,
    3719321342,
    3648080713,
    3342211916,
    3287746299,
    3467911202,
    3396681109,
    4063920168,
    4143685023,
    4223187782,
    4286162673,
    3779000052,
    3858754371,
    3904687514,
    3967668269,
    881225847,
    809987520,
    1023691545,
    969234094,
    662832811,
    591600412,
    771767749,
    717299826,
    311336399,
    374308984,
    453813921,
    533576470,
    25881363,
    88864420,
    134795389,
    214552010,
    2023205639,
    2086057648,
    1897238633,
    1976864222,
    1804852699,
    1867694188,
    1645340341,
    1724971778,
    1587496639,
    1516133128,
    1461550545,
    1406951526,
    1302016099,
    1230646740,
    1142491917,
    1087903418,
    2896545431,
    2825181984,
    2770861561,
    2716262478,
    3215044683,
    3143675388,
    3055782693,
    3001194130,
    2326604591,
    2389456536,
    2200899649,
    2280525302,
    2578013683,
    2640855108,
    2418763421,
    2498394922,
    3769900519,
    3832873040,
    3912640137,
    3992402750,
    4088425275,
    4151408268,
    4197601365,
    4277358050,
    3334271071,
    3263032808,
    3476998961,
    3422541446,
    3585640067,
    3514407732,
    3694837229,
    3640369242,
    1762451694,
    1842216281,
    1619975040,
    1682949687,
    2047383090,
    2127137669,
    1938468188,
    2001449195,
    1325665622,
    1271206113,
    1183200824,
    1111960463,
    1543535498,
    1489069629,
    1434599652,
    1363369299,
    622672798,
    568075817,
    748617968,
    677256519,
    907627842,
    853037301,
    1067152940,
    995781531,
    51762726,
    131386257,
    177728840,
    240578815,
    269590778,
    349224269,
    429104020,
    491947555,
    4046411278,
    4126034873,
    4172115296,
    4234965207,
    3794477266,
    3874110821,
    3953728444,
    4016571915,
    3609705398,
    3555108353,
    3735388376,
    3664026991,
    3290680682,
    3236090077,
    3449943556,
    3378572211,
    3174993278,
    3120533705,
    3032266256,
    2961025959,
    2923101090,
    2868635157,
    2813903052,
    2742672763,
    2604032198,
    2683796849,
    2461293480,
    2524268063,
    2284983834,
    2364738477,
    2175806836,
    2238787779,
    1569362073,
    1498123566,
    1409854455,
    1355396672,
    1317987909,
    1246755826,
    1192025387,
    1137557660,
    2072149281,
    2135122070,
    1912620623,
    1992383480,
    1753615357,
    1816598090,
    1627664531,
    1707420964,
    295390185,
    358241886,
    404320391,
    483945776,
    43990325,
    106832002,
    186451547,
    266083308,
    932423249,
    861060070,
    1041341759,
    986742920,
    613929101,
    542559546,
    756411363,
    701822548,
    3316196985,
    3244833742,
    3425377559,
    3370778784,
    3601682597,
    3530312978,
    3744426955,
    3689838204,
    3819031489,
    3881883254,
    3928223919,
    4007849240,
    4037393693,
    4100235434,
    4180117107,
    4259748804,
    2310601993,
    2373574846,
    2151335527,
    2231098320,
    2596047829,
    2659030626,
    2470359227,
    2550115596,
    2947551409,
    2876312838,
    2788305887,
    2733848168,
    3165939309,
    3094707162,
    3040238851,
    2985771188,
)


# 1. Define the Packed Struct
# The __attribute__((__packed__)) in C++ is handled by '_pack_ = 1'
# class CmdHeaderStruct(ctypes.LittleEndianStructure):
class CmdHeaderStruct(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("id8", ctypes.c_uint8),  # 0xAD
        ("chunk_size16", ctypes.c_uint16),  # 256 is flash page size
        ("cmd8", ctypes.c_uint8),  # CMD for fw upgrade
        ("total_size_fw32", ctypes.c_uint32),  # total size of firmware
        ("header_checksum32", ctypes.c_uint32),  # header checksum
        ("crc_of_fw32", ctypes.c_uint32),  # CRC of the Firmware Binary
    ]


# 2. Define the Union
class CmdHeaderUnion(ctypes.Union):
    _fields_ = [
        ("cmd_header_byte", ctypes.c_uint8 * 16),  # 16 byte array
        ("fields", CmdHeaderStruct),  # The struct defined above
    ]


# From compute_crc.h
IS_CRC_MIRROR = 1 << 0  #


class CRC_TYPE:  #
    CRC_8bit = 8
    CRC_16bit = 16
    CRC_32bit = 32


class CrcOutputUnion(ctypes.Union):  #
    _fields_ = [
        ("crc_8bit", ctypes.c_uint8),
        ("crc_16bit", ctypes.c_uint16),
        ("crc_32bit", ctypes.c_uint32),
    ]


class CrcParametersUnion(ctypes.Structure):  #
    _fields_ = [
        ("type", ctypes.c_int),  # CRC_TYPE enum
        ("polynomial", CrcOutputUnion),  # Nested Union
        ("initial_crc", CrcOutputUnion),  # Nested Union
        ("crc_compute_flags", ctypes.c_uint32),
    ]


# This is the structure for FW data chunk of 256 bytes
class FwUpdateBinData(ctypes.BigEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("raw_data", ctypes.c_uint8 * 256),  # 256 bytes of FW data chunk
    ]


def generate_mirror(value):
    """
    Replicates the generate_mirror logic from compute_crc.c
    Reflects the bits of an 8-bit byte.
    """
    mirror_value = 0
    for i in range(8):
        if (value >> i) & 1:
            mirror_value |= 1 << (7 - i)
    return mirror_value


def compute_crc_python(crc_parameters, data):
    """
    Python equivalent of compute_crc(crc_parameters_t *crc_parameters, ...)
    """
    # Initialize temp_value with the initial_crc (equivalent to memcpy)
    # Accessing the .crc_32bit field from our previously defined ctypes union
    temp_value_crc32 = crc_parameters.initial_crc.crc_32bit
    # Check if we are doing 32-bit CRC
    if crc_parameters.type == 32:  # CRC_32bit = 32
        # In Python, we just use the tuple/list we defined earlier as the table
        # CRC32_TABLE is the tuple you converted in the previous step
        for byte in data:
            if crc_parameters.crc_compute_flags & 1:  # IS_CRC_MIRROR = 1
                # 1. Mirror the input byte
                mirrored_byte = generate_mirror(byte)
                # 2. Calculate the index: mirrored_byte ^ (top byte of current CRC)
                # (temp_value.crc_32bit >> 0x18) & 0xFF extracts the most significant byte
                index = (mirrored_byte ^ (temp_value_crc32 >> 24)) & 0xFF
                # 3. Update CRC: Table[index] ^ (Current CRC shifted left by 8)
                # We MUST mask with 0xFFFFFFFF to keep it 32-bit
                temp_value_crc32 = (
                    crc32_table[index] ^ (temp_value_crc32 << 8)
                ) & 0xFFFFFFFF
    return temp_value_crc32


# Till here : Firmware update related changes

# ---------------------------------------------------------------------------
# Mode configuration tables — Python mirror of adcam_lib.hpp
# ---------------------------------------------------------------------------
#   Fields: mode_number, mipi_w, mipi_h, px_w, px_h,
#           phase_depth_bits, ab_bits, confidence_bits,
#           ab_averaging, depth_enable, output_mipi
AdcamModeConfig = namedtuple(
    "AdcamModeConfig",
    [
        "mode_number",
        "width",
        "height",
        "pixel_width",
        "pixel_height",
        "phase_depth_bits",
        "ab_bits",
        "confidence_bits",
        "ab_averaging",
        "depth_enable",
        "output_mipi",
    ],
)

# ADSD3100: MP (1024×1024, 2 Gbps) and QMP (512×512, 1 Gbps)
ADSD3100_STANDARD_MODES = [
    # ---- MP modes (1024×1024, 2 Gbps MIPI) ----
    AdcamModeConfig(0, 3072, 1707, 1024, 1024, 6, 6, 2, 0, 1, 2),
    AdcamModeConfig(1, 3072, 1707, 1024, 1024, 6, 6, 2, 0, 1, 2),
    # ---- QMP modes (512×512, 1 Gbps MIPI) ----
    AdcamModeConfig(2, 2560, 512, 512, 512, 6, 6, 2, 1, 1, 2),
    AdcamModeConfig(3, 2560, 512, 512, 512, 6, 6, 2, 1, 1, 2),
    AdcamModeConfig(5, 2560, 512, 512, 512, 6, 6, 2, 1, 1, 2),
    AdcamModeConfig(6, 2560, 512, 512, 512, 6, 6, 2, 1, 1, 2),
]

# ADTF3066: VGA (512×640, 1 Gbps) and QVGA (256×320, 1 Gbps)
ADTF3066_STANDARD_MODES = [
    # ---- VGA modes (512×640, 1 Gbps MIPI) ? modes 0,1,7 ----
    AdcamModeConfig(0, 2560, 640, 512, 640, 6, 6, 2, 1, 1, 2),
    AdcamModeConfig(1, 2560, 640, 512, 640, 6, 6, 2, 1, 1, 2),
    AdcamModeConfig(7, 2560, 640, 512, 640, 6, 6, 2, 1, 1, 2),
    # ---- QVGA modes (256×320, 1 Gbps MIPI) ? modes 3,6,8 ----
    AdcamModeConfig(3, 1280, 320, 256, 320, 6, 6, 2, 1, 1, 2),
    AdcamModeConfig(6, 1280, 320, 256, 320, 6, 6, 2, 1, 1, 2),
    AdcamModeConfig(8, 1280, 320, 256, 320, 6, 6, 2, 1, 1, 2),
]


def adcam_find_mode(table, mode_number):
    """Search a mode-config table by mode_number. Returns None if not found."""
    for cfg in table:
        if cfg.mode_number == mode_number:
            return cfg
    return None


def adcam_make_mode_settings(cfg):
    """Build the 16-bit Set Imager Mode word (Word 2) from an AdcamModeConfig.

    Bit layout (mirrors adcam_lib.hpp adcam_make_mode_settings):
      Bit  0      : depth_enable
      Bit  1      : data_interleaving (always 1)
      Bit  2      : ab_enable (always 1)
      Bit  3      : ab_averaging
      Bits [6:4]  : phase_depth_bits encoded as (6 - value)
      Bits [9:7]  : ab_bits encoded as (6 - value)
      Bits [11:10]: confidence_bits
      Bits [13:12]: output_mipi
    """
    w = (cfg.depth_enable & 0x1) << 0
    w |= 1 << 1  # data_interleaving
    w |= 1 << 2  # ab_enable
    w |= (cfg.ab_averaging & 0x1) << 3
    w |= ((6 - cfg.phase_depth_bits) & 0x7) << 4
    w |= ((6 - cfg.ab_bits) & 0x7) << 7
    w |= (cfg.confidence_bits & 0x3) << 10
    w |= (cfg.output_mipi & 0x3) << 12
    return w & 0xFFFF


class ADCAMEXPANDER:
    EXPANDER_0_I2C_BUS_ADDRESS = 0x68

    def __init__(self, hololink_channel, hololink_i2c_controller_address, expan_addr):
        # Get handles to these controllers but don't actually talk to them yet
        self._hololink = hololink_channel.hololink()
        self._i2c = self._hololink.get_i2c(hololink_i2c_controller_address)
        self._expan_addr = expan_addr

    def set_register(self, register, value, timeout=None):
        logging.debug(
            "set_register(register=%d(0x%X), value=%d(0x%X))"
            % (register, register, value, value)
        )
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(register)
        serializer.append_uint8(value)
        read_byte_count = 0
        self._i2c.i2c_transaction(
            self._expan_addr,
            write_bytes[: serializer.length()],
            read_byte_count,
            timeout=timeout,
        )
        # print("Expander", self._expan_addr, "byte_count = ", read_byte_count)


class polarfireGpio:

    # configurations to test
    configs = [
        "ALL_OUT_L",  # All pins output low
        "ALL_OUT_H",  # All pins output high
        "ALL_IN",  # All pins inputs
        "ODD_OUT_H",  # Odd pins output high, even pins input - use jumper to short & test
        "EVEN_OUT_H",  # Even pins output high, Odd pins input - use jumper to short & test
    ]

    # dictionary for print beautification
    dir = {0: "output", 1: "input"}

    # def __init__(self, fragment, hololink_channel, gpio, *args, **kwargs):
    def __init__(self, hololink_channel, gpio, pin_num):
        self._hololink = hololink_channel.hololink()
        self._gpio = gpio
        self._pin = pin_num
        self.test_config = 0

        # how many pins are supported on the platform running the example
        self._supported_pins_number = self._gpio.get_supported_pin_num()

        logging.info(f"Totel supported GPIOs = {self._supported_pins_number}")

    def setup(self):
        # spec.output("gpio_changed_out")
        # spec.output("test_config_out")

        # set all gpios as output, high - test fast setting via loop
        for i in range(self._supported_pins_number):
            logging.debug(f"GPIO left as is = {i}")

    def pull_gpio_low(self):
        self._gpio.set_direction(self._pin, self._gpio.OUT)
        self._gpio.set_value(self._pin, self._gpio.LOW)

    def pull_gpio_high(self):
        self._gpio.set_direction(self._pin, self._gpio.OUT)
        self._gpio.set_value(self._pin, self._gpio.HIGH)


# class adtf3175:
class adcam:
    ADCAM_I2C_BUS_ADDRESS = 0x38
    EXPANDER_0_I2C_BUS_ADDRESS = 0x68
    EXPANDER_1_I2C_BUS_ADDRESS = 0x58

    def __init__(
        self,
        hololink_channel,
        hololink_i2c_controller_address,
        channel_metadata,
        adcam_mode,
        reset_pin,
        num_planes,
        tof_fps,
        metadata_sz,
        mipi_lane_speed,
    ):
        # Get handles to these controllers but don't actually talk to them yet
        self._hololink = hololink_channel.hololink()
        self._i2c = self._hololink.get_i2c(hololink_i2c_controller_address)
        self._adcam_mode = adcam_mode
        self._reset_pin = reset_pin
        self._num_planes = num_planes
        self._tof_fps = tof_fps
        self._metadata_sz = metadata_sz
        self._mipi_lane_speed = mipi_lane_speed
        self._imager_type = 0  # set by get_imager_type_and_ccb_version() after start()
        self._byteperpixel = 1
        self._pixel_format = hololink_module.sensors.csi.PixelFormat.RAW_8

        # Best-effort geometry init from mode tables before imager type is known.
        # Search ADSD3100 first then ADTF3066; call get_imager_type_and_ccb_version()
        # after hololink.start() to pick the correct table.
        init_cfg = adcam_find_mode(ADSD3100_STANDARD_MODES, adcam_mode)
        if init_cfg is None:
            init_cfg = adcam_find_mode(ADTF3066_STANDARD_MODES, adcam_mode)
        if init_cfg is None:
            raise RuntimeError(f"adcam: unsupported adcam_mode {adcam_mode}")
        self._width = init_cfg.width
        self._height = init_cfg.height
        self._pixel_width = init_cfg.pixel_width
        self._pixel_height = init_cfg.pixel_height

        self._expander0 = ADCAMEXPANDER(
            hololink_channel,
            hololink_module.CAM_I2C_BUS,
            self.EXPANDER_0_I2C_BUS_ADDRESS,
        )
        self._expander1 = ADCAMEXPANDER(
            hololink_channel,
            hololink_module.CAM_I2C_BUS,
            self.EXPANDER_1_I2C_BUS_ADDRESS,
        )
        self._pf_gpio = polarfireGpio(
            hololink_channel, self._hololink.get_gpio(channel_metadata), self._reset_pin
        )

    def get_version(self):
        logging.debug("Fatching Chip version")
        REGISTER = GET_MASTER_CHIP_ID_CMD
        resp = self.set_register16_response(REGISTER, 2)
        logging.debug(f"Response = {resp}")

        # Fetch status
        REGISTER = ADSD3500_CMD_GET_STATUS
        resp = self.set_register16_response(REGISTER, 2)
        logging.debug(f"Response = {resp}")

        return resp

    def bytes_to_uint16_array(self, data: bytes):
        """Convert bytes/bytearray to list of 16-bit unsigned integers (big-endian)."""
        if len(data) % 2 != 0:
            raise ValueError("Data length must be even for 16-bit conversion.")
        return list(struct.unpack(f">{len(data)//2}H", data))

    def registers_to_byte_array(self, register):
        """Convert bytes/bytearray to list of 16-bit unsigned integers (big-endian)."""
        length = (register.bit_length() + 7) // 8

        if length == 0:
            length = 2
        if length % 2 != 0:
            length = length + 1

        byte_array = register.to_bytes(length, byteorder="big")
        return byte_array

    def format_registers(self, reg):
        """Convert odd size register to uint16 list (big-endian)."""
        return self.bytes_to_uint16_array(self.registers_to_byte_array(reg))

    def set_mipi(self):
        """Configure MIPI lane speed and enable deskew.

        Mirrors Adcam::set_mipi() in adcam_lib.cpp:
          1. Read current chip status (GET_IMAGER_STATUS_CMD = 0x0020)
          2. Set MIPI output speed to 1 Gbps (MIPI_OUTPUT_SPEED_CMD = 0x0031, value = 0x0004)
          3. Read status again (get_status)
          4. Enable deskew (DESKEW_ENABLE_CMD = 0x00AB, value = 0x0001)
        """
        logging.debug(f"Setting MIPI speed width={self._width}")

        # 1. Status check before changing link settings
        resp = self.set_register16_response(ADSD3500_CMD_GET_STATUS, 2)
        logging.debug(f"Chip Status before set_mipi = {resp}")

        # 2. Set MIPI output speed
        self.set_register16_no_response(
            (MIPI_OUTPUT_SPEED_CMD << 16) | MIPI_SPEED_1GBPS  # self._mipi_lane_speed
        )

        # 3. Status check after speed change
        self.get_status()

        # 4. Enable deskew
        logging.debug("Enabling deskew")
        self.set_register16_no_response((DESKEW_ENABLE_CMD << 16) | ENABLE_VAL)

        # set FPS
        logging.debug("Setting FPS")
        self.set_register16_no_response((SET_FRAME_RATE_REG << 16) | self._tof_fps)

    def set_mode(self):
        """Dynamically build and send the Set Imager Mode command.

        Word 1 (0xDAXX): XX = capture mode number
        Word 2 (0xYYYY): built from AdcamModeConfig via adcam_make_mode_settings()
        Mirrors Adcam::set_mode() in adcam_lib.cpp.
        """
        imager_str = {
            ADCAM_IMAGER_TYPE_ADSD3100: "ADSD3100",
            ADCAM_IMAGER_TYPE_ADTF3066: "ADTF3066",
        }.get(self._imager_type, "Unknown")

        if self._imager_type == ADCAM_IMAGER_TYPE_ADSD3100:
            cfg = adcam_find_mode(ADSD3100_STANDARD_MODES, self._adcam_mode)
        elif self._imager_type == ADCAM_IMAGER_TYPE_ADTF3066:
            cfg = adcam_find_mode(ADTF3066_STANDARD_MODES, self._adcam_mode)
        else:
            # Imager type not yet detected — try both tables
            logging.warning(
                f"set_mode: imager type not yet detected (raw={self._imager_type}), "
                "trying ADSD3100 then ADTF3066"
            )
            cfg = adcam_find_mode(ADSD3100_STANDARD_MODES, self._adcam_mode)
            if cfg is None:
                cfg = adcam_find_mode(ADTF3066_STANDARD_MODES, self._adcam_mode)

        if cfg is None:
            raise RuntimeError(
                f"set_mode: mode {self._adcam_mode} not found for imager "
                f"{imager_str} (raw={self._imager_type})"
            )

        mode_reg = 0xDA00 | (self._adcam_mode & 0xFF)
        mode_setting = adcam_make_mode_settings(cfg)
        REGISTER = (mode_reg << 16) | mode_setting
        logging.info(
            f"Setting imager mode={self._adcam_mode} "
            f"reg=0x{mode_reg:04X} settings=0x{mode_setting:04X}"
        )
        self.set_register16_no_response(REGISTER)

    def set_mode_for_slave(self):
        logging.debug("Setting mode for Slave pulsatrix")
        print("Setting mode for Slave pulsatrix")
        REGISTER = 0xDA01280F  # Set mode to 0x01
        self.set_register16_no_response(REGISTER)

    def set_slave_threshold(self, RadialValue):
        REGISTER = 0x0027  # Slave Radial value write register
        REGISTER = REGISTER << 16 | RadialValue
        logging.debug(f"Setting radial threshold for Slave pulsatrix to 0x{REGISTER:X}")
        print(f"Setting radial threshold for Slave pulsatrix to 0x{REGISTER:X}")
        self.set_register16_no_response(REGISTER)

    def get_slave_threshold(self):
        logging.debug("Getting radial threshold for Slave pulsatrix")
        print("Getting radial threshold for Slave pulsatrix")
        REGISTER = 0x0028  # Slave Radial value read register
        resp = self.set_register16_response(REGISTER, 2)
        logging.debug(f"Response = {resp}")
        print(f"Slave reResponse = {resp}")
        return resp

    def softreset(self):
        print("Softresetting the chip")
        logging.info("Softresetting the chip")
        REGISTER = RESET_ADSD3500_CMD
        self.set_register16_no_response(REGISTER)

    def read_nvm_config(self):
        logging.debug("Reading NVM Config")
        time.sleep(1)
        REGISTER = SET_SWITCH_TO_BURST_MODE << 16
        self.set_register16_no_response(REGISTER)

        # Read fw version
        REGISTER = 0xAD002C05000000003100000001000000
        resp = self.set_register16_response(REGISTER, 44)
        logging.info(f"Firmware ID = {resp}")

        # Read NVM header
        REGISTER = 0xAD000013000000001300000001000000
        resp = self.set_register16_response(REGISTER, 44)
        logging.debug(f"NVM Header = {resp}")

        # Read fw version
        REGISTER = 0xAD002C05000000003100000001000000
        resp = self.set_register16_response(REGISTER, 44)
        logging.debug(f"Firmware ID = {resp}")

        # turn off burst mode
        REGISTER = 0xAD000010000000001000000000000000
        REGISTER = 0xAD000010000000001000000001000000
        self.set_register16_no_response(REGISTER)
        logging.debug("Reading NVM Config done")

    def get_status(self):
        logging.debug("Fetching status")
        # Get Status
        REGISTER = ADSD3500_CMD_GET_STATUS
        resp = self.set_register16_response(REGISTER, 2)
        logging.info(f"Chip Status = {resp}")
        REGISTER = GET_IMAGER_ERROR_CMD
        resp = self.set_register16_response(REGISTER, 2)
        logging.debug(f"0x0038 Status = {resp}")

    def switch_from_standard_to_burst(self):
        """Switch from standard mode to burst mode. Returns True on success."""
        REGISTER = SET_SWITCH_TO_BURST_MODE << 16
        self.set_register16_no_response(REGISTER)
        return True

    def switch_from_burst_to_standard(self):
        """Switch from burst mode to standard mode. Returns True on success."""
        # 8-word command: AD00 0010 0000 0000 1000 0000 0100 0000
        REGISTER = 0xAD000010000000001000000001000000
        self.set_register16_no_response(REGISTER)
        return True

    def force_stop_burst_mode(self):
        logging.debug("Forcing burst mode off")
        self.switch_from_burst_to_standard()
        return self.get_status()
        # Get chip ID of 2nd Pulsatrix which is on the slave side
        REGISTER = GET_SLAVE_CHIP_ID_CMD
        print("Getting Slave Pulsatrix Chip ID")
        resp = self.set_register16_response(REGISTER, 2)
        logging.info(f"Fun: Slave Pulsatrix Chip ID = {resp}")
        print(f"Fun: Slave Pulsatrix Chip ID = {resp}")
        return resp

    def get_generic_resp(self):
        # Get chip ID of 2nd Pulsatrix which is on the slave side
        REGISTER = 0x005A
        print("Getting generic respons")
        resp = self.set_register16_response(REGISTER, 2)
        logging.info(f"Fun: Generic Response = {resp}")
        print(f"Fun: Generic Response  = {resp}")
        return resp

    def get_only_status(self):
        logging.debug("Fetching status")
        # Get chip ID
        REGISTER = ADSD3500_CMD_GET_STATUS
        resp = self.set_register16_response(REGISTER, 2)
        print(f"Chip Status = {resp}")
        logging.info(f"Chip Status = {resp}")
        return resp

    def get_imager_type_and_ccb_version(self):
        """Read register 0x0032 to detect imager type and CCB version.

        Response layout (2 bytes):
          resp[0] = Imager Type  (1=ADSD3100, 2=ADTF3066)
          resp[1] = CCB Version  (1?V0, 2?V1, 3?V2, 4?V3)

        Updates _imager_type, _width, _height, _pixel_width, _pixel_height.
        Mirrors Adcam::get_imager_type_and_ccb_version() in adcam_lib.cpp.
        """
        REGISTER = ADSD3500_CMD_GET_CHIP_INFO
        resp = self.set_register16_response(REGISTER, 2)
        if resp is None or len(resp) < 2:
            logging.error("get_imager_type_and_ccb_version: incomplete response")
            return

        imager_type = resp[0] & LOW_BYTE_MASK
        ccb_version = resp[1] & LOW_BYTE_MASK

        ccb_str = {1: "Version 0", 2: "Version 1", 3: "Version 2", 4: "Version 3"}.get(
            ccb_version, "Unknown"
        )
        imager_str = {
            ADCAM_IMAGER_TYPE_ADSD3100: "ADSD3100",
            ADCAM_IMAGER_TYPE_ADTF3066: "ADTF3066",
        }.get(imager_type, "Unknown")

        logging.info(
            f"Imager Type: {imager_str} (raw={imager_type}), "
            f"CCB Version: {ccb_str} (raw={ccb_version})"
        )
        print(
            f"Imager Type: {imager_str} (raw={imager_type}), "
            f"CCB Version: {ccb_str} (raw={ccb_version})"
        )

        self._imager_type = imager_type

        if imager_type == ADCAM_IMAGER_TYPE_ADSD3100:
            table = ADSD3100_STANDARD_MODES
        elif imager_type == ADCAM_IMAGER_TYPE_ADTF3066:
            table = ADTF3066_STANDARD_MODES
        else:
            logging.error(
                f"get_imager_type_and_ccb_version: unsupported imager type (raw={imager_type})"
            )
            return

        mode_cfg = adcam_find_mode(table, self._adcam_mode)
        if mode_cfg is None:
            logging.error(
                f"get_imager_type_and_ccb_version: mode {self._adcam_mode} "
                f"not found for {imager_str}; keeping current geometry"
            )
            return

        self._width = mode_cfg.width
        self._height = mode_cfg.height
        self._pixel_width = mode_cfg.pixel_width
        self._pixel_height = mode_cfg.pixel_height

        logging.info(
            f"Mode table selected: {imager_str} mode={self._adcam_mode} "
            f"mipi={self._width}x{self._height} pixel={self._pixel_width}x{self._pixel_height}"
        )
        print(
            f"Mode table selected: {imager_str} mode={self._adcam_mode} "
            f"mipi={self._width}x{self._height} pixel={self._pixel_width}x{self._pixel_height}"
        )

    def probe_adcam_adtf3175(self):
        # Get chip ID
        REGISTER = GET_MASTER_CHIP_ID_CMD
        resp = self.set_register16_response(REGISTER, 2)
        logging.info(f"Chip ID = {resp}")

        if resp is None or len(resp) < 2:
            return 0
        if resp[0] == 0x59 and resp[1] == 0x31:
            return 1
        else:
            return 0

    """def force_stop_burst_mode(self):
        logging.debug("Forcing burst mode off")
        self.switch_from_burst_to_standard()
        return self.get_status()"""

    def get_fw_version(self, cmd=GET_MASTER_FIRMWARE_COMMAND):
        """Read firmware version: switches to burst mode, reads, then returns to standard."""
        logging.debug("Fetching FW version")
        self.switch_from_standard_to_burst()
        resp = self.get_fw_version_burst_mode(cmd)
        self.switch_from_burst_to_standard()
        return resp

    def get_fw_version_burst_mode(self, cmd=GET_MASTER_FIRMWARE_COMMAND):
        """Read firmware version while already in burst mode (no mode switching).

        cmd = GET_MASTER_FIRMWARE_COMMAND (0x01) or GET_SLAVE_FIRMWARE_COMMAND (0x04)
        Mirrors Adcam::get_fw_version_burst_mode() in adcam_lib.cpp.
        """
        # 8-word command: AD00 2C05 0000 0000 3100 0000 [cmd]00 0000
        # Byte 12 (0-indexed) encodes the command byte.
        BASE = 0xAD002C05000000003100000000000000
        REGISTER = BASE | (cmd << 24)
        resp = self.set_register16_response(REGISTER, 44)
        if resp is not None and len(resp) == 44:
            ver = f"{resp[0]}.{resp[1]}.{resp[2]}.{resp[3]}"
            label = "Master" if cmd == GET_MASTER_FIRMWARE_COMMAND else "Slave"
            logging.info(f"{label} Firmware version = {ver}")
            print(f"{label} Firmware version = {ver}")
        return resp

    def get_master_fw_version(self):
        """Read master firmware version (alias for get_fw_version_burst_mode(master))."""
        return self.get_fw_version_burst_mode(GET_MASTER_FIRMWARE_COMMAND)

    def get_slave_fw_version(self):
        """Read slave firmware version (alias for get_fw_version_burst_mode(slave))."""
        return self.get_fw_version_burst_mode(GET_SLAVE_FIRMWARE_COMMAND)

    def get_chip_status(self):
        logging.debug("Fetching status")
        # Get Status
        REGISTER = ADSD3500_CMD_GET_STATUS
        resp = self.set_register16_response(REGISTER, 2)
        logging.info(f"Chip Status = {resp}")

        REGISTER = GET_IMAGER_ERROR_CMD
        resp = self.set_register16_response(REGISTER, 2)
        logging.debug(f"Register 0x0038 Status = {resp}")

    def set_register(self, register, value, timeout=None):
        logging.debug(
            "set_register(register=%d(0x%X), value=%d(0x%X))"
            % (register, register, value, value)
        )
        write_bytes = bytearray(100)
        serializer = hololink_module.Serializer(write_bytes)
        serializer.append_uint16_be(register)
        serializer.append_uint8(value)
        read_byte_count = 0
        self._i2c.i2c_transaction(
            self.ADCAM_I2C_BUS_ADDRESS,
            write_bytes[: serializer.length()],
            read_byte_count,
            timeout=timeout,
        )

    def set_register16_no_response(self, register, resp_len=0, timeout=None):
        try:
            write_bytes = bytearray(100)
            uint16_array = self.format_registers(register)
            logging.debug("ADCAM REGISTER NORESPREQD =(0x%X))" % (register))

            serializer = hololink_module.Serializer(write_bytes)
            for i in range(0, len(uint16_array), 1):
                chunk = uint16_array[i]
                # The inner function handles the format string correctly for each chunk
                # serializer.append_uint16_be(struct.unpack('>h', chunk))
                serializer.append_uint16_be(chunk)

            read_byte_count = 0
            self._i2c.i2c_transaction(
                self.ADCAM_I2C_BUS_ADDRESS,
                serializer.data(),
                read_byte_count,
                timeout=timeout,
            )
            # print("ADCAM:", hex(register), "Response", None)
            # logging.debug("set_register16_no_response write =",serializer.data())
        except AttributeError as e:
            logging.info(f"[ERROR] Attribute missing or invalid object used: {e}")
            return None
        except ValueError as e:
            logging.info(f"[ERROR] Value or data format issue: {e}")
            return None
        except OSError as e:
            logging.info(f"[ERROR] I2C communication failed (OS error): {e}")
            return None
        except Exception as e:
            logging.info(f"[ERROR] Unexpected failure during I2C transaction: {e}")
            return None

    def set_register16_response(self, register, resp_len, timeout=None):
        try:
            write_bytes = bytearray(100)
            uint16_array = self.format_registers(register)
            logging.debug("ADCAM REGISTER RESPREQD =(0x%X))" % (register))
            serializer = hololink_module.Serializer(write_bytes)
            for i in range(0, len(uint16_array), 1):
                chunk = uint16_array[i]
                # The inner function handles the format string correctly for each chunk
                serializer.append_uint16_be(chunk)
            read_byte_count = resp_len
            adi_tof_timeout = hololink_module.Timeout(30, retry_s=0.2)
            reply = self._i2c.i2c_transaction(
                self.ADCAM_I2C_BUS_ADDRESS,
                serializer.data(),
                read_byte_count,
                timeout=adi_tof_timeout,
            )
            deserializer = hololink_module.Deserializer(reply)
            deserializer.next_uint8()
            logging.debug(f"ADCAM REGISTER REPLY from Chip = {reply}")
            print("ADCAM:", hex(register), "Response", reply)
            return reply
        except AttributeError as e:
            logging.info(f"[ERROR] Attribute missing or invalid object used: {e}")
            print("ADCAM Attribute Error:", hex(register))
            return reply
        except ValueError as e:
            logging.info(f"[ERROR] Value or data format issue: {e}")
            print("ADCAM ValueError Error:", hex(register))
            return reply
        except OSError as e:
            logging.info(f"[ERROR] I2C communication failed (OS error): {e}")
            print("ADCAM OSError Error:", hex(register))
            return None
        except Exception as e:
            logging.info(f"[ERROR] Unexpected failure during I2C transaction: {e}")
            print("ADCAM exception Error:", hex(register))
            return None

    # Conceptually this funciton is same as def set_register16_no_response(self, register, resp_len=0, timeout=None):
    def write_raw_data(self, data_chunk, data_len, timeout=None):
        try:
            write_bytes = bytearray(1000)
            uint16_array = self.bytes_to_uint16_array(data_chunk)
            serializer = hololink_module.Serializer(write_bytes)
            # print (f"Debug: With response write_raw_data : Len =  {len(uint16_array)}")
            for i in range(0, len(uint16_array), 1):
                chunk = uint16_array[i]
                # The inner function handles the format string correctly for each chunk
                serializer.append_uint16_be(chunk)
            read_byte_count = 0
            # print(f"write_raw_data: Serializer data (hex): {[f'0x{b:02x}' for b in serializer.data()]}")
            # Print data in 16 bytes per line for easier debugging
            # data = serializer.data()
            # print("write_raw_data: Serializer data (hex):")
            # for i in range(0, len(data), 16):
            #    line = ' '.join(f'0x{b:02x}' for b in data[i:i+16])
            #    print(f"  [{i:04d}]: {line}")
            self._i2c.i2c_transaction(
                self.ADCAM_I2C_BUS_ADDRESS,
                serializer.data(),
                read_byte_count,
                timeout=timeout,
            )
            return True
        except AttributeError as e:
            logging.info(f"[ERROR] Attribute missing or invalid object used: {e}")
            return None
        except ValueError as e:
            logging.info(f"[ERROR] Value or data format issue: {e}")
            return None
        except OSError as e:
            logging.info(f"[ERROR] I2C communication failed (OS error): {e}")
            return None
        except Exception as e:
            logging.info(f"[ERROR] Unexpected failure during I2C transaction: {e}")
            return None

    def read_raw_data(self, data_chunk, read_len, timeout=None):
        try:
            write_dummy_bytes = b""  # This is plain read. So write data is null.
            read_byte_count = read_len
            reply = self._i2c.i2c_transaction(
                self.ADCAM_I2C_BUS_ADDRESS,
                write_dummy_bytes,
                read_byte_count,
                timeout=timeout,
            )
            deserializer = hololink_module.Deserializer(reply)
            deserializer.next_uint8()
            logging.info(f"read_raw_data = {reply}")
            print("read_raw_data : Response", reply)
            return reply
        except AttributeError as e:
            logging.info(f"[ERROR] Attribute missing or invalid object used: {e}")
            return None
        except ValueError as e:
            logging.info(f"[ERROR] Value or data format issue: {e}")
            return None
        except OSError as e:
            logging.info(f"[ERROR] I2C communication failed (OS error): {e}")
            return None
        except Exception as e:
            logging.info(f"[ERROR] Unexpected failure during I2C transaction: {e}")
            return None

    def stream_on(self):
        logging.debug("Setting Clock continuous mode in stream_on")
        CONT_MODE = (MIPI_CLK_CONTINUOUS_CMD << 16) | ENABLE_VAL
        self.set_register16_no_response(CONT_MODE)
        time.sleep(0.2)
        logging.debug("Turning ON Streaming")
        STREAM_MODE = (STREAM_ON_CMD << 16) | STREAM_ON_VAL
        self.set_register16_no_response(STREAM_MODE)
        time.sleep(0.2)
        self.get_status()

    def stream_off(self):
        logging.debug("Turning OFF Streaming")
        STREAM_MODE = (STREAM_OFF_CMD << 16) | STREAM_OFF_VAL
        self.set_register16_no_response(STREAM_MODE)
        time.sleep(0.2)
        self.get_status()

    def get_ChipID(self, cmd=GET_MASTER_CHIP_ID_CMD):
        """Read chip ID. cmd = GET_MASTER_CHIP_ID_CMD (0x0112) or GET_SLAVE_CHIP_ID_CMD (0x0116)."""
        resp = self.set_register16_response(cmd, 2)
        label = "Master" if cmd == GET_MASTER_CHIP_ID_CMD else "Slave"
        logging.info(f"{label} Chip ID = {resp}")
        if resp is not None and len(resp) >= 2:
            print(f"{label} Chip ID = 0x{resp[0]:02X}{resp[1]:02X}")
            return True
        return False

    def get_Status(self):
        # Get Status
        REGISTER = ADSD3500_CMD_GET_STATUS
        resp = self.set_register16_response(REGISTER, 2)
        logging.info(f"Chip Status = {resp}")
        return resp

    def get_ClockContinuousMode(self):
        # Get Status
        REGISTER = GET_MIPI_CLK_CONTINUOUS_CMD
        resp = self.set_register16_response(REGISTER, 2)
        logging.debug(f"Clock continuous mode = {resp}")
        return resp

    # ---- Accessors (mirror Adcam::get_*() in adcam_lib.hpp) ----

    def get_width(self):
        """MIPI frame line width in bytes."""
        return self._width

    def get_height(self):
        """MIPI frame line count."""
        return self._height

    def get_pixel_width(self):
        """Actual image pixels per row."""
        return self._pixel_width

    def get_pixelsize(self):
        """Actual image pixels per row."""
        return self._byteperpixel

    def get_pixel_height(self):
        """Actual image pixel rows."""
        return self._pixel_height

    def get_mode(self):
        """Current capture mode index."""
        return self._adcam_mode

    def get_num_planes(self):
        """Current capture mode index."""
        return self._num_planes

    def get_imager_type(self):
        """Detected imager type: ADCAM_IMAGER_TYPE_ADSD3100=1, ADCAM_IMAGER_TYPE_ADTF3066=2."""
        return self._imager_type

    # def adcam_reset_power_on(self, hololink, hololink_channel, channel_metadata):
    def adcam_reset_power_on(self):
        logging.debug("Resetting ADCAM")
        self._pf_gpio.pull_gpio_low()
        self._expander0.set_register(0x0, 0x0)  # Force all the Expandoer 0 bits as 0
        self._expander1.set_register(0x0, 0x0)  # Force all the Expandoer 1 bits as 0

        # O1 (EN_0P8) => 1 //158  Power enable //O7 O6 P5 P4 P3 P2 O1 O0 bits
        # Expander0.get_register(0x02) #This should turn on the DS1 LED ON
        self._expander0.set_register(0x02, 0x02)  # This should turn on the DS1 LED ON
        logging.info("Check DS1 LED - it should be ON. This will be on ")
        # Expander0.get_register(0x02) #This should turn on the DS1 LED ON
        time.sleep(0.2)  # LED ON for 10 sec
        self._expander0.set_register(0x0, 0x0)  # This should turn OFF the DS1 LED
        logging.info("DS1 LED turned off")
        # Checking DONE!

        # P5  (HOST_IO_SEL) => 1 //114 //O7 O6 P5 P4 P3 P2 O1 O0 bits
        self._expander0.set_register(0x20, 0x20)  # E0 = 0x20
        # O8 (HOST_IO_DIR) => 1 //117 //O15 O14 O13 O12 O11 O10 O9 O8 bits
        # O11 (FSYNC_DIR) => 1 //120 //O15 O14 O13 O12 O11 O10 O9 O8 bits
        self._expander1.set_register(0x9, 0x9)  # E1 = 0x09

        # RST to low
        # Add code to make RST GPIO Low

        # O0 (EN_1P8) => 0 //127. Power disable //O7 O6 P5 P4 P3 P2 O1 O0 bits
        # O1 (EN_0P8) => 0 //130. Power disable //O7 O6 P5 P4 P3 P2 O1 O0 bits
        self._expander0.set_register(0x20, 0x20)  # E0 = 0x20 //No change. Remains same
        time.sleep(0.2)  # Pauses execution for 0.2 seconds (200 milliseconds)

        # P3  (I2CM_SET) => 0 //135. Enable SPI for imager to pulsatrix //O7 O6 P5 P4 P3 P2 O1 O0 bits
        # O6 (ISP_BS0) => 0 //138  - Boot strap pins //O7 O6 P5 P4 P3 P2 O1 O0 bits
        # O7 (ISP_BS1) => 0 //141 - Boot strap pins //O7 O6 P5 P4 P3 P2 O1 O0 bits
        self._expander0.set_register(0x20, 0x20)  # E0 = 0x20 //No change. Remains same

        # O9 (ISP_BS4) => 0 //143 - Boot strap pins //O15 O14 O13 O12 O11 O10 O9 O8 bits
        # O10 (ISP_BS5) => 0 //147 - Boot strap pins //O15 O14 O13 O12 O11 O10 O9 O8 bits
        self._expander1.set_register(0x9, 0x9)  # E1 = 0x09 //No change. Remains same

        # O0 (EN_1P8) => 1 //153. Power enable //O7 O6 P5 P4 P3 P2 O1 O0 bits
        self._expander0.set_register(0x21, 0x21)  # E0 = 0x21
        time.sleep(0.2)  # Pauses execution for 0.2 seconds (200 milliseconds)

        # O1 (EN_0P8) => 1 //158  Power enable //O7 O6 P5 P4 P3 P2 O1 O0 bits
        self._expander0.set_register(0x23, 0x23)  # E0 = 0x23
        time.sleep(0.2)  # Pauses execution for 0.2 seconds (200 milliseconds)

        # O14 (EN_VSYS) => 1 //163 //O15 O14 O13 O12 O11 O10 O9 O8 bits
        # O13 (EN_VAUX_LS) => 1 //166 //O15 O14 O13 O12 O11 O10 O9 O8 bits
        # O12 (EN_VAUX) => 1 //169 //O15 O14 O13 O12 O11 O10 O9 O8 bits
        self._expander1.set_register(0x79, 0x79)  # E1 = 0x79
        time.sleep(0.2)  # Pauses execution for 0.2 seconds (200 milliseconds)
        self._pf_gpio.pull_gpio_high()

        logging.info("booting up ADSD, wait for 5 seconds")
        time.sleep(5)  # Boot-up ADSD3500

    # def adcam_Only_reset(self, hololink, hololink_channel, channel_metadata):
    def adcam_Only_reset(self):
        # pf_gpio = polargpio.polarfireGpio(hololink_channel, hololink.get_gpio(channel_metadata) )
        logging.debug("ADCAM - Making Reset LOW ONLY")
        self._pf_gpio.pull_gpio_low()

        time.sleep(1)  # Pauses execution for 0.2 seconds (200 milliseconds)
        logging.debug("ADCAM - Making Reset HIGH ONLY")
        self._pf_gpio.pull_gpio_high()

        logging.info("Waiting 5 secs after reset")
        time.sleep(5)  # Boot-up ADSD3500

    def configure_converter(self, converter):
        # where do we find the first received byte?
        start_byte = converter.receiver_start_byte()
        transmitted_line_bytes = converter.transmitted_line_bytes(
            self._pixel_format, self._width * self._byteperpixel
        )
        received_line_bytes = converter.received_line_bytes(transmitted_line_bytes)

        embedded_data_bytes = self._metadata_sz
        start_byte += converter.received_line_bytes(embedded_data_bytes)
        converter.configure(
            start_byte,
            received_line_bytes,
            self._width * self._byteperpixel,
            self._height,
            self._pixel_format,
        )

    def pixel_format(self):
        return self._pixel_format

    def bayer_format(self):
        return hololink_module.sensors.csi.BayerFormat.RGGB

    def start(self):
        """Set clock continuous mode then enable streaming (mirrors Adcam::start() in C++)."""
        logging.debug("Setting Clock continuous mode")
        CONT_MODE = (MIPI_CLK_CONTINUOUS_CMD << 16) | ENABLE_VAL
        self.set_register16_no_response(CONT_MODE)
        time.sleep(0.2)

        logging.info(f"Turning ON Streaming TS in sec= {int(time.time())}")
        STREAM_MODE = (STREAM_ON_CMD << 16) | STREAM_ON_VAL
        self.set_register16_no_response(STREAM_MODE)
        time.sleep(0.2)
        self.get_status()
        # self.set_register16_response(0x0058, 2) # read FSYNC status, enabled for debug

    def stop(self):
        """Stop Streaming"""
        logging.info(f"Turning OFF Streaming TS in sec= {int(time.time())}")
        self.set_register16_response(0x0058, 2)  # read FSYNC status
        STREAM_MODE = (STREAM_OFF_CMD << 16) | STREAM_OFF_VAL
        self.set_register16_no_response(STREAM_MODE)
        time.sleep(0.2)
        self.get_status()

    # =========================================================================
    # Firmware flash (dual-slot binary) — mirrors adsd3500_flash.cpp
    # =========================================================================

    def adsd3500_flash(self, file_data, force=False):
        """Flash ADSD3500 firmware from a dual-slot binary file.

        Binary layout:
          Slot 0 (offset 0):               Master FW (chunkId=0xAD, chunkType=0x54)
          Slot 1 (offset ADI_DUAL_FW_SLOT_SIZE): Slave FW (chunkId=0xAD, chunkType=0x60)

        Each slot: [20-byte chunk header] [firmware payload] [4-byte LE CRC trailer]
        Mirrors Adsd3500::adsd3500_flash() in adsd3500_flash.cpp.
        """
        if len(file_data) < 2 * ADI_DUAL_FW_SLOT_SIZE:
            print("Firmware file too small to contain both firmware slots")
            return False

        # --- Validate Slot 0 (master) ---
        if file_data[0] != 0xAD or file_data[1] != 0x54:
            print(
                f"Invalid Slot 0 header (expected 0xAD 0x54, "
                f"got 0x{file_data[0]:02X} 0x{file_data[1]:02X})"
            )
            return False
        master_len = (
            file_data[8]
            | (file_data[9] << 8)
            | (file_data[10] << 16)
            | (file_data[11] << 24)
        )
        if (
            master_len == 0
            or master_len > ADI_DUAL_FW_SLOT_SIZE - ADI_CHUNK_HEADER_SIZE
        ):
            print(f"Invalid master firmware size: {master_len} bytes")
            return False

        # --- Validate Slot 1 (slave) ---
        s1 = ADI_DUAL_FW_SLOT_SIZE
        if file_data[s1] != 0xAD or file_data[s1 + 1] != 0x60:
            print(
                f"Invalid Slot 1 header (expected 0xAD 0x60, "
                f"got 0x{file_data[s1]:02X} 0x{file_data[s1+1]:02X})"
            )
            return False
        slave_len = (
            file_data[s1 + 8]
            | (file_data[s1 + 9] << 8)
            | (file_data[s1 + 10] << 16)
            | (file_data[s1 + 11] << 24)
        )
        if slave_len == 0 or slave_len > ADI_DUAL_FW_SLOT_SIZE - ADI_CHUNK_HEADER_SIZE:
            print(f"Invalid slave firmware size: {slave_len} bytes")
            return False

        # --- Extract payloads ---
        master_fw = bytes(
            file_data[ADI_CHUNK_HEADER_SIZE : ADI_CHUNK_HEADER_SIZE + master_len]
        )
        slave_fw = bytes(
            file_data[
                s1 + ADI_CHUNK_HEADER_SIZE : s1 + ADI_CHUNK_HEADER_SIZE + slave_len
            ]
        )

        if all(b == 0 for b in master_fw):
            print("[ERR] Slot 0 master firmware payload is all zeros. Aborting.")
            return False
        if all(b == 0 for b in slave_fw):
            print("[ERR] Slot 1 slave firmware payload is all zeros. Aborting.")
            return False

        # --- Extract CRC trailers (little-endian uint32 after each payload) ---
        off_m = ADI_CHUNK_HEADER_SIZE + master_len
        master_expected_crc = (
            file_data[off_m]
            | (file_data[off_m + 1] << 8)
            | (file_data[off_m + 2] << 16)
            | (file_data[off_m + 3] << 24)
        )
        off_s = s1 + ADI_CHUNK_HEADER_SIZE + slave_len
        slave_expected_crc = (
            file_data[off_s]
            | (file_data[off_s + 1] << 8)
            | (file_data[off_s + 2] << 16)
            | (file_data[off_s + 3] << 24)
        )
        print(f"[INFO] Header Master CRC : 0x{master_expected_crc:08X}")
        print(f"[INFO] Header Slave CRC  : 0x{slave_expected_crc:08X}")

        # --- Version consistency check (first 4 bytes = version) ---
        if master_len < 4 or slave_len < 4:
            print("Firmware payload too small to contain a version number")
            return False
        master_ver = f"{master_fw[0]}.{master_fw[1]}.{master_fw[2]}.{master_fw[3]}"
        slave_ver = f"{slave_fw[0]}.{slave_fw[1]}.{slave_fw[2]}.{slave_fw[3]}"
        print(f"[INFO] Master firmware version : {master_ver}")
        print(f"[INFO] Slave  firmware version : {slave_ver}")
        if master_fw[:4] != slave_fw[:4]:
            print(
                f"[ERR] Version mismatch: master={master_ver} slave={slave_ver}. Aborting."
            )
            return False
        print(f"[INFO] Firmware version match confirmed: {master_ver}")

        # --- Probe master device (mandatory) ---
        master_resp = self.set_register16_response(GET_MASTER_CHIP_ID_CMD, 2)
        if master_resp is None or len(master_resp) < 2:
            print("No ADSD3500 master device detected. Aborting firmware update.")
            return False
        master_chip_id = (master_resp[0] << 8) | master_resp[1]
        print(f"[INFO] Master Chip ID is: 0x{master_chip_id:04X}")

        # --- Probe slave device (optional) ---
        # NOTE: Reading slave chip ID (0x0116) directly causes I2C NAK and
        # leaves the bus disturbed, making the subsequent 0x005A query also
        # fail. This matches the C++ adsd3500_flash.cpp behaviour where the
        # slave chip ID probe is commented out (marked "for debugging").
        # Instead, query 0x005A (GET_DUAL_ADSD3500_ENABLED_CMD) directly via
        # the master — this is the reliable way to detect dual configuration.
        slave_found = False
        dual_resp = self.set_register16_response(GET_DUAL_ADSD3500_ENABLED_CMD, 2)
        if dual_resp is not None and len(dual_resp) >= 2:
            dual_enabled = (dual_resp[0] << 8) | dual_resp[1]
            print(f"[INFO] Get Is Dual ADSD3500 Enabled (0x005A): 0x{dual_enabled:04X}")
            if dual_enabled == ENABLE_VAL:
                print(
                    "[INFO] Dual ADSD3500 is enabled. Slave confirmed via master query."
                )
                slave_found = True
            else:
                print("[INFO] Dual ADSD3500 disabled. Single-device configuration.")
        else:
            print(
                "[INFO] Dual-enable query (0x005A) failed. Assuming single-device configuration."
            )

        # --- Execute update(s) ---
        if slave_found:
            print(
                "\nBoth ADSD3500 devices detected. Updating master and slave firmware."
            )
            if not self._update_adsd3500_master_firmware(
                master_fw, master_len, force, master_expected_crc
            ):
                print("Master firmware update failed.")
                return False
            if not self._update_adsd3500_slave_firmware(
                slave_fw, slave_len, force, slave_expected_crc
            ):
                print("Slave firmware update failed.")
                return False
        else:
            print("\nSingle ADSD3500 device detected. Updating master firmware only.")
            if not self._update_adsd3500_master_firmware(
                master_fw, master_len, force, master_expected_crc
            ):
                print("Master firmware update failed.")
                return False

        return True

    def _compute_fw_crc(self, fw_data):
        """Compute CRC-32 of firmware data matching C++ compute_crc() with IS_CRC_MIRROR."""
        crc_params = CrcParametersUnion()
        crc_params.type = CRC_TYPE.CRC_32bit
        crc_params.initial_crc.crc_32bit = ADI_ROM_CFG_CRC_SEED_VALUE
        crc_params.crc_compute_flags = IS_CRC_MIRROR
        raw_crc = compute_crc_python(crc_params, fw_data)
        return (~raw_crc) & 0xFFFFFFFF

    def _check_fw_version_constraints(self, fw_data, current_ver_bytes, label, force):
        """Validate minimum version and check for downgrades. Returns False to abort."""
        if len(fw_data) < 4:
            return True  # too short to check; let it proceed
        new_ver = tuple(fw_data[:4])
        cur_ver = tuple(current_ver_bytes[:4])
        new_str = ".".join(str(v) for v in new_ver)
        cur_str = ".".join(str(v) for v in cur_ver)
        print(f"[{label}] Update firmware version   : {new_str}")

        # Minimum version check: must be >= FW_MIN_VERSION
        if new_ver < FW_MIN_VERSION:
            min_str = ".".join(str(v) for v in FW_MIN_VERSION)
            print(
                f"[{label}] ERROR: Firmware version {new_str} is below "
                f"the minimum required version {min_str}. Aborting."
            )
            return False

        # Downgrade check
        if new_ver < cur_ver:
            print(f"\n[{label}] WARNING: Downgrade detected!")
            print(f"  Current version : {cur_str}")
            print(f"  Update version  : {new_str}")
            if not force:
                print(
                    "Downgrade requires explicit confirmation. Re-run with force=True."
                )
                return False
            print(f"[{label}] Proceeding with downgrade (force=True).")
        return True

    def _update_adsd3500_master_firmware(self, fw_data, fw_len, force, expected_crc):
        """Flash master ADSD3500 firmware. Mirrors updateAdsd3500MasterFirmware()."""
        print(
            "\n===== _update_adsd3500_master_firmware: Starting Master Firmware Update ====="
        )
        self.get_ChipID(GET_MASTER_CHIP_ID_CMD)
        time.sleep(1)

        print("[MASTER] Switching to burst mode")
        self.switch_from_standard_to_burst()
        time.sleep(1)

        print("[MASTER] Before upgrading new firmware")
        current_ver = self.get_fw_version_burst_mode(GET_MASTER_FIRMWARE_COMMAND)
        if current_ver is None or len(current_ver) < 44:
            print(
                f"[MASTER] Failed to read current firmware version (got "
                f"{0 if current_ver is None else len(current_ver)} bytes, expected 44)"
            )
            self.switch_from_burst_to_standard()
            return False
        print(
            f"[MASTER] Current firmware version  : "
            f"{current_ver[0]}.{current_ver[1]}.{current_ver[2]}.{current_ver[3]}"
        )

        if not self._check_fw_version_constraints(
            fw_data, current_ver, "MASTER", force
        ):
            self.switch_from_burst_to_standard()
            return False

        # Compute and verify CRC
        computed_crc = self._compute_fw_crc(fw_data)
        print(f"[MASTER] nResidualCRC   : 0x{computed_crc:08X}")
        print(f"[MASTER] Expected CRC   : 0x{expected_crc:08X}")
        if computed_crc != expected_crc:
            print(
                f"[MASTER] CRC MISMATCH: computed 0x{computed_crc:08X} "
                f"!= expected 0x{expected_crc:08X}"
            )
            return False
        print("[MASTER] CRC OK: computed CRC matches expected CRC.")

        # Send firmware header
        if not self.sendHeader(fw_data, fw_len, "master"):
            print("[MASTER] Failed to send fw upgrade header")
            return False

        # Send firmware packets
        packets_to_send = math.ceil(fw_len / FLASH_PAGE_SIZE)
        print(f"\n[MASTER] Writing Firmware packets ({packets_to_send} total)...")
        for i in range(packets_to_send):
            chunk = bytes(fw_data[i * FLASH_PAGE_SIZE : (i + 1) * FLASH_PAGE_SIZE])
            if len(chunk) < FLASH_PAGE_SIZE:
                chunk = chunk.ljust(FLASH_PAGE_SIZE, b"\x00")
            if not self.write_raw_data(chunk, FLASH_PAGE_SIZE):
                print(f"\n[MASTER] Failed to send packet {i + 1} of {packets_to_send}!")
                return False
            print(f"[MASTER] Packet number: {i + 1} / {packets_to_send}", end="\r")
        print()

        print("\n[MASTER] Adsd3500 master firmware packets sent successfully!")
        print()
        for i in range(20, -1, -1):
            time.sleep(1)
            print(f"[MASTER] Waiting for {i} seconds", end="\r")
        print()

        status_resp = self.set_register16_response(ADSD3500_CMD_GET_STATUS, 2)
        status = 0
        if status_resp is not None and len(status_resp) >= 2:
            status = (status_resp[0] << 8) | status_resp[1]
        print(f"[MASTER] Get status Command 0x{status:04X}")
        if status != ADI_STATUS_FIRMWARE_UPDATE:
            print("[MASTER] Firmware update failed")
            return False

        time.sleep(2)
        print("[MASTER] Firmware soft resetting...")
        self.softreset()

        print()
        for i in range(9, -1, -1):
            time.sleep(1)
            print(f"[MASTER] Waiting for {i} seconds", end="\r")
        print()

        self.get_ChipID(GET_MASTER_CHIP_ID_CMD)
        time.sleep(1)

        self.switch_from_standard_to_burst()
        time.sleep(1)

        print("\n[MASTER] After upgrading new firmware")
        updated_ver = self.get_fw_version_burst_mode(GET_MASTER_FIRMWARE_COMMAND)
        if updated_ver is not None and len(updated_ver) >= 4:
            print(
                f"[MASTER] Updated firmware version   : "
                f"{updated_ver[0]}.{updated_ver[1]}.{updated_ver[2]}.{updated_ver[3]}"
            )
        time.sleep(1)

        self.switch_from_burst_to_standard()
        time.sleep(1)

        self.get_ChipID(GET_MASTER_CHIP_ID_CMD)
        return True

    def _update_adsd3500_slave_firmware(self, fw_data, fw_len, force, expected_crc):
        """Flash slave ADSD3500 firmware. Mirrors updateAdsd3500SlaveFirmware()."""
        print(
            "\n===== _update_adsd3500_slave_firmware: Starting Slave Firmware Update ====="
        )
        time.sleep(1)

        print("[SLAVE] Switching to burst mode")
        self.switch_from_standard_to_burst()
        time.sleep(1)

        print("\n[SLAVE] Before upgrading new firmware")
        current_ver = self.get_fw_version_burst_mode(GET_SLAVE_FIRMWARE_COMMAND)
        if current_ver is not None and len(current_ver) >= 4:
            print(
                f"[SLAVE] Current firmware version   : "
                f"{current_ver[0]}.{current_ver[1]}.{current_ver[2]}.{current_ver[3]}"
            )

        if current_ver is not None and len(current_ver) >= 4:
            if not self._check_fw_version_constraints(
                fw_data, current_ver, "SLAVE", force
            ):
                self.switch_from_burst_to_standard()
                return False

        # Compute and verify CRC
        computed_crc = self._compute_fw_crc(fw_data)
        print(f"[SLAVE] nResidualCRC   : 0x{computed_crc:08X}")
        print(f"[SLAVE] Expected CRC   : 0x{expected_crc:08X}")
        if computed_crc != expected_crc:
            print(
                f"[SLAVE] CRC MISMATCH: computed 0x{computed_crc:08X} "
                f"!= expected 0x{expected_crc:08X}"
            )
            return False
        print("[SLAVE] CRC OK: computed CRC matches expected CRC.")

        # Send firmware header
        if not self.sendHeader(fw_data, fw_len, "slave"):
            print("[SLAVE] Failed to send fw upgrade header")
            return False

        # Send firmware packets
        packets_to_send = math.ceil(fw_len / FLASH_PAGE_SIZE)
        print(f"\n[SLAVE] Writing Firmware packets ({packets_to_send} total)...")
        for i in range(packets_to_send):
            chunk = bytes(fw_data[i * FLASH_PAGE_SIZE : (i + 1) * FLASH_PAGE_SIZE])
            if len(chunk) < FLASH_PAGE_SIZE:
                chunk = chunk.ljust(FLASH_PAGE_SIZE, b"\x00")
            if not self.write_raw_data(chunk, FLASH_PAGE_SIZE):
                print(f"\n[SLAVE] Failed to send packet {i + 1} of {packets_to_send}!")
                return False
            print(f"[SLAVE] Packet number: {i + 1} / {packets_to_send}", end="\r")
        print()

        print("\n[SLAVE] Adsd3500 slave firmware packets sent successfully!")
        print()
        for i in range(20, -1, -1):
            time.sleep(1)
            print(f"[SLAVE] Waiting for {i} seconds", end="\r")
        print()

        time.sleep(2)
        self.switch_from_burst_to_standard()
        time.sleep(1)

        status_resp = self.set_register16_response(ADSD3500_CMD_GET_STATUS, 2)
        status = 0
        if status_resp is not None and len(status_resp) >= 2:
            status = (status_resp[0] << 8) | status_resp[1]
        print(f"[SLAVE] Get status Command 0x{status:04X}")
        if status != ADI_STATUS_SECOND_FIRMWARE_FLASH_UPDATE:
            print("Slave Firmware write failed")
            return False
        print("Slave Firmware Flash write completed and is successful.")

        print("[SLAVE] Firmware soft resetting...")
        self.softreset()

        print()
        for i in range(9, -1, -1):
            time.sleep(1)
            print(f"[SLAVE] Waiting for {i} seconds", end="\r")
        print()

        self.switch_from_standard_to_burst()
        time.sleep(1)

        print("\n[SLAVE] After upgrading new firmware")
        updated_ver = self.get_fw_version_burst_mode(GET_SLAVE_FIRMWARE_COMMAND)
        if updated_ver is not None and len(updated_ver) >= 4:
            print(
                f"[SLAVE] Updated firmware version    : "
                f"{updated_ver[0]}.{updated_ver[1]}.{updated_ver[2]}.{updated_ver[3]}"
            )
        time.sleep(1)

        self.switch_from_burst_to_standard()
        time.sleep(1)
        return True

    def sendHeader(self, FWData, FWLen, target):
        header = CmdHeaderUnion()
        header.fields.id8 = 0xAD
        header.fields.chunk_size16 = FLASH_PAGE_SIZE
        if target == "master":
            header.fields.cmd8 = WRITE_MASTER_FIRMWARE_COMMAND
        else:
            header.fields.cmd8 = WRITE_SLAVE_FIRMWARE_COMMAND

        header.fields.total_size_fw32 = FWLen
        header.fields.header_checksum32 = 0
        temp_pack = struct.pack(
            ">H B I",
            header.fields.chunk_size16,
            header.fields.cmd8,
            header.fields.total_size_fw32,
        )
        header.fields.header_checksum32 = sum(temp_pack)

        crc_params = CrcParametersUnion()
        crc_params.type = CRC_TYPE.CRC_32bit
        nResidualCRC = ADI_ROM_CFG_CRC_SEED_VALUE
        crc_params.initial_crc.crc_32bit = nResidualCRC
        crc_params.crc_compute_flags = IS_CRC_MIRROR
        res_crc_32bit = compute_crc_python(crc_params, FWData)
        if target == "master":
            print(
                f"compute_crc_python() for master: returned CRC of: 0x{res_crc_32bit:08X}"
            )
        else:
            print(
                f"compute_crc_python() for slave : returned CRC of: 0x{res_crc_32bit:08X}"
            )
        nResidualCRC = (~res_crc_32bit) & 0xFFFFFFFF
        print(f"Calculated nResidualCRC: 0x{nResidualCRC:08X}")
        header.fields.crc_of_fw32 = nResidualCRC
        write_header_status = self.write_raw_data(bytes(header), 16)
        return write_header_status
