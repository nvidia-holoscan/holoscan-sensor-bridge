/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* IMX258 mode table, register map and geometry constants for the module-native
 * Imx258Cam driver on the CertusPro-NX Versa. Ports the reference bring-up camera
 * setup (camera_hdmi_setup.py: the IMX258_CONFIG_49HZ base table and the
 * CAM_CONFIG_REFERENCE_1080P60_1194MBPS profile built on it), the I2C addressing
 * and the frame-geometry constants into the shape of
 * hololink_module/host/sensors/imx274/imx274_mode.hpp. Data only. */

#ifndef HOLOLINK_MODULE_SENSORS_IMX258_MODE_HPP
#define HOLOLINK_MODULE_SENSORS_IMX258_MODE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "hololink/module/csi_converter.hpp"

namespace hololink::module::sensors::imx258 {

// The CSI format/converter types live in hololink::module::csi (they are
// not sensor types); alias them here so this driver's code can name them
// unqualified.
using ::hololink::module::csi::BayerFormat;
using ::hololink::module::csi::CsiConverterV1;
using ::hololink::module::csi::PixelFormat;

// The CertusPro-NX Versa's dynamic MIPI receiver locks to the 1194.75 Mb/s/lane
// profile; that is the one mode this driver offers.
enum class Imx258_Mode : uint32_t {
    IMX258_MODE_1920X1080_60FPS_1194MBPS = 0, // 1194.75 Mb/s/lane (CertusPro-NX Versa receiver)
    Unknown = 1,
};

struct FrameFormat {
    uint32_t width;
    uint32_t height;
    uint32_t framerate;
    PixelFormat pixel_format;
};

inline constexpr std::array<FrameFormat, 1> IMX258_FRAME_FORMAT { {
    FrameFormat { 1920, 1080, 60, PixelFormat::RAW_10 },
} };

// I2C addressing. The IMX258 sits on I2C bus 0 of the Versa board, not the
// CAM_I2C_BUS (1) an HSB board's camera uses; 7-bit device address 0x1A.
static constexpr uint32_t IMX258_I2C_ADDRESS = 0x1A;
static constexpr uint32_t DEFAULT_CAM_I2C_BUS = 0;

// Register addresses (16-bit, big-endian on the wire).
static constexpr uint16_t REG_MODEL_ID_HI = 0x0016;
static constexpr uint16_t REG_MODEL_ID_LO = 0x0017;
static constexpr uint16_t MODEL_ID = 0x0258; // expected {REG_MODEL_ID_HI, REG_MODEL_ID_LO}
static constexpr uint16_t REG_MODE_SELECT = 0x0100; // 0x01 streaming on, 0x00 off
static constexpr uint16_t REG_FRAME_COUNT = 0x0005; // SMIA frame counter: advances per frame while streaming, 0xFF idle
static constexpr uint16_t REG_VT_PLL_MULT = 0x0307; // video-timing PLL multiplier
static constexpr uint16_t REG_COARSE_INTEG_HI = 0x0202;
static constexpr uint16_t REG_COARSE_INTEG_LO = 0x0203;
static constexpr uint16_t REG_ANA_GAIN_HI = 0x0204;
static constexpr uint16_t REG_ANA_GAIN_LO = 0x0205;

// Frame geometry.
//
// RAW10_START_BYTE: measured, not derivable from the register table: the
// sensor emits a non-image prefix ahead of pixel row 0 even with PDAF
// disabled. If the image comes out skewed, this is the first constant to
// re-measure.
static constexpr uint32_t RAW10_START_BYTE = 640;

// Array size constants needed since C++17 doesn't support std::array size deduction with std::pair
constexpr size_t BASE_SEQUENCE_SIZE = 82;

// clang-format off
    /* The reference's base register table: 1920x1080, four-lane RAW10, as
     * transcribed verbatim from CAM_CONFIG_49HZ in camera_hdmi_setup.py. Not
     * a mode by itself here -- its output PLL runs 371.25 Mb/s/lane, which the
     * Versa's receiver does not lock to -- but the table the one offered mode
     * is composed from, so the reference's programming order is preserved.
     * That order is load-bearing (it is the order from the reference-design
     * ROMs), so this is a flat list rather than anything sorted or grouped.
     *
     * The trailing (0x0100, 0x01) of the original table is deliberately
     * absent; Imx258Cam::start() writes it, so that streaming begins after
     * the receiver ring is armed rather than before. */
    constexpr std::array<std::pair<uint16_t, uint8_t>, BASE_SEQUENCE_SIZE>
        IMX258_BASE_SEQUENCE = {{
        { 0x0100, 0x00 }, // streaming off while we reprogram
        { 0x0136, 0x1B }, { 0x0137, 0x00 }, // EXCK/INCK = 27.0 MHz
        { 0x0105, 0x00 }, { 0x0106, 0x00 },
        { 0x0114, 0x03 }, // 4 MIPI lanes
        { 0x0220, 0x00 }, { 0x0221, 0x11 }, { 0x0222, 0x01 },
        { 0x0340, 0x04 }, { 0x0341, 0x5C }, // VTS = 1116
        { 0x0342, 0x15 }, { 0x0343, 0x18 }, // HTS = 5400
        { 0x0344, 0x00 }, { 0x0345, 0x00 }, { 0x0346, 0x00 }, { 0x0347, 0x00 },
        { 0x0348, 0x0E }, { 0x0349, 0xFF }, { 0x034A, 0x08 }, { 0x034B, 0x6F }, // crop to 3839x2159
        { 0x0381, 0x01 }, { 0x0383, 0x01 }, { 0x0385, 0x01 }, { 0x0387, 0x01 },
        { 0x0900, 0x00 }, { 0x0901, 0x12 }, { 0x0902, 0x00 }, // binning H=1 V=2
        { 0x0112, 0x0A }, { 0x0113, 0x0A }, // CSI data format RAW10/RAW10
        { 0x034C, 0x07 }, { 0x034D, 0x80 }, // X_OUT_SIZE = 1920
        { 0x034E, 0x04 }, { 0x034F, 0x38 }, // Y_OUT_SIZE = 1080
        { 0x0401, 0x00 }, { 0x0408, 0x00 }, { 0x0409, 0x00 }, { 0x040A, 0x00 }, { 0x040B, 0x00 },
        { 0x040C, 0x07 }, { 0x040D, 0x80 }, { 0x040E, 0x04 }, { 0x040F, 0x38 }, // digital crop
        { 0x0301, 0x05 }, { 0x0303, 0x02 }, { 0x0305, 0x04 }, { 0x0306, 0x00 },
        { 0x0307, 0x6E }, // VT PLL multiplier 110 -> measured ~49.3 Hz
        { 0x0309, 0x0A }, { 0x030B, 0x02 }, { 0x0310, 0x01 }, { 0x030D, 0x02 },
        { 0x030E, 0x00 }, { 0x030F, 0x37 },
        { 0x0820, 0x05 }, { 0x0821, 0xCD }, { 0x0822, 0x00 }, { 0x0823, 0x00 }, // 1485 Mbps / 4 lanes
        { 0x0B06, 0x01 }, { 0x5A5C, 0x01 },
        { 0x0202, 0x06 }, { 0x0203, 0x40 }, // coarse integration time 1600
        { 0x0224, 0x01 }, { 0x0225, 0x80 },
        { 0x0204, 0x01 }, { 0x0205, 0xC0 }, // analog gain
        { 0x020E, 0x03 }, { 0x020F, 0x00 }, { 0x0210, 0x04 }, { 0x0211, 0x00 },
        { 0x0212, 0x05 }, { 0x0213, 0x00 }, { 0x0214, 0x03 }, { 0x0215, 0x00 },
        { 0x0216, 0x00 }, { 0x0217, 0x00 },
        { 0x4040, 0x00 }, { 0x0350, 0x00 }, { 0x0808, 0x00 },
        // Disable auxiliary phase-detection output; RX bypass expects only RAW10 image packets.
        // Without these the receiver sees interleaved PDAF packets and the image tears.
        { 0x3030, 0x00 }, { 0x3032, 0x00 }, { 0x7BCD, 0x00 },
    }};
    static_assert(IMX258_BASE_SEQUENCE.size() == BASE_SEQUENCE_SIZE,
        "IMX258_BASE_SEQUENCE size mismatch");

    /* 1920x1080 @ 60 Hz at 1194.75 Mb/s per lane: the Lattice IMX258 reference driver's
     * complete mode, transcribed from CAM_CONFIG_REFERENCE_1080P60_1194MBPS in
     * tools/hololink_pcie_bringup/examples/igx_imx258_hdmi/camera_hdmi_setup.py -- the
     * profile the CertusPro-NX Versa bringup streams with (cpnx_versa_imx258/board.py
     * DEFAULT_CAMERA_PROFILE). Built the way the reference's _camera_profile() builds it:
     * the base table with these registers overridden in place, and the registers the
     * base table lacks (0x0404/0x0405 scaler, 0x3038/0x303A/0x303B) appended. The base
     * table alone runs the output PLL at 371.25 Mb/s/lane (0x0820..0x0823 = 0x05CD0000,
     * 1485 Mb/s over four lanes), which the Versa's dynamic MIPI receiver does not lock
     * to (the sensor streams, PACK_SIF_TVLD_STAT never moves); this profile drives
     * 0x12AB0000 = 4779 Mb/s over four lanes. Output geometry is unchanged -- 1920x1080
     * RAW10, RAW10_START_BYTE 640 (the Versa measures valid_bytes = 640 + 1080 * 2400). */
    constexpr std::array<std::pair<uint16_t, uint8_t>, 32> IMX258_1194MBPS_OVERRIDES = {{
        { 0x0340, 0x05 }, { 0x0341, 0xD0 }, // VTS = 1488
        { 0x0342, 0x14 }, { 0x0343, 0xE8 }, // HTS = 5352
        { 0x0348, 0x10 }, { 0x0349, 0x6F }, { 0x034A, 0x0C }, { 0x034B, 0x2F }, // crop end 4207x3119
        { 0x0900, 0x01 }, { 0x0901, 0x12 }, // 2x2 binning on
        { 0x0401, 0x01 }, { 0x0404, 0x00 }, { 0x0405, 0x20 }, // scaler on, 32/32
        { 0x0408, 0x00 }, { 0x0409, 0xB8 }, { 0x040A, 0x00 }, { 0x040B, 0xF0 }, // digital crop origin
        { 0x040C, 0x0F }, { 0x040D, 0x08 }, // digital crop width 3848
        { 0x3038, 0x00 }, { 0x303A, 0x00 }, { 0x303B, 0x00 },
        { 0x0307, 0xB1 }, { 0x030B, 0x01 }, { 0x030D, 0x03 }, { 0x030F, 0x85 }, // VT PLL 177, OP PLL 3/389
        { 0x0820, 0x12 }, { 0x0821, 0xAB }, { 0x0822, 0x00 }, { 0x0823, 0x00 }, // 4779 Mbps / 4 lanes
        { 0x0202, 0x03 }, { 0x0203, 0x20 }, // exposure 800 lines
    }};

    constexpr size_t count_registers_missing_from_base_table()
    {
        size_t missing = 0;
        for (const auto& override_entry : IMX258_1194MBPS_OVERRIDES) {
            bool present = false;
            for (const auto& base_entry : IMX258_BASE_SEQUENCE) {
                present = present || (base_entry.first == override_entry.first);
            }
            missing += present ? 0 : 1;
        }
        return missing;
    }
    constexpr size_t MODE_1920X1080_60FPS_1194MBPS_SEQUENCE_SIZE
        = BASE_SEQUENCE_SIZE + count_registers_missing_from_base_table();
    static_assert(count_registers_missing_from_base_table() == 5,
        "the 1194 Mb/s profile adds exactly 0x0404, 0x0405, 0x3038, 0x303A, 0x303B");

    constexpr std::array<std::pair<uint16_t, uint8_t>, MODE_1920X1080_60FPS_1194MBPS_SEQUENCE_SIZE>
        make_imx258_mode_1920x1080_60fps_1194mbps_sequence()
    {
        // std::pair's assignment is not constexpr in C++17, so members are set one at a time.
        std::array<std::pair<uint16_t, uint8_t>, MODE_1920X1080_60FPS_1194MBPS_SEQUENCE_SIZE> sequence {};
        size_t n = 0;
        // The base table in order, each register overridden in place when the profile names it.
        for (const auto& base_entry : IMX258_BASE_SEQUENCE) {
            uint8_t value = base_entry.second;
            for (const auto& override_entry : IMX258_1194MBPS_OVERRIDES) {
                if (override_entry.first == base_entry.first) {
                    value = override_entry.second;
                }
            }
            sequence[n].first = base_entry.first;
            sequence[n].second = value;
            n++;
        }
        // Then the registers the base table never wrote, in profile order.
        for (const auto& override_entry : IMX258_1194MBPS_OVERRIDES) {
            bool present = false;
            for (const auto& base_entry : IMX258_BASE_SEQUENCE) {
                present = present || (base_entry.first == override_entry.first);
            }
            if (!present) {
                sequence[n].first = override_entry.first;
                sequence[n].second = override_entry.second;
                n++;
            }
        }
        return sequence;
    }
    constexpr std::array<std::pair<uint16_t, uint8_t>, MODE_1920X1080_60FPS_1194MBPS_SEQUENCE_SIZE>
        IMX258_MODE_1920X1080_60FPS_1194MBPS_SEQUENCE = make_imx258_mode_1920x1080_60fps_1194mbps_sequence();
    static_assert(IMX258_MODE_1920X1080_60FPS_1194MBPS_SEQUENCE.back().first == 0x303B,
        "the appended registers end the 1194 Mb/s table");
// clang-format on

} // namespace hololink::module::sensors::imx258

#endif // HOLOLINK_MODULE_SENSORS_IMX258_MODE_HPP
