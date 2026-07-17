/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_SENSORS_IMX274_MODE_HPP
#define HOLOLINK_MODULE_SENSORS_IMX274_MODE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "hololink/module/csi_converter.hpp"

namespace hololink::module::sensors::imx274 {

// The CSI format/converter types live in hololink::module::csi (they are
// not sensor types); alias them here so this driver's code can name them
// unqualified.
using ::hololink::module::csi::BayerFormat;
using ::hololink::module::csi::CsiConverterV1;
using ::hololink::module::csi::PixelFormat;

enum class Imx274_Mode : uint32_t {
    IMX274_MODE_3840X2160_60FPS = 0,
    IMX274_MODE_1920X1080_60FPS = 1,
    IMX274_MODE_3840X2160_60FPS_12BITS = 2,
    Unknown = 3,
};

struct FrameFormat {
    uint32_t width;
    uint32_t height;
    uint32_t framerate;
    PixelFormat pixel_format;
};

inline constexpr std::array<FrameFormat, 3> IMX_FRAME_FORMAT { {
    FrameFormat { 3840, 2160, 60, PixelFormat::RAW_10 },
    FrameFormat { 1920, 1080, 60, PixelFormat::RAW_10 },
    FrameFormat { 3840, 2160, 60, PixelFormat::RAW_12 },
} };

// IMX274 register tables, vendored byte-for-byte from the legacy
// src/hololink/sensors/camera/imx274/imx274_mode.hpp so the module
// driver carries no dependency on the legacy sensor library. Data only.
// Register addresses
static constexpr uint16_t REG_EXP_MSB = 0x300D;
static constexpr uint16_t REG_EXP_LSB = 0x300C;
static constexpr uint16_t REG_AG_MSB = 0x300B;
static constexpr uint16_t REG_AG_LSB = 0x300A;
static constexpr uint16_t REG_DG = 0x3012;

static constexpr uint16_t REG_TEST_ENABLE = 0x303C;
static constexpr uint16_t REG_TEST_PATTERN = 0x303D;
static constexpr uint16_t REG_TEST_SETUP1 = 0x377F;
static constexpr uint16_t REG_TEST_SETUP2 = 0x3781;
static constexpr uint16_t REG_TEST_SETUP3 = 0x370B;
static constexpr uint16_t REG_TEST_SETUP4 = 0x370E;

// Special register values
static constexpr uint16_t IMX274_TABLE_WAIT_MS = 0xFFFF; // Special value to indicate wait
static constexpr uint8_t IMX274_WAIT_MS = 0x01; // Wait time in ms
static constexpr uint8_t IMX274_WAIT_MS_START = 0x0F; // Wait time in ms for start sequence

// Array size constants needed since C++17 doesn't support std::array size deduction with std::pair
constexpr size_t START_SEQUENCE_SIZE = 6;
constexpr size_t STOP_SEQUENCE_SIZE = 2;
constexpr size_t MODE_3840X2160_60FPS_SEQUENCE_SIZE = 83;
constexpr size_t MODE_1920X1080_60FPS_SEQUENCE_SIZE = 84;
constexpr size_t MODE_3840X2160_60FPS_12BITS_SEQUENCE_SIZE = 83;

// clang-format off
    // Declare the arrays
    constexpr std::array<std::pair<uint16_t, uint8_t>, START_SEQUENCE_SIZE> IMX274_START_SEQUENCE = {{
        { 0x3000, 0x00 }, // Mode select streaming on
        { 0x303E, 0x02 },
        { IMX274_TABLE_WAIT_MS, IMX274_WAIT_MS_START },
        { 0x30F4, 0x00 },
        { 0x3018, 0xA2 },
        { IMX274_TABLE_WAIT_MS, IMX274_WAIT_MS_START },
    }};
    static_assert(IMX274_START_SEQUENCE.size() == START_SEQUENCE_SIZE, "IMX274_START_SEQUENCE size mismatch");

    constexpr std::array<std::pair<uint16_t, uint8_t>, STOP_SEQUENCE_SIZE> IMX274_STOP_SEQUENCE = {{
        { IMX274_TABLE_WAIT_MS, IMX274_WAIT_MS },
        { 0x3000, 0x01 }, // Mode select streaming off
    }};
    static_assert(IMX274_STOP_SEQUENCE.size() == STOP_SEQUENCE_SIZE, "IMX274_STOP_SEQUENCE size mismatch");

    constexpr std::array<std::pair<uint16_t, uint8_t>, MODE_3840X2160_60FPS_SEQUENCE_SIZE>
        IMX274_MODE_3840X2160_60FPS_SEQUENCE = {{
        // Mode: 3840x2160 60fps RAW10
        { IMX274_TABLE_WAIT_MS, IMX274_WAIT_MS },
        { 0x3000, 0x12 },
        { 0x3120, 0xF0 },
        { 0x3122, 0x02 },
        { 0x3129, 0x9C },
        { 0x312A, 0x02 },
        { 0x312D, 0x02 },
        { 0x310B, 0x00 },
        { 0x304C, 0x00 },
        { 0x304D, 0x03 },
        { 0x331C, 0x1A },
        { 0x3502, 0x02 },
        { 0x3529, 0x0E },
        { 0x352A, 0x0E },
        { 0x352B, 0x0E },
        { 0x3538, 0x0E },
        { 0x3539, 0x0E },
        { 0x3553, 0x00 },
        { 0x357D, 0x05 },
        { 0x357F, 0x05 },
        { 0x3581, 0x04 },
        { 0x3583, 0x76 },
        { 0x3587, 0x01 },
        { 0x35BB, 0x0E },
        { 0x35BC, 0x0E },
        { 0x35BD, 0x0E },
        { 0x35BE, 0x0E },
        { 0x35BF, 0x0E },
        { 0x366E, 0x00 },
        { 0x366F, 0x00 },
        { 0x3670, 0x00 },
        { 0x3671, 0x00 },
        { 0x30EE, 0x01 },
        { 0x3304, 0x32 },
        { 0x3306, 0x32 },
        { 0x3590, 0x32 },
        { 0x3686, 0x32 },
        // Resolution settings
        { 0x30E2, 0x01 },
        { 0x30F6, 0x07 },
        { 0x30F7, 0x01 },
        { 0x30F8, 0xC6 },
        { 0x30F9, 0x11 },
        { 0x3130, 0x78 },
        { 0x3131, 0x08 },
        { 0x3132, 0x70 },
        { 0x3133, 0x08 },
        // Crop settings
        { 0x30DD, 0x01 },
        { 0x30DE, 0x04 },
        { 0x30E0, 0x03 },
        { 0x3037, 0x01 },
        { 0x3038, 0x0C },
        { 0x3039, 0x00 },
        { 0x303A, 0x0C },
        { 0x303B, 0x0F },
        // Mode settings
        { 0x3004, 0x01 },
        { 0x3005, 0x01 },
        { 0x3006, 0x00 },
        { 0x3007, 0x02 },
        { 0x300C, 0x0C },
        { 0x300D, 0x00 },
        { 0x300E, 0x00 },
        { 0x3019, 0x00 },
        { 0x3A41, 0x08 },
        { 0x3342, 0x0A },
        { 0x3343, 0x00 },
        { 0x3344, 0x16 },
        { 0x3345, 0x00 },
        { 0x3528, 0x0E },
        { 0x3554, 0x1F },
        { 0x3555, 0x01 },
        { 0x3556, 0x01 },
        { 0x3557, 0x01 },
        { 0x3558, 0x01 },
        { 0x3559, 0x00 },
        { 0x355A, 0x00 },
        { 0x35BA, 0x0E },
        { 0x366A, 0x1B },
        { 0x366B, 0x1A },
        { 0x366C, 0x19 },
        { 0x366D, 0x17 },
        { 0x33A6, 0x01 },
        { 0x306B, 0x05 },
        { IMX274_TABLE_WAIT_MS, IMX274_WAIT_MS },
    }};
    static_assert(IMX274_MODE_3840X2160_60FPS_SEQUENCE.size() == MODE_3840X2160_60FPS_SEQUENCE_SIZE, "IMX274_MODE_3840X2160_60FPS_SEQUENCE size mismatch");

    constexpr std::array<std::pair<uint16_t, uint8_t>, MODE_1920X1080_60FPS_SEQUENCE_SIZE>
        IMX274_MODE_1920X1080_60FPS_SEQUENCE = {{
        // Mode: 1920x1080 60fps RAW10
        { IMX274_TABLE_WAIT_MS, IMX274_WAIT_MS },
        { 0x3000, 0x12 },
        // Input freq. 24M
        { 0x3120, 0xF0 },
        { 0x3122, 0x02 },
        { 0x3129, 0x9C },
        { 0x312A, 0x02 },
        { 0x312D, 0x02 },
        { 0x310B, 0x00 },
        { 0x304C, 0x00 },
        { 0x304D, 0x03 },
        { 0x331C, 0x1A },
        { 0x3502, 0x02 },
        { 0x3529, 0x0E },
        { 0x352A, 0x0E },
        { 0x352B, 0x0E },
        { 0x3538, 0x0E },
        { 0x3539, 0x0E },
        { 0x3553, 0x00 },
        { 0x357D, 0x05 },
        { 0x357F, 0x05 },
        { 0x3581, 0x04 },
        { 0x3583, 0x76 },
        { 0x3587, 0x01 },
        { 0x35BB, 0x0E },
        { 0x35BC, 0x0E },
        { 0x35BD, 0x0E },
        { 0x35BE, 0x0E },
        { 0x35BF, 0x0E },
        { 0x366E, 0x00 },
        { 0x366F, 0x00 },
        { 0x3670, 0x00 },
        { 0x3671, 0x00 },
        { 0x30EE, 0x01 },
        { 0x3304, 0x32 },
        { 0x3306, 0x32 },
        { 0x3590, 0x32 },
        { 0x3686, 0x32 },
        // Resolution settings
        { 0x30E2, 0x02 },
        { 0x30F6, 0x04 },
        { 0x30F7, 0x01 },
        { 0x30F8, 0x0C },
        { 0x30F9, 0x12 },
        { 0x3130, 0x40 },
        { 0x3131, 0x04 },
        { 0x3132, 0x38 },
        { 0x3133, 0x04 },
        // Crop settings
        { 0x30DD, 0x01 },
        { 0x30DE, 0x07 },
        { 0x30DF, 0x00 },
        { 0x30E0, 0x04 },
        { 0x30E1, 0x00 },
        { 0x3037, 0x01 },
        { 0x3038, 0x0C },
        { 0x3039, 0x00 },
        { 0x303A, 0x0C },
        { 0x303B, 0x0F },
        // Mode settings
        { 0x3004, 0x02 },
        { 0x3005, 0x21 },
        { 0x3006, 0x00 },
        { 0x3007, 0xB1 },
        { 0x300C, 0x08 },
        { 0x300D, 0x00 },
        { 0x3019, 0x00 },
        { 0x3A41, 0x08 },
        { 0x3342, 0x0A },
        { 0x3343, 0x00 },
        { 0x3344, 0x1A },
        { 0x3345, 0x00 },
        { 0x3528, 0x0E },
        { 0x3554, 0x00 },
        { 0x3555, 0x01 },
        { 0x3556, 0x01 },
        { 0x3557, 0x01 },
        { 0x3558, 0x01 },
        { 0x3559, 0x00 },
        { 0x355A, 0x00 },
        { 0x35BA, 0x0E },
        { 0x366A, 0x1B },
        { 0x366B, 0x1A },
        { 0x366C, 0x19 },
        { 0x366D, 0x17 },
        { 0x33A6, 0x01 },
        { 0x306B, 0x05 },
        { IMX274_TABLE_WAIT_MS, IMX274_WAIT_MS },
    }};
    constexpr std::array<std::pair<uint16_t, uint8_t>, MODE_3840X2160_60FPS_12BITS_SEQUENCE_SIZE>
        IMX274_MODE_3840X2160_60FPS_12BITS_SEQUENCE = {{
        // Mode: 3840x2160 60fps RAW12
        { IMX274_TABLE_WAIT_MS, IMX274_WAIT_MS },
        { 0x3000, 0x12 },
        { 0x3120, 0xF0 },
        { 0x3122, 0x02 },
        { 0x3129, 0x9C },
        { 0x312A, 0x02 },
        { 0x312D, 0x02 },
        { 0x310B, 0x00 },
        { 0x304C, 0x00 },
        { 0x304D, 0x03 },
        { 0x331C, 0x1A },
        { 0x3502, 0x02 },
        { 0x3529, 0x0E },
        { 0x352A, 0x0E },
        { 0x352B, 0x0E },
        { 0x3538, 0x0E },
        { 0x3539, 0x0E },
        { 0x3553, 0x00 },
        { 0x357D, 0x05 },
        { 0x357F, 0x05 },
        { 0x3581, 0x04 },
        { 0x3583, 0x76 },
        { 0x3587, 0x01 },
        { 0x35BB, 0x0E },
        { 0x35BC, 0x0E },
        { 0x35BD, 0x0E },
        { 0x35BE, 0x0E },
        { 0x35BF, 0x0E },
        { 0x366E, 0x00 },
        { 0x366F, 0x00 },
        { 0x3670, 0x00 },
        { 0x3671, 0x00 },
        { 0x30EE, 0x01 },
        { 0x3304, 0x32 },
        { 0x3306, 0x32 },
        { 0x3590, 0x32 },
        { 0x3686, 0x32 },
        // Resolution settings
        { 0x30E2, 0x00 },
        { 0x30F6, 0xED },
        { 0x30F7, 0x01 },
        { 0x30F8, 0x08 },
        { 0x30F9, 0x13 },
        { 0x3130, 0x94 },
        { 0x3131, 0x08 },
        { 0x3132, 0x70 },
        { 0x3133, 0x08 },
        // Crop settings
        { 0x30DD, 0x01 },
        { 0x30DE, 0x04 },
        { 0x30E0, 0x03 },
        { 0x3037, 0x01 },
        { 0x3038, 0x0C },
        { 0x3039, 0x00 },
        { 0x303A, 0x0C },
        { 0x303B, 0x0F },
        // Mode settings
        { 0x3004, 0x00 },
        { 0x3005, 0x07 },
        { 0x3006, 0x00 },
        { 0x3007, 0x02 },
        { 0x300C, 0x0C },
        { 0x300D, 0x00 },
        { 0x300E, 0x00 },
        { 0x3019, 0x00 },
        { 0x3A41, 0x10 },
        { 0x3342, 0xFF },
        { 0x3343, 0x01 },
        { 0x3344, 0xFF },
        { 0x3345, 0x01 },
        { 0x3528, 0x0F },
        { 0x3554, 0x00 },
        { 0x3555, 0x00 },
        { 0x3556, 0x00 },
        { 0x3557, 0x00 },
        { 0x3558, 0x00 },
        { 0x3559, 0x1F },
        { 0x355A, 0x1F },
        { 0x35BA, 0x0F },
        { 0x366A, 0x00 },
        { 0x366B, 0x00 },
        { 0x366C, 0x00 },
        { 0x366D, 0x00 },
        { 0x33A6, 0x01 },
        { 0x306B, 0x07 },
        { IMX274_TABLE_WAIT_MS, IMX274_WAIT_MS },
    }};
    static_assert(IMX274_MODE_3840X2160_60FPS_12BITS_SEQUENCE.size() == MODE_3840X2160_60FPS_12BITS_SEQUENCE_SIZE, "IMX274_MODE_3840X2160_60FPS_12BITS_SEQUENCE size mismatch");
// clang-format on

} // namespace hololink::module::sensors::imx274

#endif // HOLOLINK_MODULE_SENSORS_IMX274_MODE_HPP
