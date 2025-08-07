/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SENSORS_CAMERA_IMX274_IMX274_MODE_HPP
#define SENSORS_CAMERA_IMX274_IMX274_MODE_HPP

#include "hololink/sensors/camera/camera_mode.hpp"
#include <array>
#include <utility>
#include <vector>

namespace hololink::sensors {

namespace imx274_mode {
    enum Mode {
        IMX274_MODE_3840X2160_60FPS = 0,
        IMX274_MODE_1920X1080_60FPS = 1,
        IMX274_MODE_3840X2160_60FPS_12BITS = 2,
        IMX274_MODE_COUNT = 3,
    };

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
    constexpr size_t MODE_1920X1080_60FPS_SEQUENCE_SIZE = 87;
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
} // namespace imx274_mode

class Imx274FrameFormat : public CameraFrameFormat {
public:
    Imx274FrameFormat(CameraMode mode_id, const std::string& mode_name, int64_t width,
        int64_t height, double frame_rate, csi::PixelFormat pixel_format);

    const Format& format() const override;
};

} // namespace hololink::sensors

#endif /* SENSORS_CAMERA_IMX274_IMX274_MODE_HPP */
