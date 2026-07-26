/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOLINK_MODULE_SENSORS_AR0234_AR0234_MODE_HPP
#define HOLOLINK_MODULE_SENSORS_AR0234_AR0234_MODE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <hololink/core/csi_formats.hpp>
#include <utility>

namespace hololink::module::sensors::ar0234 {

namespace ar0234_mode {

    enum Mode {
        AR0234_MODE_1920X1200_60FPS = 0,
        Unknown = 1,
        AR0234_MODE_COUNT = 2,
    };

    // Register addresses for camera properties.
    static constexpr uint16_t REG_AG = 0x3060; // Analog Gain
    static constexpr uint16_t REG_EXP = 0x3012; // Exposure
    static constexpr uint16_t REG_TP = 0x3070; // Test Pattern

    // Special register values
    static constexpr uint16_t AR0234_TABLE_WAIT_MS = 0xFFFF; // Special value to indicate wait
    static constexpr uint16_t AR0234_WAIT_MS = 0x01; // Wait time in ms
    static constexpr uint16_t AR0234_WAIT_MS_START = 0xC8; // Wait time in ms for start sequence

    // Array size constants needed since C++17 doesn't support std::array size deduction with std::pair
    constexpr size_t START_SEQUENCE_SIZE = 2;
    constexpr size_t STOP_SEQUENCE_SIZE = 2;
    constexpr size_t START_FSYNC_SEQUENCE_SIZE = 3;
    constexpr size_t MODE_1920X1200_60FPS_SEQUENCE_SIZE = 33;

    // clang-format off
// Stream start (no FSYNC): master mode, kick off streaming.
constexpr std::array<std::pair<uint16_t, uint16_t>, START_SEQUENCE_SIZE> AR0234_START_SEQUENCE = {{
    { 0x301A, 0x205C },
    { AR0234_TABLE_WAIT_MS, AR0234_WAIT_MS_START },
}};
static_assert(AR0234_START_SEQUENCE.size() == START_SEQUENCE_SIZE, "AR0234_START_SEQUENCE size mismatch");

// Stream stop: streaming off.
constexpr std::array<std::pair<uint16_t, uint16_t>, STOP_SEQUENCE_SIZE> AR0234_STOP_SEQUENCE = {{
    { 0x301A, 0x2058 },
    { AR0234_TABLE_WAIT_MS, AR0234_WAIT_MS },
}};
static_assert(AR0234_STOP_SEQUENCE.size() == STOP_SEQUENCE_SIZE, "AR0234_STOP_SEQUENCE size mismatch");

// Stream start (sync-sink / external trigger mode).
constexpr std::array<std::pair<uint16_t, uint16_t>, START_FSYNC_SEQUENCE_SIZE> AR0234_START_FSYNC_SEQUENCE = {{
    { 0x30CE, 0x0100 },
    { 0x301A, 0x295C },
    { AR0234_TABLE_WAIT_MS, AR0234_WAIT_MS_START },
}};
static_assert(AR0234_START_FSYNC_SEQUENCE.size() == START_FSYNC_SEQUENCE_SIZE, "AR0234_START_FSYNC_SEQUENCE size mismatch");

// AR0234 Full resolution: 1920x1200 Output: 10bit Master Mode 60fps
// MIPI4Lane_1920x1200@60fps_Pxlclk45MHz_Extclk27MHz
constexpr std::array<std::pair<uint16_t, uint16_t>, MODE_1920X1200_60FPS_SEQUENCE_SIZE>
    AR0234_MODE_1920X1200_60FPS_SEQUENCE = {{
        { AR0234_TABLE_WAIT_MS, AR0234_WAIT_MS },
        // pxlclk45_extclk27 PLL setting
        { 0x302A, 0x0005 },
        { 0x302C, 0x0001 },
        { 0x302E, 0x0003 },
        { 0x3030, 0x0019 },
        { 0x3036, 0x000A },
        { 0x3038, 0x0001 },
        { 0x30B0, 0x0028 }, // enable PLL (30B0[14]=0)
        { 0x31B0, 0x0082 }, // FRAME_PREAMBLE
        { 0x31B2, 0x005C }, // LINE_PREAMBLE
        // mipi setting
        { 0x31B4, 0x51C8 },
        { 0x31B6, 0x3257 },
        { 0x31B8, 0x904B },
        { 0x31BA, 0x030B },
        { 0x31BC, 0x8E09 },
        { 0x3354, 0x002B }, // mipi_cntrl
        //
        { 0x31AC, 0x0A0A }, // in-out format: in 10bit, out 10bit
        { 0x31AE, 0x0204 }, // 4 lane
        // crop
        { 0x3002, 0x0008 }, // y_addr_start
        { 0x3004, 0x0008 }, // x_addr_start
        { 0x3006, 0x04B7 }, // y_addr_end
        { 0x3008, 0x0787 }, // x_addr_end
        { 0x30A2, 0x0001 }, // x_odd_inc
        { 0x30A6, 0x0001 }, // y_odd_inc
        { 0x3040, 0x0000 }, // read_mode
        { 0x3064, 0x1802 }, // smia_test (default 0x1802; embedded data disabled)
        { 0x300A, 0x04C4 }, // FRAME_LENGTH_LINES
        { 0x300C, 0x0264 }, // LINE_LENGTH_PCK
        { REG_EXP, 0x02DC }, // COARSE_INTEGRATION_TIME
        { 0x31D0, 0x0000 },
        { 0x3786, 0x0006 }, // digital_ctrl_1
        { REG_TP, 0x0000 }, // disable test pattern
        { AR0234_TABLE_WAIT_MS, AR0234_WAIT_MS },
    }};
static_assert(AR0234_MODE_1920X1200_60FPS_SEQUENCE.size() == MODE_1920X1200_60FPS_SEQUENCE_SIZE,
    "AR0234_MODE_1920X1200_60FPS_SEQUENCE size mismatch");
    // clang-format on

    struct FrameFormat {
        uint32_t width;
        uint32_t height;
        uint32_t framerate;
        hololink::csi::PixelFormat pixel_format;
    };

    constexpr FrameFormat AR0234_1920X1200_60FPS_FORMAT = {
        1920, 1200, 60, hololink::csi::PixelFormat::RAW_10
    };

} // namespace ar0234_mode

} // namespace hololink::module::sensors::ar0234

#endif /* HOLOLINK_MODULE_SENSORS_AR0234_AR0234_MODE_HPP */
