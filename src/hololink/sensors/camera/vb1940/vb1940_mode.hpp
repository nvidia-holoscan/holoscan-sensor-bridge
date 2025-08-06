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

#ifndef SENSORS_CAMERA_VB1940_VB1940_MODE_HPP
#define SENSORS_CAMERA_VB1940_VB1940_MODE_HPP

#include "hololink/sensors/camera/camera_mode.hpp"
#include <array>
#include <utility>
#include <vector>

namespace hololink::sensors {

namespace vb1940_mode {

    // values are on hex number system to be consistent with rest of the list
    static constexpr uint16_t VB1940_TABLE_WAIT_MS = 0xFFFF; // Special value to indicate wait
    static constexpr uint8_t VB1940_WAIT_MS = 0x0A; // Wait time in ms
    static constexpr uint8_t VB1940_WAIT_MS_START = 0x64; // Wait time in ms for start sequence

    static constexpr uint32_t VB1940_PAGE_SIZE = 128;

    // Register addresses
    static constexpr uint16_t REG_AG = 0x0CA1; // ctx1
    static constexpr uint16_t REG_EXP = 0x0CA2; // ctx1

    // Array size constants
    constexpr size_t START_SEQUENCE_SIZE = 2;
    constexpr size_t STOP_SEQUENCE_SIZE = 2;

    // Start/Stop sequences
    constexpr std::array<std::pair<uint16_t, uint8_t>, START_SEQUENCE_SIZE> VB1940_START_SEQUENCE = { {
        { 0x516, 0x01 }, // start streaming
        { VB1940_TABLE_WAIT_MS, VB1940_WAIT_MS_START },
    } };

    constexpr std::array<std::pair<uint16_t, uint8_t>, STOP_SEQUENCE_SIZE> VB1940_STOP_SEQUENCE = { {
        { 0x517, 0x01 }, // stop streaming
        { VB1940_TABLE_WAIT_MS, VB1940_WAIT_MS },
    } };

    static constexpr uint32_t VB1940_CERTIFICATE_START_ADDR = 0x1AA8;
    extern const std::vector<uint8_t> VB1940_CERTIFICATE;

    static constexpr uint32_t VB1940_FWP_START_ADDR = 0x2000;
    extern const std::vector<uint8_t> VB1940_FW;

    static constexpr uint32_t LDEC_RAM_CONTENT_START_ADDR = 0x2000;
    extern const std::vector<uint8_t> LDEC_RAM_CONTENT;

    static constexpr uint32_t RD_RAM_SEQ_1_CONTENT_START_ADDR = 0x5000;
    extern const std::vector<uint8_t> RD_RAM_SEQ_1_CONTENT;

    static constexpr uint32_t GT_RAM_PAT_CONTENT_START_ADDR = 0x5BC0;
    extern const std::vector<uint8_t> GT_RAM_PAT_CONTENT;

    static constexpr uint32_t GT_RAM_SEQ_1_CONTENT_START_ADDR = 0x5E40;
    extern const std::vector<uint8_t> GT_RAM_SEQ_1_CONTENT;

    static constexpr uint32_t GT_RAM_SEQ_2_CONTENT_START_ADDR = 0x5F80;
    extern const std::vector<uint8_t> GT_RAM_SEQ_2_CONTENT;

    static constexpr uint32_t GT_RAM_SEQ_3_CONTENT_START_ADDR = 0x60C0;
    extern const std::vector<uint8_t> GT_RAM_SEQ_3_CONTENT;

    static constexpr uint32_t GT_RAM_SEQ_4_CONTENT_START_ADDR = 0x6160;
    extern const std::vector<uint8_t> GT_RAM_SEQ_4_CONTENT;

    static constexpr uint32_t RD_RAM_PAT_CONTENT_START_ADDR = 0x6AC0;
    extern const std::vector<uint8_t> RD_RAM_PAT_CONTENT;

    extern const std::vector<std::pair<uint16_t, uint8_t>> VB1940_MODE_2560X1984_30FPS_SEQUENCE;
    extern const std::vector<std::pair<uint16_t, uint8_t>> VB1940_MODE_1920X1080_30FPS_SEQUENCE;
    extern const std::vector<std::pair<uint16_t, uint8_t>> VB1940_MODE_2560X1984_30FPS_8BIT_SEQUENCE;

    enum Mode {
        VB1940_MODE_2560X1984_30FPS = 0,
        VB1940_MODE_1920X1080_30FPS = 1,
        VB1940_MODE_2560X1984_30FPS_8BIT = 2,
        VB1940_MODE_COUNT = 3,
    };
} // namespace vb1940_mode

class Vb1940FrameFormat : public CameraFrameFormat {
public:
    Vb1940FrameFormat(CameraMode mode_id, const std::string& mode_name, int64_t width,
        int64_t height, double frame_rate, csi::PixelFormat pixel_format);
    ~Vb1940FrameFormat() override = default;
};

} // namespace hololink::sensors

#endif /* SENSORS_CAMERA_VB1940_VB1940_MODE_HPP */