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

// How much the HSB's packetizer sends for a line, per the program loaded into it. The
// sizes come from the program's schedule (fpga/nv_hsb_ip/packetizer/packetizer_top.sv): a
// line completes a whole loop of the schedule, then is padded to 64 bytes. No board.

#include <gtest/gtest.h>

#include <hololink/core/csi_formats.hpp>
#include <hololink/core/packetizer_program.hpp>

namespace {

using hololink::csi::PixelFormat;

TEST(PacketizerProgram, Raw10LinesCompleteWholeLoops)
{
    // Schedule 0x11f1fff7: 15 beats (120 bytes) in per 16 beats (128 bytes) out.
    auto program = hololink::csi::get_packetizer_program(PixelFormat::RAW_10);
    ASSERT_TRUE(program);
    EXPECT_TRUE(program->packetizes());
    // A partial loop still emits all 16 beats.
    EXPECT_EQ(program->get_output_size(1), 128u);
    EXPECT_EQ(program->get_output_size(5), 128u);
    EXPECT_EQ(program->get_output_size(120), 128u);
    EXPECT_EQ(program->get_output_size(121), 256u);
    // The IMX274's 1080p and 4K lines are whole loops, so these match 3 pixels per 4 bytes...
    EXPECT_EQ(program->get_output_size(1920 * 5 / 4), 2560u);
    EXPECT_EQ(program->get_output_size(3840 * 5 / 4), 5120u);
    // ...but a 720p line (1600 CSI-2 bytes, 13.33 loops) does not: 14 loops, not
    // ceil(1280 / 3) * 4 = 1708 rounded up to 1728.
    EXPECT_EQ(program->get_output_size(1280 * 5 / 4), 1792u);
    EXPECT_EQ(program->padded_output_size(1280 * 5 / 4), 1792u);
    // The 175-byte embedded-data line: 2 loops, already 64-aligned.
    EXPECT_EQ(program->padded_output_size(175), 256u);
}

TEST(PacketizerProgram, Raw12LinesRoundToTheirLoop)
{
    // Schedule 0x11210007: 3 beats (24 bytes) in per 3 out.
    auto program = hololink::csi::get_packetizer_program(PixelFormat::RAW_12);
    ASSERT_TRUE(program);
    EXPECT_TRUE(program->packetizes());
    EXPECT_EQ(program->get_output_size(6), 24u);
    EXPECT_EQ(program->get_output_size(24), 24u);
    EXPECT_EQ(program->get_output_size(25), 48u);
    // A 1280-pixel RAW_12 line is 80 loops exactly; 1288 pixels is 80.5.
    EXPECT_EQ(program->get_output_size(1280 * 3 / 2), 1920u);
    EXPECT_EQ(program->get_output_size(1288 * 3 / 2), 1944u);
    EXPECT_EQ(program->padded_output_size(1288 * 3 / 2), 1984u);
}

// Every IMX274 and VB1940 mode the packetizer can pack keeps the padded line size the
// pixel arithmetic these programs replaced gave it (3 RAW_10 pixels per 4 bytes, 2 RAW_12
// pixels per 3, rounded up to 64): the lines are whole loops, or (VB1940 2560 RAW_10, 26.67
// loops) round to the same 64-byte boundary.
TEST(PacketizerProgram, SensorModesKeepTheirPaddedLineSize)
{
    struct Mode {
        const char* name;
        PixelFormat format;
        uint32_t width;
        uint32_t padded_line_bytes;
    };
    const Mode modes[] = {
        { "IMX274 3840x2160 RAW_10", PixelFormat::RAW_10, 3840, 5120 },
        { "IMX274 1920x1080 RAW_10", PixelFormat::RAW_10, 1920, 2560 },
        { "IMX274 3840x2160 RAW_12", PixelFormat::RAW_12, 3840, 5760 },
        { "VB1940 2560x1984 RAW_10", PixelFormat::RAW_10, 2560, 3456 },
        { "VB1940 1920x1080 RAW_10", PixelFormat::RAW_10, 1920, 2560 },
    };
    for (const auto& mode : modes) {
        const uint32_t csi_line_bytes = mode.format == PixelFormat::RAW_10
            ? mode.width * 5 / 4
            : mode.width * 3 / 2;
        auto program = hololink::csi::get_packetizer_program(mode.format);
        EXPECT_EQ(program->padded_output_size(csi_line_bytes), mode.padded_line_bytes) << mode.name;
    }
}

TEST(PacketizerProgram, NullProgramNeitherResizesNorPackets)
{
    auto program = hololink::csi::get_packetizer_program(PixelFormat::RAW_8);
    ASSERT_TRUE(program);
    EXPECT_FALSE(program->packetizes());
    EXPECT_EQ(program->get_output_size(1280), 1280u);
    // padded_output_size() describes an enabled packetizer; the null program never is.
}

} // namespace
