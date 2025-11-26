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

#include "packetizer_program.hpp"

namespace hololink {

// Packetizer configuration SIF offsets
constexpr uint32_t PACKETIZER_MODE = 0x0C;
constexpr uint32_t PACKETIZER_RAM = 0x04;
constexpr uint32_t PACKETIZER_DATA = 0x08;

/*******************************************************************************
 PacketizerProgram base implementation
*******************************************************************************/

PacketizerProgram::PacketizerProgram()
{
}

PacketizerProgram::~PacketizerProgram()
{
}

uint32_t PacketizerProgram::get_output_size(uint32_t input_size)
{
    // Default is that packetizer program does not resize the data (i.e. just swizzles).
    return input_size;
}

void PacketizerProgram::disable(Hololink& hololink, uint32_t sif_address)
{
    hololink.write_uint32(sif_address + PACKETIZER_MODE, 0);
}

/*******************************************************************************
 NullPacketizerProgram
*******************************************************************************/

void NullPacketizerProgram::enable(Hololink& hololink, uint32_t sif_address)
{
    disable(hololink, sif_address);
}

/*******************************************************************************
 CSI Packetizing Programs
*******************************************************************************/

std::shared_ptr<PacketizerProgram> csi::get_packetizer_program(csi::PixelFormat pixel_format)
{
    switch (pixel_format) {
    case csi::PixelFormat::RAW_10:
        return std::make_shared<Csi10ToPacked10>();
    case csi::PixelFormat::RAW_12:
        return std::make_shared<Csi12ToPacked12>();
    default:
        return std::make_shared<NullPacketizerProgram>();
    }
}

/*******************************************************************************
 Csi10ToPacked10
*******************************************************************************/

uint32_t Csi10ToPacked10::get_output_size(uint32_t input_size)
{
    // Input is tightly-packed 10-bit CSI data, output is 3 pixels per 4 bytes
    // (including 2 bits of padding).
    uint32_t num_pixels = (input_size * 8) / 10;
    return ((num_pixels + 2) / 3) * 4;
}

void Csi10ToPacked10::enable(Hololink& hololink, uint32_t sif_address)
{
    hololink.write_uint32(sif_address + PACKETIZER_MODE, 0x11f1fff7);
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000000); // RAM: 0 ELEMENT:0
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x2AAAAAAA); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000010); // RAM: 1 ELEMENT:0
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x2A9AA6AA); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000020); // RAM: 2 ELEMENT:0
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x02020000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000040); // RAM: 4 ELEMENT:0
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x05000000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000050); // RAM: 5 ELEMENT:0
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00001865); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000060); // RAM: 6 ELEMENT:0
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x20300000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000070); // RAM: 7 ELEMENT:0
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x1C007061); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000090); // RAM: 9 ELEMENT:0
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x40044400); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000a0); // RAM: 10 ELEMENT:0
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x50000600); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000b0); // RAM: 11 ELEMENT:0
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x20802080); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000c0); // RAM: 12 ELEMENT:0
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x9362E000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000d0); // RAM: 13 ELEMENT:0
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x01B18626); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000140); // RAM: 20 ELEMENT:0
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00000001); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000001); // RAM: 0 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x2AAAABFE); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000011); // RAM: 1 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x2AAAAAAA); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000081); // RAM: 8 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00010000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000091); // RAM: 9 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x440CCC81); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000a1); // RAM: 10 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x01555600); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000b1); // RAM: 11 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x18006085); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000c1); // RAM: 12 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x9362E000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000d1); // RAM: 13 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x01B10724); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000f1); // RAM: 15 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00222200); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000101); // RAM: 16 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xA802A800); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000111); // RAM: 17 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xE0000002); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000121); // RAM: 18 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x6DB0F000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000131); // RAM: 19 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x0000C001); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000141); // RAM: 20 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00000001); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000002); // RAM: 0 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x2AAAABFE); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000012); // RAM: 1 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x2A9AA6AA); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000022); // RAM: 2 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x01000000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000032); // RAM: 3 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x11110000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000042); // RAM: 4 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x05000000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000052); // RAM: 5 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00001865); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000062); // RAM: 6 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x20300000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000072); // RAM: 7 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x1C007061); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000092); // RAM: 9 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00111100); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000a2); // RAM: 10 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xA9557800); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000b2); // RAM: 11 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xD80C668D); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000c2); // RAM: 12 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xDB63E001); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000d2); // RAM: 13 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x01C18022); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000f2); // RAM: 15 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00222200); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000102); // RAM: 16 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xA802A800); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000112); // RAM: 17 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xE0000002); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000122); // RAM: 18 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x6DB0F000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000132); // RAM: 19 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x0000C001); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000142); // RAM: 20 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00000001); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000003); // RAM: 0 ELEMENT:3
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x155AA6A9); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000013); // RAM: 1 ELEMENT:3
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x15555555); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000023); // RAM: 2 ELEMENT:3
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00010000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000033); // RAM: 3 ELEMENT:3
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00444401); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000043); // RAM: 4 ELEMENT:3
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x02AAA000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000053); // RAM: 5 ELEMENT:3
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x3C00C108); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000063); // RAM: 6 ELEMENT:3
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x9B7FC05C); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000073); // RAM: 7 ELEMENT:3
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x03624AC0); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000a3); // RAM: 10 ELEMENT:3
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x05555000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000b3); // RAM: 11 ELEMENT:3
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xC0000000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000c3); // RAM: 12 ELEMENT:3
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x013FE01F); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000d3); // RAM: 13 ELEMENT:3
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00018206); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000143); // RAM: 20 ELEMENT:3
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00000001); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000004); // RAM: 0 ELEMENT:4
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x155AA6A9); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000014); // RAM: 1 ELEMENT:4
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x15555555); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000024); // RAM: 2 ELEMENT:4
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00800000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000034); // RAM: 3 ELEMENT:4
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00444480); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000044); // RAM: 4 ELEMENT:4
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xF2A15000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000054); // RAM: 5 ELEMENT:4
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xB006840F); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000064); // RAM: 6 ELEMENT:4
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x0B7FC07F); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000074); // RAM: 7 ELEMENT:4
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x038344CC); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000a4); // RAM: 10 ELEMENT:4
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x05555000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000b4); // RAM: 11 ELEMENT:4
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xC0000000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000c4); // RAM: 12 ELEMENT:4
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x013FE01F); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000d4); // RAM: 13 ELEMENT:4
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00018206); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000144); // RAM: 20 ELEMENT:4
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00000001); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000005); // RAM: 0 ELEMENT:5
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x155AAAAA); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000015); // RAM: 1 ELEMENT:5
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x15555555); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000035); // RAM: 3 ELEMENT:5
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00444400); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000045); // RAM: 4 ELEMENT:5
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xA2A00000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000055); // RAM: 5 ELEMENT:5
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xF000820A); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000065); // RAM: 6 ELEMENT:5
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x64BA1FFF); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000075); // RAM: 7 ELEMENT:5
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x6360C8C3); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000085); // RAM: 8 ELEMENT:5
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x40400000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000095); // RAM: 9 ELEMENT:5
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x44226600); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000a5); // RAM: 10 ELEMENT:5
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x50501000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000b5); // RAM: 11 ELEMENT:5
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xE0C04145); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000c5); // RAM: 12 ELEMENT:5
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x06DF807F); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000d5); // RAM: 13 ELEMENT:5
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00003C78); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000145); // RAM: 20 ELEMENT:5
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00000001); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000006); // RAM: 0 ELEMENT:6
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x2A9AA6A9); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000016); // RAM: 1 ELEMENT:6
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x15555555); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000026); // RAM: 2 ELEMENT:6
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x40400000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000036); // RAM: 3 ELEMENT:6
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x88191100); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000046); // RAM: 4 ELEMENT:6
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00000800); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000056); // RAM: 5 ELEMENT:6
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x80000000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000066); // RAM: 6 ELEMENT:6
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x66F9DFFF); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000076); // RAM: 7 ELEMENT:6
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x63E3CCCF); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000096); // RAM: 9 ELEMENT:6
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00222200); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000a6); // RAM: 10 ELEMENT:6
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x55500000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000b6); // RAM: 11 ELEMENT:6
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xC0000005); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000c6); // RAM: 12 ELEMENT:6
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xB7FEE1FF); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000d6); // RAM: 13 ELEMENT:6
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00318607); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000146); // RAM: 20 ELEMENT:6
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00000001); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000007); // RAM: 0 ELEMENT:7
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x2A9AAAAA); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000017); // RAM: 1 ELEMENT:7
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x15555555); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000037); // RAM: 3 ELEMENT:7
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00222200); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000047); // RAM: 4 ELEMENT:7
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x50005000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000057); // RAM: 5 ELEMENT:7
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xC0180C11); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000067); // RAM: 6 ELEMENT:7
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00A01FFF); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000077); // RAM: 7 ELEMENT:7
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x7FE0F033); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000087); // RAM: 8 ELEMENT:7
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x20200000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000097); // RAM: 9 ELEMENT:7
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x44440000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000a7); // RAM: 10 ELEMENT:7
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x50001000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000b7); // RAM: 11 ELEMENT:7
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xE0C020C0); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000c7); // RAM: 12 ELEMENT:7
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xB05807FF); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000d7); // RAM: 13 ELEMENT:7
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x003078F9); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000147); // RAM: 20 ELEMENT:7
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00000001); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000008); // RAM: 0 ELEMENT:8
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x2A9AA6A9); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000018); // RAM: 1 ELEMENT:8
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x155556A9); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000028); // RAM: 2 ELEMENT:8
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x20200000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000038); // RAM: 3 ELEMENT:8
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00222200); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000048); // RAM: 4 ELEMENT:8
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x52054000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000058); // RAM: 5 ELEMENT:8
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x2003C31F); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000068); // RAM: 6 ELEMENT:8
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xD03FDFE0); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000078); // RAM: 7 ELEMENT:8
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x60E0CC8A); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000b8); // RAM: 11 ELEMENT:8
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xC0000000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000c8); // RAM: 12 ELEMENT:8
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x9362FFFF); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000d8); // RAM: 13 ELEMENT:8
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x01B18626); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000148); // RAM: 20 ELEMENT:8
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00000001); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000009); // RAM: 0 ELEMENT:9
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x2A9AAAAA); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000019); // RAM: 1 ELEMENT:9
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x155556A9); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000039); // RAM: 3 ELEMENT:9
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00222200); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000049); // RAM: 4 ELEMENT:9
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x05555000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000059); // RAM: 5 ELEMENT:9
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00180804); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000069); // RAM: 6 ELEMENT:9
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xB66FDFFC); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000079); // RAM: 7 ELEMENT:9
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x7CE0F073); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000089); // RAM: 8 ELEMENT:9
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x10100000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000099); // RAM: 9 ELEMENT:9
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x22113300); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000a9); // RAM: 10 ELEMENT:9
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xAD02A800); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000b9); // RAM: 11 ELEMENT:9
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x20006187); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000c9); // RAM: 12 ELEMENT:9
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xEB3FE7FE); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000d9); // RAM: 13 ELEMENT:9
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x01B1F8DB); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000149); // RAM: 20 ELEMENT:9
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00000001); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000000a); // RAM: 0 ELEMENT:10
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x2AAAAAAA); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000001a); // RAM: 1 ELEMENT:10
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x155556AA); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000002a); // RAM: 2 ELEMENT:10
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x10100000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000003a); // RAM: 3 ELEMENT:10
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x44440000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000004a); // RAM: 4 ELEMENT:10
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x07555000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000005a); // RAM: 5 ELEMENT:10
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x2000C30A); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000006a); // RAM: 6 ELEMENT:10
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x403FDFC0); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000007a); // RAM: 7 ELEMENT:10
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x6000C082); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000009a); // RAM: 9 ELEMENT:10
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00555500); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000aa); // RAM: 10 ELEMENT:10
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xF802F800); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000ba); // RAM: 11 ELEMENT:10
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x26003043); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000ca); // RAM: 12 ELEMENT:10
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xDB63FFFE); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000da); // RAM: 13 ELEMENT:10
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x01C18022); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000014a); // RAM: 20 ELEMENT:10
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00000001); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000000b); // RAM: 0 ELEMENT:11
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x2A9AAAAA); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000001b); // RAM: 1 ELEMENT:11
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x155AA6A9); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000003b); // RAM: 3 ELEMENT:11
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00111100); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000004b); // RAM: 4 ELEMENT:11
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x05555000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000006b); // RAM: 6 ELEMENT:11
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x49BFDF00); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000007b); // RAM: 7 ELEMENT:11
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x7CE0F072); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000008b); // RAM: 8 ELEMENT:11
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x08080000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000009b); // RAM: 9 ELEMENT:11
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x22220000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000ab); // RAM: 10 ELEMENT:11
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x07AAA800); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000bb); // RAM: 11 ELEMENT:11
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x20006185); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000cb); // RAM: 12 ELEMENT:11
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x35FFE600); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000db); // RAM: 13 ELEMENT:11
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x01B1DA9F); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000014b); // RAM: 20 ELEMENT:11
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00000001); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000000c); // RAM: 0 ELEMENT:12
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x2AAAAAAA); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000001c); // RAM: 1 ELEMENT:12
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x155AA6AA); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000002c); // RAM: 2 ELEMENT:12
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x08080000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000003c); // RAM: 3 ELEMENT:12
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00111100); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000004c); // RAM: 4 ELEMENT:12
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x55500000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000005c); // RAM: 5 ELEMENT:12
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00000005); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000006c); // RAM: 6 ELEMENT:12
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x4DBF1F00); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000007c); // RAM: 7 ELEMENT:12
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x7C00F872); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000ac); // RAM: 10 ELEMENT:12
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x52ABF800); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000bc); // RAM: 11 ELEMENT:12
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x21801005); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000cc); // RAM: 12 ELEMENT:12
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x05BFFFC0); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000dc); // RAM: 13 ELEMENT:12
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x01C1A266); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000014c); // RAM: 20 ELEMENT:12
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00000001); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000000d); // RAM: 0 ELEMENT:13
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x2A9AAAAA); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000001d); // RAM: 1 ELEMENT:13
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x2A9AA6A9); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000003d); // RAM: 3 ELEMENT:13
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00111100); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000004d); // RAM: 4 ELEMENT:13
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x55500000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000005d); // RAM: 5 ELEMENT:13
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00200045); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000006d); // RAM: 6 ELEMENT:13
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x24300000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000007d); // RAM: 7 ELEMENT:13
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x1CC07061); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000008d); // RAM: 8 ELEMENT:13
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x04040000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000009d); // RAM: 9 ELEMENT:13
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x11089980); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000ad); // RAM: 10 ELEMENT:13
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xA8280800); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000bd); // RAM: 11 ELEMENT:13
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x0001882A); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000cd); // RAM: 12 ELEMENT:13
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xB25D1000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000dd); // RAM: 13 ELEMENT:13
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x31B06461); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000014d); // RAM: 20 ELEMENT:13
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00000001); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000000e); // RAM: 0 ELEMENT:14
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x2AAAAAAA); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000001e); // RAM: 1 ELEMENT:14
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x2A9AA6AA); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000002e); // RAM: 2 ELEMENT:14
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x04040000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000003e); // RAM: 3 ELEMENT:14
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x22220000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000006e); // RAM: 6 ELEMENT:14
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x20300000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000007e); // RAM: 7 ELEMENT:14
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x1C007061); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000009e); // RAM: 9 ELEMENT:14
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x002AAA80); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000ae); // RAM: 10 ELEMENT:14
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xAAA80000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000be); // RAM: 11 ELEMENT:14
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x20000002); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000ce); // RAM: 12 ELEMENT:14
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xB37CF000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000de); // RAM: 13 ELEMENT:14
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x31F1E667); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000014e); // RAM: 20 ELEMENT:14
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00000001); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000000f); // RAM: 0 ELEMENT:15
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x2AAAAAAA); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000001f); // RAM: 1 ELEMENT:15
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x2AAAAAAA); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000008f); // RAM: 8 ELEMENT:15
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x02020000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000009f); // RAM: 9 ELEMENT:15
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x11554400); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000af); // RAM: 10 ELEMENT:15
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x78005800); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000bf); // RAM: 11 ELEMENT:15
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x0601B459); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000cf); // RAM: 12 ELEMENT:15
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x80501000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000df); // RAM: 13 ELEMENT:15
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x3FF07819); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x0000014f); // RAM: 20 ELEMENT:15
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00000001); // DATA
}

/*******************************************************************************
 Csi12ToPacked12
*******************************************************************************/

void Csi12ToPacked12::enable(Hololink& hololink, uint32_t sif_address)
{
    hololink.write_uint32(sif_address + PACKETIZER_MODE, 0x11210007);
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000000); // RAM: 0 ELEMENT:0
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xAAAAAAAA); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000010); // RAM: 1 ELEMENT:0
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x5AA5AAAA); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000030); // RAM: 3 ELEMENT:0
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00111100); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000040); // RAM: 4 ELEMENT:0
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x55500000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000050); // RAM: 5 ELEMENT:0
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00000005); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000060); // RAM: 6 ELEMENT:0
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x6DBC3000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000070); // RAM: 7 ELEMENT:0
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00008103); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000090); // RAM: 9 ELEMENT:0
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x44226600); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000a0); // RAM: 10 ELEMENT:0
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xA8002800); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000b0); // RAM: 11 ELEMENT:0
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x0000020A); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000c0); // RAM: 12 ELEMENT:0
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x02400000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000d0); // RAM: 13 ELEMENT:0
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00000810); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000140); // RAM: 20 ELEMENT:0
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00000001); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000001); // RAM: 0 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xAAAAAAFF); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000011); // RAM: 1 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xA5AAAAAA); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000021); // RAM: 2 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x02020000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000041); // RAM: 4 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x05000000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000051); // RAM: 5 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00001865); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000061); // RAM: 6 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x60300000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000071); // RAM: 7 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x0000F1E3); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000081); // RAM: 8 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x40400000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000091); // RAM: 9 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00111100); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000a1); // RAM: 10 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x78005000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000b1); // RAM: 11 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00001455); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000f1); // RAM: 15 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00222200); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000101); // RAM: 16 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xA802A800); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000111); // RAM: 17 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xF0000002); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000121); // RAM: 18 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x6DB0F000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000141); // RAM: 20 ELEMENT:1
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00000001); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000002); // RAM: 0 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xAAAAAAAF); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000012); // RAM: 1 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0xAAAAAAAA); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000092); // RAM: 9 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x004CCC80); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000a2); // RAM: 10 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x50000000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000b2); // RAM: 11 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00002080); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000c2); // RAM: 12 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x92406000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000d2); // RAM: 13 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00000408); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000e2); // RAM: 14 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x20200000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x000000f2); // RAM: 15 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x22002200); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000102); // RAM: 16 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x28002800); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000112); // RAM: 17 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x30001860); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000122); // RAM: 18 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x01B03000); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000132); // RAM: 19 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x0000078F); // DATA
    hololink.write_uint32(sif_address + PACKETIZER_RAM, 0x00000142); // RAM: 20 ELEMENT:2
    hololink.write_uint32(sif_address + PACKETIZER_DATA, 0x00000001); // DATA
}

} // namespace hololink
