/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Hardware-free checks for the module-native IMX258 driver
 * (hololink_module/host/sensors/imx258/): pins the register-table
 * invariants the PCIe receive path depends on (streaming-off first, no
 * trailing streaming-on -- start() owns that -- PDAF disabled, and the
 * 1194.75 Mb/s/lane mode composed from the base table exactly as the
 * reference does it), the IMX258_FRAME_FORMAT values, and the CSI-converter
 * geometry Imx258Cam derives (start byte 640, 2400-byte RAW10 lines).
 * No I2C is attempted: Imx258Cam's constructor resolves its I2C service
 * through the module registry, so the geometry is driven through the
 * static Imx258Cam::configure_converter overload against a stub
 * CsiConverterV1.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "hololink/module/csi_converter.hpp"
#include "hololink/module/sensors/imx258/imx258_cam.hpp"
#include "hololink/module/sensors/imx258/imx258_mode.hpp"

namespace {

using namespace hololink::module::sensors::imx258;

using RegisterEntry = std::pair<uint16_t, uint8_t>;

bool table_contains(const RegisterEntry* begin, const RegisterEntry* end,
    uint16_t reg, uint8_t value)
{
    return std::find(begin, end, RegisterEntry { reg, value }) != end;
}

/* Stub converter with the receiver geometry the PCIe path exposes to
 * the CSI converter: no receiver prefix, lines padded to 8 bytes, RAW10
 * packing 4 pixels into 5 bytes. */
class StubCsiConverter : public CsiConverterV1 {
public:
    struct Configured {
        uint32_t start_byte = 0;
        uint32_t received_bytes_per_line = 0;
        uint32_t pixel_width = 0;
        uint32_t pixel_height = 0;
        PixelFormat pixel_format = PixelFormat::RAW_8;
        uint32_t trailing_bytes = 0;
    };

    void configure(uint32_t start_byte, uint32_t received_bytes_per_line,
        uint32_t pixel_width, uint32_t pixel_height, PixelFormat pixel_format,
        uint32_t trailing_bytes) override
    {
        calls.push_back(Configured { start_byte, received_bytes_per_line,
            pixel_width, pixel_height, pixel_format, trailing_bytes });
    }

    uint32_t receiver_start_byte() override { return 0; }

    uint32_t received_line_bytes(uint32_t line_bytes) override
    {
        return (line_bytes + 7) & ~7u;
    }

    uint32_t transmitted_line_bytes(PixelFormat pixel_format, uint32_t pixel_width) override
    {
        switch (pixel_format) {
        case PixelFormat::RAW_8:
            return pixel_width;
        case PixelFormat::RAW_10:
            return pixel_width * 10 / 8;
        case PixelFormat::RAW_12:
            return pixel_width * 12 / 8;
        }
        return 0;
    }

    std::vector<Configured> calls;
};

TEST(HololinkModuleImx258, Constants)
{
    EXPECT_EQ(IMX258_I2C_ADDRESS, 0x1Au);
    EXPECT_EQ(DEFAULT_CAM_I2C_BUS, 0u);
    EXPECT_EQ(Imx258Cam::CAM_I2C_ADDRESS, IMX258_I2C_ADDRESS);
    EXPECT_EQ(REG_MODEL_ID_HI, 0x0016);
    EXPECT_EQ(REG_MODEL_ID_LO, 0x0017);
    EXPECT_EQ(MODEL_ID, 0x0258);
    EXPECT_EQ(REG_MODE_SELECT, 0x0100);
    EXPECT_EQ(REG_VT_PLL_MULT, 0x0307);
    EXPECT_EQ(REG_COARSE_INTEG_HI, 0x0202);
    EXPECT_EQ(REG_COARSE_INTEG_LO, 0x0203);
    EXPECT_EQ(REG_ANA_GAIN_HI, 0x0204);
    EXPECT_EQ(REG_ANA_GAIN_LO, 0x0205);
    EXPECT_EQ(RAW10_START_BYTE, 640u);
}

TEST(HololinkModuleImx258, FrameFormatTable)
{
    ASSERT_EQ(IMX258_FRAME_FORMAT.size(), static_cast<size_t>(Imx258_Mode::Unknown));

    const FrameFormat& fps60_1194
        = IMX258_FRAME_FORMAT[static_cast<size_t>(Imx258_Mode::IMX258_MODE_1920X1080_60FPS_1194MBPS)];
    EXPECT_EQ(fps60_1194.width, 1920u);
    EXPECT_EQ(fps60_1194.height, 1080u);
    EXPECT_EQ(fps60_1194.framerate, 60u);
    EXPECT_EQ(fps60_1194.pixel_format, PixelFormat::RAW_10);

    EXPECT_EQ(static_cast<uint32_t>(Imx258_Mode::IMX258_MODE_1920X1080_60FPS_1194MBPS), 0u);
    EXPECT_EQ(static_cast<uint32_t>(Imx258_Mode::Unknown), 1u);
}

// The 1194.75 Mb/s/lane reference profile (CertusPro-NX Versa): the base table with the
// reference's overrides applied in place and its five new registers appended, exactly as
// camera_hdmi_setup.py's _camera_profile() composes CAM_CONFIG_REFERENCE_1080P60_1194MBPS.
TEST(HololinkModuleImx258, Table1194MbpsIsTheReferenceProfile)
{
    const auto& base = IMX258_BASE_SEQUENCE;
    const auto& ref = IMX258_MODE_1920X1080_60FPS_1194MBPS_SEQUENCE;
    ASSERT_EQ(ref.size(), base.size() + 5);

    // Same register order as the base table for its length; only overridden values differ.
    for (size_t i = 0; i < base.size(); ++i) {
        EXPECT_EQ(ref[i].first, base[i].first) << "register order diverges at " << i;
    }
    // The five registers the base table never wrote come last.
    EXPECT_EQ(ref[base.size() + 0], (RegisterEntry { 0x0404, 0x00 }));
    EXPECT_EQ(ref[base.size() + 1], (RegisterEntry { 0x0405, 0x20 }));
    EXPECT_EQ(ref[base.size() + 2], (RegisterEntry { 0x3038, 0x00 }));
    EXPECT_EQ(ref[base.size() + 3], (RegisterEntry { 0x303A, 0x00 }));
    EXPECT_EQ(ref[base.size() + 4], (RegisterEntry { 0x303B, 0x00 }));

    // No register appears twice.
    for (size_t i = 0; i < ref.size(); ++i) {
        for (size_t j = i + 1; j < ref.size(); ++j) {
            EXPECT_NE(ref[i].first, ref[j].first) << "duplicate register " << std::hex << ref[i].first;
        }
    }

    // The lane rate: 0x0820..0x0823 = 0x12AB0000 (4779 Mb/s over four lanes), not 0x05CD0000.
    EXPECT_TRUE(table_contains(ref.begin(), ref.end(), 0x0820, 0x12));
    EXPECT_TRUE(table_contains(ref.begin(), ref.end(), 0x0821, 0xAB));
    EXPECT_TRUE(table_contains(ref.begin(), ref.end(), 0x0822, 0x00));
    EXPECT_TRUE(table_contains(ref.begin(), ref.end(), 0x0823, 0x00));
    EXPECT_FALSE(table_contains(ref.begin(), ref.end(), 0x0820, 0x05));
    // PLLs, timing, exposure from the reference profile.
    EXPECT_TRUE(table_contains(ref.begin(), ref.end(), REG_VT_PLL_MULT, 0xB1));
    EXPECT_TRUE(table_contains(ref.begin(), ref.end(), 0x030B, 0x01));
    EXPECT_TRUE(table_contains(ref.begin(), ref.end(), 0x030D, 0x03));
    EXPECT_TRUE(table_contains(ref.begin(), ref.end(), 0x030F, 0x85));
    EXPECT_TRUE(table_contains(ref.begin(), ref.end(), 0x0340, 0x05));
    EXPECT_TRUE(table_contains(ref.begin(), ref.end(), 0x0341, 0xD0));
    EXPECT_TRUE(table_contains(ref.begin(), ref.end(), 0x0342, 0x14));
    EXPECT_TRUE(table_contains(ref.begin(), ref.end(), 0x0343, 0xE8));
    EXPECT_TRUE(table_contains(ref.begin(), ref.end(), REG_COARSE_INTEG_HI, 0x03));
    EXPECT_TRUE(table_contains(ref.begin(), ref.end(), REG_COARSE_INTEG_LO, 0x20));
    // Untouched by the profile: four lanes, RAW10, the 1920x1080 output size, PDAF off.
    EXPECT_TRUE(table_contains(ref.begin(), ref.end(), 0x0114, 0x03));
    EXPECT_TRUE(table_contains(ref.begin(), ref.end(), 0x0112, 0x0A));
    EXPECT_TRUE(table_contains(ref.begin(), ref.end(), 0x034C, 0x07));
    EXPECT_TRUE(table_contains(ref.begin(), ref.end(), 0x034D, 0x80));
    EXPECT_TRUE(table_contains(ref.begin(), ref.end(), 0x034E, 0x04));
    EXPECT_TRUE(table_contains(ref.begin(), ref.end(), 0x034F, 0x38));
    EXPECT_TRUE(table_contains(ref.begin(), ref.end(), 0x3030, 0x00));
    // Same lifecycle invariants as the other tables: stream off first, never on.
    EXPECT_EQ(ref.front(), (RegisterEntry { REG_MODE_SELECT, 0x00 }));
    EXPECT_FALSE(table_contains(ref.begin(), ref.end(), REG_MODE_SELECT, 0x01));
}

// The base table the mode is composed from keeps the reference's lifecycle and geometry
// invariants, so the composed mode inherits them.
TEST(HololinkModuleImx258, BaseTableInvariants)
{
    const auto& table = IMX258_BASE_SEQUENCE;
    ASSERT_EQ(table.size(), 82u);

    // Streaming is switched off before anything is reprogrammed ...
    EXPECT_EQ(table.front().first, REG_MODE_SELECT);
    EXPECT_EQ(table.front().second, 0x00);
    // ... and never switched on by the table: Imx258Cam::start() owns
    // that so streaming begins after the receiver ring is armed.
    EXPECT_FALSE(table_contains(table.begin(), table.end(), REG_MODE_SELECT, 0x01));
    EXPECT_EQ(std::count_if(table.begin(), table.end(),
                  [](const RegisterEntry& e) { return e.first == REG_MODE_SELECT; }),
        1);

    // PDAF-off triple: the RX bypass expects only RAW10 image packets.
    EXPECT_TRUE(table_contains(table.begin(), table.end(), 0x3030, 0x00));
    EXPECT_TRUE(table_contains(table.begin(), table.end(), 0x3032, 0x00));
    EXPECT_TRUE(table_contains(table.begin(), table.end(), 0x7BCD, 0x00));
    // ... and it is the tail of the table, in the reference order.
    EXPECT_EQ(table[79], (RegisterEntry { 0x3030, 0x00 }));
    EXPECT_EQ(table[80], (RegisterEntry { 0x3032, 0x00 }));
    EXPECT_EQ(table[81], (RegisterEntry { 0x7BCD, 0x00 }));

    // Geometry the receiver relies on: RAW10, 4 lanes, 1920x1080, VT PLL 110.
    EXPECT_TRUE(table_contains(table.begin(), table.end(), 0x0114, 0x03));
    EXPECT_TRUE(table_contains(table.begin(), table.end(), 0x0112, 0x0A));
    EXPECT_TRUE(table_contains(table.begin(), table.end(), 0x0113, 0x0A));
    EXPECT_TRUE(table_contains(table.begin(), table.end(), 0x034C, 0x07));
    EXPECT_TRUE(table_contains(table.begin(), table.end(), 0x034D, 0x80));
    EXPECT_TRUE(table_contains(table.begin(), table.end(), 0x034E, 0x04));
    EXPECT_TRUE(table_contains(table.begin(), table.end(), 0x034F, 0x38));
    EXPECT_TRUE(table_contains(table.begin(), table.end(), REG_VT_PLL_MULT, 0x6E));
}

TEST(HololinkModuleImx258, ConfigureConverterGeometry)
{
    StubCsiConverter converter;
    const FrameFormat& fmt
        = IMX258_FRAME_FORMAT[static_cast<size_t>(Imx258_Mode::IMX258_MODE_1920X1080_60FPS_1194MBPS)];
    Imx258Cam::configure_converter(converter, fmt.width, fmt.height);

    ASSERT_EQ(converter.calls.size(), 1u);
    const auto& c = converter.calls.front();
    EXPECT_EQ(c.start_byte, 640u);
    EXPECT_EQ(c.received_bytes_per_line, 2400u);
    EXPECT_EQ(c.pixel_width, 1920u);
    EXPECT_EQ(c.pixel_height, 1080u);
    EXPECT_EQ(c.pixel_format, PixelFormat::RAW_10);
    EXPECT_EQ(c.trailing_bytes, 0u);

    // Parity with the PCIe receiver's frame geometry: prefix + 1080 lines.
    EXPECT_EQ(c.start_byte + c.received_bytes_per_line * c.pixel_height, 2592640u);
}

TEST(HololinkModuleImx258, ConfigureConverterHonoursReceiverStartByte)
{
    class OffsetConverter : public StubCsiConverter {
    public:
        uint32_t receiver_start_byte() override { return 4096; }
    } converter;

    Imx258Cam::configure_converter(converter, 1920, 1080);
    ASSERT_EQ(converter.calls.size(), 1u);
    EXPECT_EQ(converter.calls.front().start_byte, 4096u + RAW10_START_BYTE);
    EXPECT_EQ(converter.calls.front().received_bytes_per_line, 2400u);
}

} // namespace
