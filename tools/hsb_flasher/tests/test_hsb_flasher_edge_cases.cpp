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

#include <gtest/gtest.h>

#include "hsb_discovery.hpp"
#include "hsb_flasher.hpp"
#include "hsb_fw_lookup.hpp"

#include <cstdint>
#include <string>
#include <vector>

using namespace hsb_flasher;

// ── Mock providers ───────────────────────────────────────────────────

/// Fires one callback with the configured metadata.
class SingleDeviceProvider : public IEnumerationProvider {
public:
    SingleDeviceProvider(std::string ip, std::string mac,
        int64_t version, std::string uuid)
        : ip_(std::move(ip))
        , mac_(std::move(mac))
        , version_(version)
        , uuid_(std::move(uuid))
    {
    }

    void enumerate(Callback callback, float) override
    {
        hololink::Metadata md;
        md["peer_ip"] = ip_;
        md["mac_id"] = mac_;
        md["hsb_ip_version"] = version_;
        md["fpga_uuid"] = uuid_;
        md["data_plane"] = int64_t(0);

        std::vector<uint8_t> packet(64, 0);
        callback(packet, md);
    }

private:
    std::string ip_;
    std::string mac_;
    int64_t version_;
    std::string uuid_;
};

/// Simulates a legacy device that lacks vendor data.
/// Only provides peer_ip in metadata; no mac_id, no data_plane, no fpga_uuid,
/// no hsb_ip_version. The raw packet carries MAC at offset 28 with hlen at
/// offset 2.
class LegacyDeviceProvider : public IEnumerationProvider {
public:
    LegacyDeviceProvider(std::string ip, std::vector<uint8_t> mac_bytes,
        int64_t version)
        : ip_(std::move(ip))
        , mac_bytes_(std::move(mac_bytes))
        , version_(version)
    {
    }

    void enumerate(Callback callback, float) override
    {
        hololink::Metadata md;
        md["peer_ip"] = ip_;
        // hsb_ip_version is provided so the test doesn't trigger a real
        // UDP probe (probe_version_udp would fail without a real device)
        md["hsb_ip_version"] = version_;

        std::vector<uint8_t> packet(64, 0);
        packet[2] = static_cast<uint8_t>(mac_bytes_.size()); // hlen
        for (size_t i = 0; i < mac_bytes_.size(); ++i)
            packet[28 + i] = mac_bytes_[i];

        callback(packet, md);
    }

private:
    std::string ip_;
    std::vector<uint8_t> mac_bytes_;
    int64_t version_;
};

// ── Helpers ──────────────────────────────────────────────────────────

static hsb_flasher_context make_context(const std::string& ip,
    const std::string& target_version = "")
{
    hsb_flasher_context ctx {};
    ctx.hololink_ip = ip;
    ctx.target_version = target_version;
    ctx.log_level = hsb_flasher_log_level::INFO;
    ctx.timeout = 1.0f;
    return ctx;
}

// ── Legacy device fallback ───────────────────────────────────────────

// Legacy device without vendor data — MAC is extracted from raw BOOTP packet
TEST(LegacyDevice, ExtractsMacFromPacketWhenVendorDataAbsent)
{
    auto ctx = make_context("192.168.0.2", "2507");
    LegacyDeviceProvider provider(
        "192.168.0.2",
        { 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF },
        0x2504);

    ASSERT_TRUE(discover_device(ctx, provider));

    auto mac = ctx.enumeration_metadata.get<std::string>("mac_id").value_or("");
    EXPECT_EQ(mac, "AA:BB:CC:DD:EE:FF");
}

// Legacy device without vendor data — fpga_uuid defaults to "N/A"
TEST(LegacyDevice, SetsFpgaUuidToNA)
{
    auto ctx = make_context("192.168.0.2", "2507");
    LegacyDeviceProvider provider(
        "192.168.0.2",
        { 0x11, 0x22, 0x33, 0x44, 0x55, 0x66 },
        0x2504);

    ASSERT_TRUE(discover_device(ctx, provider));

    auto uuid = ctx.enumeration_metadata.get<std::string>("fpga_uuid").value_or("");
    EXPECT_EQ(uuid, "N/A");
}

// UUID override (-u) replaces the "N/A" default for legacy devices
TEST(LegacyDevice, UuidOverrideReplacesNA)
{
    auto ctx = make_context("192.168.0.2", "2507");
    ctx.fpga_uuid_override = "889b7ce3-65a5-4247-8b05-4ff1904c3359";

    LegacyDeviceProvider provider(
        "192.168.0.2",
        { 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF },
        0x2504);

    ASSERT_TRUE(discover_device(ctx, provider));

    // Simulate what main() does after discover_device()
    auto uuid = ctx.enumeration_metadata.get<std::string>("fpga_uuid").value_or("N/A");
    if (uuid == "N/A" && !ctx.fpga_uuid_override.empty()) {
        ctx.enumeration_metadata["fpga_uuid"] = ctx.fpga_uuid_override;
    }

    EXPECT_EQ(ctx.enumeration_metadata.get<std::string>("fpga_uuid").value_or(""),
        "889b7ce3-65a5-4247-8b05-4ff1904c3359");
}

// ── Direct flash mode ────────────────────────────────────────────────
//
// Test matrix:
//   args:           cpnx | cpnx+clnx | cpnx+flash-version | cpnx+clnx+flash-version
//   device UUID:    known | unknown
//   device version: known (0x2507) | unknown (0x0001)
//   flash-version:  known ("2507") | unknown ("0001")

static constexpr const char* HSB_LITE_UUID = "889b7ce3-65a5-4247-8b05-4ff1904c3359";
static constexpr const char* UNKNOWN_UUID = "00000000-0000-0000-0000-000000000000";

/// Replicate what main.cpp does: inject --flash-version into enumeration_metadata
static void apply_flash_version_override(hsb_flasher_context& ctx)
{
    if (!ctx.direct_flash.flash_version.empty()) {
        int64_t ver = std::stoi(ctx.direct_flash.flash_version, nullptr, 16);
        ctx.enumeration_metadata["hsb_ip_version"] = ver;
    }
}

// ── --cpnx only (no --clnx, no --flash-version) ─────────────────────

// Known UUID, known device version → strategy found
TEST(DirectMode, CpnxOnly_KnownUuid_KnownVersion_ExpectSuccess)
{
    auto ctx = make_context("192.168.0.2");
    ctx.direct_flash.cpnx_path = "/path/to/cpnx.bin";

    SingleDeviceProvider provider(
        "192.168.0.2", "AA:BB:CC:DD:EE:FF", 0x2507, HSB_LITE_UUID);

    ASSERT_TRUE(discover_device(ctx, provider));
    apply_flash_version_override(ctx);

    auto flasher = get_flasher(ctx);
    EXPECT_NE(flasher, nullptr);
}

// Known UUID, unknown device version → no strategy matches
TEST(DirectMode, CpnxOnly_KnownUuid_UnknownVersion_ExpectFailure)
{
    auto ctx = make_context("192.168.0.2");
    ctx.direct_flash.cpnx_path = "/path/to/cpnx.bin";

    SingleDeviceProvider provider(
        "192.168.0.2", "AA:BB:CC:DD:EE:FF", 0x0001, HSB_LITE_UUID);

    ASSERT_TRUE(discover_device(ctx, provider));
    apply_flash_version_override(ctx);

    auto flasher = get_flasher(ctx);
    EXPECT_EQ(flasher, nullptr);
}

// Unknown UUID, known device version → UUID mismatch, no strategy matches
TEST(DirectMode, CpnxOnly_UnknownUuid_KnownVersion_ExpectFailure)
{
    auto ctx = make_context("192.168.0.2");
    ctx.direct_flash.cpnx_path = "/path/to/cpnx.bin";

    SingleDeviceProvider provider(
        "192.168.0.2", "AA:BB:CC:DD:EE:FF", 0x2507, UNKNOWN_UUID);

    ASSERT_TRUE(discover_device(ctx, provider));
    apply_flash_version_override(ctx);

    auto flasher = get_flasher(ctx);
    EXPECT_EQ(flasher, nullptr);
}

// ── --cpnx + --clnx (no --flash-version) ────────────────────────────

// Known UUID, known device version → strategy found
TEST(DirectMode, CpnxClnx_KnownUuid_KnownVersion_ExpectSuccess)
{
    auto ctx = make_context("192.168.0.2");
    ctx.direct_flash.cpnx_path = "/path/to/cpnx.bin";
    ctx.direct_flash.clnx_path = "/path/to/clnx.bin";

    SingleDeviceProvider provider(
        "192.168.0.2", "AA:BB:CC:DD:EE:FF", 0x2507, HSB_LITE_UUID);

    ASSERT_TRUE(discover_device(ctx, provider));
    apply_flash_version_override(ctx);

    auto flasher = get_flasher(ctx);
    EXPECT_NE(flasher, nullptr);
}

// Known UUID, unknown device version → no strategy matches
TEST(DirectMode, CpnxClnx_KnownUuid_UnknownVersion_ExpectFailure)
{
    auto ctx = make_context("192.168.0.2");
    ctx.direct_flash.cpnx_path = "/path/to/cpnx.bin";
    ctx.direct_flash.clnx_path = "/path/to/clnx.bin";

    SingleDeviceProvider provider(
        "192.168.0.2", "AA:BB:CC:DD:EE:FF", 0x0001, HSB_LITE_UUID);

    ASSERT_TRUE(discover_device(ctx, provider));
    apply_flash_version_override(ctx);

    auto flasher = get_flasher(ctx);
    EXPECT_EQ(flasher, nullptr);
}

// ── --cpnx + --flash-version (no --clnx) ────────────────────────────

// Known UUID, unknown device version, known flash-version → override rescues
TEST(DirectMode, CpnxFlashVer_KnownUuid_UnknownVersion_KnownOverride_ExpectSuccess)
{
    auto ctx = make_context("192.168.0.2");
    ctx.direct_flash.cpnx_path = "/path/to/cpnx.bin";
    ctx.direct_flash.flash_version = "2507";

    SingleDeviceProvider provider(
        "192.168.0.2", "AA:BB:CC:DD:EE:FF", 0x0001, HSB_LITE_UUID);

    ASSERT_TRUE(discover_device(ctx, provider));
    apply_flash_version_override(ctx);

    auto flasher = get_flasher(ctx);
    EXPECT_NE(flasher, nullptr);
}

// Known UUID, known device version, known flash-version → override when not needed
TEST(DirectMode, CpnxFlashVer_KnownUuid_KnownVersion_KnownOverride_ExpectSuccess)
{
    auto ctx = make_context("192.168.0.2");
    ctx.direct_flash.cpnx_path = "/path/to/cpnx.bin";
    ctx.direct_flash.flash_version = "2507";

    SingleDeviceProvider provider(
        "192.168.0.2", "AA:BB:CC:DD:EE:FF", 0x2507, HSB_LITE_UUID);

    ASSERT_TRUE(discover_device(ctx, provider));
    apply_flash_version_override(ctx);

    auto flasher = get_flasher(ctx);
    EXPECT_NE(flasher, nullptr);
}

// Known UUID, known device version, unknown flash-version → override to bad version
TEST(DirectMode, CpnxFlashVer_KnownUuid_KnownVersion_UnknownOverride_ExpectFailure)
{
    auto ctx = make_context("192.168.0.2");
    ctx.direct_flash.cpnx_path = "/path/to/cpnx.bin";
    ctx.direct_flash.flash_version = "0001";

    SingleDeviceProvider provider(
        "192.168.0.2", "AA:BB:CC:DD:EE:FF", 0x2507, HSB_LITE_UUID);

    ASSERT_TRUE(discover_device(ctx, provider));
    apply_flash_version_override(ctx);

    auto flasher = get_flasher(ctx);
    EXPECT_EQ(flasher, nullptr);
}

// Unknown UUID, unknown device version, known flash-version → UUID mismatch
TEST(DirectMode, CpnxFlashVer_UnknownUuid_UnknownVersion_KnownOverride_ExpectFailure)
{
    auto ctx = make_context("192.168.0.2");
    ctx.direct_flash.cpnx_path = "/path/to/cpnx.bin";
    ctx.direct_flash.flash_version = "2507";

    SingleDeviceProvider provider(
        "192.168.0.2", "AA:BB:CC:DD:EE:FF", 0x0001, UNKNOWN_UUID);

    ASSERT_TRUE(discover_device(ctx, provider));
    apply_flash_version_override(ctx);

    auto flasher = get_flasher(ctx);
    EXPECT_EQ(flasher, nullptr);
}

// ── --cpnx + --clnx + --flash-version ───────────────────────────────

// Known UUID, unknown device version, known flash-version → override rescues
TEST(DirectMode, CpnxClnxFlashVer_KnownUuid_UnknownVersion_KnownOverride_ExpectSuccess)
{
    auto ctx = make_context("192.168.0.2");
    ctx.direct_flash.cpnx_path = "/path/to/cpnx.bin";
    ctx.direct_flash.clnx_path = "/path/to/clnx.bin";
    ctx.direct_flash.flash_version = "2507";

    SingleDeviceProvider provider(
        "192.168.0.2", "AA:BB:CC:DD:EE:FF", 0x0001, HSB_LITE_UUID);

    ASSERT_TRUE(discover_device(ctx, provider));
    apply_flash_version_override(ctx);

    auto flasher = get_flasher(ctx);
    EXPECT_NE(flasher, nullptr);
}

// Known UUID, known device version, known flash-version → all args, everything known
TEST(DirectMode, CpnxClnxFlashVer_KnownUuid_KnownVersion_KnownOverride_ExpectSuccess)
{
    auto ctx = make_context("192.168.0.2");
    ctx.direct_flash.cpnx_path = "/path/to/cpnx.bin";
    ctx.direct_flash.clnx_path = "/path/to/clnx.bin";
    ctx.direct_flash.flash_version = "2507";

    SingleDeviceProvider provider(
        "192.168.0.2", "AA:BB:CC:DD:EE:FF", 0x2507, HSB_LITE_UUID);

    ASSERT_TRUE(discover_device(ctx, provider));
    apply_flash_version_override(ctx);

    auto flasher = get_flasher(ctx);
    EXPECT_NE(flasher, nullptr);
}

// ── HSB device with CPNX only (no CLNX) ─────────────────────────────

static constexpr const char* NO_CLNX_UUID = "f1627640-b4dc-48af-a360-c55b09b3d230";

// Full pipeline for a device that only has CPNX firmware (no CLNX)
TEST(HsbNoClnx, HappyPath_ExpectSuccess)
{
    auto ctx = make_context("192.168.0.2", "2507");
    SingleDeviceProvider provider(
        "192.168.0.2", "AA:BB:CC:DD:EE:FF", 0x2507, NO_CLNX_UUID);

    // Discovery succeeds
    ASSERT_TRUE(discover_device(ctx, provider));

    // Manifest found for this UUID
    ASSERT_TRUE(find_manifest_by_uuid(ctx));
    EXPECT_FALSE(ctx.firmware_info_path.empty());

    // Target version exists in manifest (CPNX only, no CLNX)
    ASSERT_TRUE(verify_firmware_details(ctx));
    EXPECT_FALSE(ctx.cpnx.location.empty());
    EXPECT_TRUE(ctx.clnx.location.empty());

    // Correct flasher is retrieved
    auto flasher = get_flasher(ctx);
    EXPECT_NE(flasher, nullptr);
}

// UUID matches but target version is not in the manifest
TEST(HsbNoClnx, UnknownVersion_ExpectFailure)
{
    auto ctx = make_context("192.168.0.2", "9999");
    SingleDeviceProvider provider(
        "192.168.0.2", "AA:BB:CC:DD:EE:FF", 0x2507, NO_CLNX_UUID);

    ASSERT_TRUE(discover_device(ctx, provider));
    ASSERT_TRUE(find_manifest_by_uuid(ctx));
    EXPECT_FALSE(verify_firmware_details(ctx));
}
