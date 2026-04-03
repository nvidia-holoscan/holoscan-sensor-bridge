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

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <openssl/evp.h>

using namespace hsb_flasher;

// ── Mock enumeration providers ───────────────────────────────────────

class EmptyProvider : public IEnumerationProvider {
public:
    void enumerate(Callback, float) override { }
};

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

class MultiDeviceProvider : public IEnumerationProvider {
public:
    explicit MultiDeviceProvider(std::string target_ip)
        : target_ip_(std::move(target_ip))
    {
    }

    void enumerate(Callback callback, float) override
    {
        std::vector<uint8_t> packet(64, 0);

        hololink::Metadata md1;
        md1["peer_ip"] = target_ip_;
        md1["mac_id"] = std::string("AA:BB:CC:DD:EE:01");
        md1["hsb_ip_version"] = int64_t(0x2507);
        md1["fpga_uuid"] = std::string("889b7ce3-65a5-4247-8b05-4ff1904c3359");
        md1["data_plane"] = int64_t(0);
        callback(packet, md1);

        hololink::Metadata md2;
        md2["peer_ip"] = target_ip_;
        md2["mac_id"] = std::string("AA:BB:CC:DD:EE:02");
        md2["hsb_ip_version"] = int64_t(0x2507);
        md2["fpga_uuid"] = std::string("889b7ce3-65a5-4247-8b05-4ff1904c3359");
        md2["data_plane"] = int64_t(0);
        callback(packet, md2);
    }

private:
    std::string target_ip_;
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

/// RAII temp file with known content, size, and MD5.
class TempFirmwareFile {
public:
    explicit TempFirmwareFile(const std::string& content,
        const std::string& name = "hsb_test_firmware.bin")
        : path_(std::filesystem::temp_directory_path() / name)
    {
        std::ofstream out(path_, std::ios::binary);
        out.write(content.data(), content.size());
        out.close();

        size_ = content.size();
        md5_ = compute_md5(content);
    }

    ~TempFirmwareFile()
    {
        std::filesystem::remove(path_);
    }

    const std::string& path() const { return path_str_; }
    size_t size() const { return size_; }
    const std::string& md5() const { return md5_; }

private:
    static std::string compute_md5(const std::string& data)
    {
        EVP_MD_CTX* ctx = EVP_MD_CTX_new();
        EVP_DigestInit_ex(ctx, EVP_md5(), nullptr);
        EVP_DigestUpdate(ctx, data.data(), data.size());

        unsigned char digest[EVP_MAX_MD_SIZE];
        unsigned int len = 0;
        EVP_DigestFinal_ex(ctx, digest, &len);
        EVP_MD_CTX_free(ctx);

        std::ostringstream hex;
        for (unsigned int i = 0; i < len; ++i)
            hex << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<int>(digest[i]);
        return hex.str();
    }

    std::filesystem::path path_;
    std::string path_str_ = path_.string();
    size_t size_;
    std::string md5_;
};

/// Run discover → find_manifest → verify_firmware_details and return a
/// ready-to-use context. ASSERTs at each step so callers don't need to.
static hsb_flasher_context make_verified_context()
{
    auto ctx = make_context("192.168.0.2", "2507");
    SingleDeviceProvider provider(
        "192.168.0.2", "AA:BB:CC:DD:EE:FF", 0x2504,
        "889b7ce3-65a5-4247-8b05-4ff1904c3359");

    EXPECT_TRUE(discover_device(ctx, provider));
    EXPECT_TRUE(find_manifest_by_uuid(ctx));
    EXPECT_TRUE(verify_firmware_details(ctx));
    return ctx;
}

// ── Happy path ───────────────────────────────────────────────────────

// Full pipeline: discover → version differs → manifest found → firmware verified
TEST(HsbFlasher, HappyPath_ExpectSuccess)
{
    auto ctx = make_context("192.168.0.2", "2507");
    SingleDeviceProvider provider(
        "192.168.0.2", "AA:BB:CC:DD:EE:FF", 0x2504,
        "889b7ce3-65a5-4247-8b05-4ff1904c3359");

    // Discovery succeeds
    ASSERT_TRUE(discover_device(ctx, provider));

    // Device version differs from target (flash should proceed)
    int64_t current_version = ctx.enumeration_metadata.get<int64_t>("hsb_ip_version").value_or(0);
    int64_t target_version = std::stoi(ctx.target_version, nullptr, 16);
    ASSERT_NE(current_version, target_version);

    // Manifest found for this UUID
    ASSERT_TRUE(find_manifest_by_uuid(ctx));
    EXPECT_FALSE(ctx.firmware_info_path.empty());

    // Target version exists in manifest with valid firmware metadata
    ASSERT_TRUE(verify_firmware_details(ctx));
    EXPECT_FALSE(ctx.cpnx.location.empty());
    EXPECT_FALSE(ctx.cpnx.md5.empty());
    EXPECT_GT(ctx.cpnx.size, 0u);

    // Override remote URLs with local temp files for fetch verification
    TempFirmwareFile cpnx_file("fake cpnx firmware payload", "hsb_test_cpnx.bin");
    ctx.cpnx.location = cpnx_file.path();
    ctx.cpnx.size = cpnx_file.size();
    ctx.cpnx.md5 = cpnx_file.md5();

    TempFirmwareFile clnx_file("fake clnx firmware payload", "hsb_test_clnx.bin");
    ctx.clnx.location = clnx_file.path();
    ctx.clnx.size = clnx_file.size();
    ctx.clnx.md5 = clnx_file.md5();

    ASSERT_TRUE(fetch_target_firmware(ctx));
    EXPECT_FALSE(ctx.cpnx.local_location.empty());
    EXPECT_FALSE(ctx.clnx.local_location.empty());

    // Correct flasher is retrieved for this UUID/version
    auto flasher = get_flasher(ctx);
    EXPECT_NE(flasher, nullptr);
}

// ── Failure cases ────────────────────────────────────────────────────

// No device found at the given IP address
TEST(HsbFlasher, NoDeviceFound_ExpectFailure)
{
    auto ctx = make_context("192.168.0.2", "2507");
    EmptyProvider provider;

    EXPECT_FALSE(discover_device(ctx, provider));
}

// Multiple devices found with given IP address
TEST(HsbFlasher, MultipleDevicesAtSameIP_ExpectFailure)
{
    auto ctx = make_context("192.168.0.2", "2507");
    MultiDeviceProvider provider("192.168.0.2");

    EXPECT_FALSE(discover_device(ctx, provider));
}

// Same version as target — flash should not proceed
TEST(HsbFlasher, SameVersionAsTarget_ExpectFailure)
{
    auto ctx = make_context("192.168.0.2", "2507");
    SingleDeviceProvider provider(
        "192.168.0.2", "AA:BB:CC:DD:EE:FF", 0x2507,
        "889b7ce3-65a5-4247-8b05-4ff1904c3359");

    ASSERT_TRUE(discover_device(ctx, provider));

    int64_t current_version = ctx.enumeration_metadata.get<int64_t>("hsb_ip_version").value_or(0);
    int64_t target_version = std::stoi(ctx.target_version, nullptr, 16);
    EXPECT_EQ(current_version, target_version);
}

// UUID from enumeration doesn't match any YAML manifest
TEST(HsbFlasher, UnknownUUID_ExpectFailure)
{
    auto ctx = make_context("192.168.0.2", "2507");
    SingleDeviceProvider provider(
        "192.168.0.2", "AA:BB:CC:DD:EE:FF", 0x2504,
        "00000000-0000-0000-0000-000000000000");

    ASSERT_TRUE(discover_device(ctx, provider));
    EXPECT_FALSE(find_manifest_by_uuid(ctx));
}

// UUID matches but target version is not listed in the YAML manifest
TEST(HsbFlasher, VersionNotInManifest_ExpectFailure)
{
    auto ctx = make_context("192.168.0.2", "9999");
    SingleDeviceProvider provider(
        "192.168.0.2", "AA:BB:CC:DD:EE:FF", 0x2504,
        "889b7ce3-65a5-4247-8b05-4ff1904c3359");

    ASSERT_TRUE(discover_device(ctx, provider));
    ASSERT_TRUE(find_manifest_by_uuid(ctx));
    EXPECT_FALSE(verify_firmware_details(ctx));
}

// Firmware file does not exist at the given local path
TEST(HsbFlasher, FirmwareFileNotFound_ExpectFailure)
{
    auto ctx = make_verified_context();

    ctx.cpnx.location = "/tmp/nonexistent_firmware_file.bin";
    ctx.cpnx.size = 100;
    ctx.cpnx.md5 = "d41d8cd98f00b204e9800998ecf8427e";
    ctx.clnx.location.clear();

    EXPECT_FALSE(fetch_target_firmware(ctx));
}

// Firmware file exists but size doesn't match manifest
TEST(HsbFlasher, FirmwareSizeMismatch_ExpectFailure)
{
    auto ctx = make_verified_context();

    TempFirmwareFile fw("firmware with wrong size expectation");
    ctx.cpnx.location = fw.path();
    ctx.cpnx.size = fw.size() + 1;
    ctx.cpnx.md5 = fw.md5();
    ctx.clnx.location.clear();

    EXPECT_FALSE(fetch_target_firmware(ctx));
}

// Firmware file exists with correct size but MD5 doesn't match
TEST(HsbFlasher, FirmwareMd5Mismatch_ExpectFailure)
{
    auto ctx = make_verified_context();

    TempFirmwareFile fw("firmware with wrong md5 expectation");
    ctx.cpnx.location = fw.path();
    ctx.cpnx.size = fw.size();
    ctx.cpnx.md5 = "0000000000000000000000000000dead";
    ctx.clnx.location.clear();

    EXPECT_FALSE(fetch_target_firmware(ctx));
}

// No flash strategy supports the given UUID/version combination
TEST(HsbFlasher, NoFlasherForUnsupportedDevice_ExpectFailure)
{
    auto ctx = make_context("192.168.0.2", "2507");
    SingleDeviceProvider provider(
        "192.168.0.2", "AA:BB:CC:DD:EE:FF", 0x0001,
        "00000000-0000-0000-0000-000000000000");

    ASSERT_TRUE(discover_device(ctx, provider));

    auto flasher = get_flasher(ctx);
    EXPECT_EQ(flasher, nullptr);
}

// Device at a different IP than the one requested is ignored
TEST(HsbFlasher, DeviceAtDifferentIP_ExpectFailure)
{
    auto ctx = make_context("192.168.0.2", "2507");
    SingleDeviceProvider provider(
        "192.168.0.3", "AA:BB:CC:DD:EE:FF", 0x2504,
        "889b7ce3-65a5-4247-8b05-4ff1904c3359");

    EXPECT_FALSE(discover_device(ctx, provider));
}

// Duplicate packets from the same MAC are deduplicated — still counts as one device
TEST(HsbFlasher, DuplicatePacketsSameMac_ExpectSuccess)
{
    class DuplicatePacketProvider : public IEnumerationProvider {
    public:
        void enumerate(Callback callback, float) override
        {
            hololink::Metadata md;
            md["peer_ip"] = std::string("192.168.0.2");
            md["mac_id"] = std::string("AA:BB:CC:DD:EE:FF");
            md["hsb_ip_version"] = int64_t(0x2504);
            md["fpga_uuid"] = std::string("889b7ce3-65a5-4247-8b05-4ff1904c3359");
            md["data_plane"] = int64_t(0);

            std::vector<uint8_t> packet(64, 0);
            callback(packet, md);
            callback(packet, md);
        }
    };

    auto ctx = make_context("192.168.0.2", "2507");
    DuplicatePacketProvider provider;

    ASSERT_TRUE(discover_device(ctx, provider));
}
