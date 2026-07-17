/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Drives the HSB-Lite modules end-to-end: feeds an EnumerationMetadata
 * with the HSB-Lite UUID and a per-revision compat_id through
 * Adapter::enumerate, expects the adapter to load the matching .so
 * (module/hsb_lite/ as the compat-suffixed hololink_<UUID>_2603.so
 * for compat 0x2603, or module/hsb_lite_2510/ as the bare-name
 * hololink_<UUID>.so for any compat without a dedicated .so), run
 * its EnumerationInterfaceV1::update_metadata, and store the
 * enriched result. compat_id values in metadata are the numeric
 * wire form (0x2603 etc.); the .so filename's compat suffix, when
 * present, uses the matching 4-digit lowercase hex string. Verifies
 * the FrameMetadataInterfaceV1 singleton is reachable on the loaded
 * module via the typed get_service path. Also covers the 2510
 * module declining an unsupported FPGA IP version: enumerate must
 * suppress the announcement (no post-enrichment callback) without
 * throwing. Finally, pins the unified per-sensor addressing: enumerate
 * (update_metadata) and Adapter::use_sensor route through one formula,
 * so use_sensor recomputes hif_address per data plane and the two
 * paths agree field-for-field; out-of-range sensors are rejected.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "hololink/module/adapter.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/frame_metadata.hpp"

static std::filesystem::path test_module_dir()
{
    const char* dir = std::getenv("HOLOLINK_MODULE_TEST_MODULE_DIR");
    return dir ? std::filesystem::path(dir) : std::filesystem::current_path();
}

// Real HSB-Lite UUID — both module/hsb_lite/ and module/hsb_lite_2510/
// claim this UUID, distinguished by the per-compat-id .so suffix.
static constexpr const char* HSB_LITE_UUID = "889b7ce3-65a5-4247-8b05-4ff1904c3359";

TEST(HololinkAdapterHsbLite, EnumerateLoadsHsbLite2603AndEnriches)
{
    using hololink::module::Adapter;
    using hololink::module::EnumerationMetadata;

    Adapter::get_adapter().set_module_directory(test_module_dir());

    EnumerationMetadata metadata;
    metadata["fpga_uuid"] = std::string(HSB_LITE_UUID);
    metadata["peer_ip"] = std::string("192.168.0.2");
    metadata["serial_number"] = std::string("hsb-lite-001");
    metadata["compat_id"] = static_cast<std::int64_t>(0x2603);
    metadata["data_plane"] = static_cast<std::int64_t>(0); // bootp supplies this

    std::thread enum_thread([metadata]() mutable {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        Adapter::get_adapter().enumerate(std::move(metadata));
    });
    EnumerationMetadata found = Adapter::get_adapter().wait_for_channel(
        "192.168.0.2", std::chrono::seconds(1));
    enum_thread.join();

    ASSERT_TRUE(found.contains("module_name"));
    EXPECT_EQ(found.get<std::string>("module_name"), "hsb_lite");

    ASSERT_TRUE(found.contains("compat_id"));
    EXPECT_EQ(found.get<int64_t>("compat_id"), 0x2603); // bootp-supplied value preserved

    // get_module(metadata) resolves to the same cached Module that
    // enumerate() loaded, without the caller composing a .so path.
    auto module_handle = Adapter::get_adapter().get_module(found);
    ASSERT_NE(module_handle, nullptr);
}

TEST(HololinkAdapterHsbLite, EnumerateLoadsHsbLite2510AndPreservesCompat)
{
    using hololink::module::Adapter;
    using hololink::module::EnumerationMetadata;

    Adapter::get_adapter().set_module_directory(test_module_dir());

    EnumerationMetadata metadata;
    metadata["fpga_uuid"] = std::string(HSB_LITE_UUID);
    metadata["peer_ip"] = std::string("192.168.0.3");
    metadata["serial_number"] = std::string("hsb-lite-002");
    metadata["compat_id"] = static_cast<std::int64_t>(0x2510);
    metadata["data_plane"] = static_cast<std::int64_t>(0); // bootp supplies this

    std::thread enum_thread([metadata]() mutable {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        Adapter::get_adapter().enumerate(std::move(metadata));
    });
    EnumerationMetadata found = Adapter::get_adapter().wait_for_channel(
        "192.168.0.3", std::chrono::seconds(1));
    enum_thread.join();

    // The 2510 supplement chooses its own module_name (it is not
    // required to match the canonical "hsb_lite"); this test pins
    // the compat-id passthrough and the .so-resolution fallback, not
    // the module_name spelling.
    ASSERT_TRUE(found.contains("compat_id"));
    EXPECT_EQ(found.get<int64_t>("compat_id"), 0x2510); // bootp-supplied value preserved

    // For compat_id=0x2510 the loader falls back to the bare .so
    // (module/hsb_lite_2510/). get_module(metadata) goes through the
    // same lookup and returns the same cached Module.
    auto module_handle = Adapter::get_adapter().get_module(found);
    ASSERT_NE(module_handle, nullptr);
}

TEST(HololinkAdapterHsbLite, EnumerateSkipsUnsupportedIpVersion)
{
    using hololink::module::Adapter;
    using hololink::module::EnumerationMetadata;

    Adapter::get_adapter().set_module_directory(test_module_dir());

    // No dedicated .so for this compat, so the loader falls through to
    // the bare hololink_<UUID>.so (module/hsb_lite_2510/). That module
    // drives FPGA IP versions 0x2510 through 0x2603; 0x2700 is newer
    // silicon it cannot drive, so its update_metadata returns
    // HOLOLINK_MODULE_ENUMERATION_SKIPPED.
    EnumerationMetadata metadata;
    metadata["fpga_uuid"] = std::string(HSB_LITE_UUID);
    metadata["peer_ip"] = std::string("192.168.0.5");
    metadata["serial_number"] = std::string("hsb-lite-004");
    metadata["compat_id"] = static_cast<std::int64_t>(0x2510);
    metadata["hsb_ip_version"] = static_cast<std::int64_t>(0x2700);

    // A post-enrichment subscriber must NOT be notified for a skipped
    // device (a raw subscriber would still fire, but that runs before
    // the module is consulted).
    bool enriched_fired = false;
    auto handle = Adapter::get_adapter().register_all(
        [&enriched_fired](const EnumerationMetadata&) { enriched_fired = true; });

    // The skip is not an error: enumerate must not throw, and the
    // announcement is withheld from the application.
    EXPECT_NO_THROW(Adapter::get_adapter().enumerate(metadata));
    EXPECT_FALSE(enriched_fired);

    Adapter::get_adapter().unregister(handle);
}

TEST(HololinkAdapterHsbLite, UseSensorRecomputesHifAddress)
{
    using hololink::module::Adapter;
    using hololink::module::EnumerationMetadata;

    Adapter::get_adapter().set_module_directory(test_module_dir());

    // use_sensor loads the module on demand from fpga_uuid + compat_id;
    // no prior enumerate needed.
    EnumerationMetadata metadata;
    metadata["fpga_uuid"] = std::string(HSB_LITE_UUID);
    metadata["compat_id"] = static_cast<std::int64_t>(0x2603);

    // Sensor 0 lives on data plane 0.
    Adapter::get_adapter().use_sensor(metadata, 0);
    EXPECT_EQ(metadata.get<int64_t>("hif_address"), 0x02000300);
    EXPECT_EQ(metadata.get<int64_t>("sensor_number"), 0);

    // Retarget to sensor 1: HSB-Lite maps it 1:1 onto data plane 1, so
    // hif_address MUST move with it. The pre-unification use_sensor path
    // (legacy DataChannel::use_sensor) left hif_address stale here — this
    // is the regression guard for that bug.
    Adapter::get_adapter().use_sensor(metadata, 1);
    EXPECT_EQ(metadata.get<int64_t>("hif_address"), 0x02010300);
    EXPECT_EQ(metadata.get<int64_t>("sensor_number"), 1);
    EXPECT_EQ(metadata.get<int64_t>("sensor"), 2); // sensor_number * sifs(2)
    EXPECT_EQ(metadata.get<int64_t>("sif_address"), 0x01010000);
    EXPECT_EQ(metadata.get<int64_t>("data_channel"), 1);
}

TEST(HololinkAdapterHsbLite, EnumerateMatchesUseSensorFieldForField)
{
    using hololink::module::Adapter;
    using hololink::module::EnumerationMetadata;

    Adapter::get_adapter().set_module_directory(test_module_dir());

    // Enumerate with bootp data_plane = 1. update_metadata delegates to
    // use_sensor(1), so the enriched metadata must equal what an explicit
    // use_sensor(1) produces — the two paths can no longer drift.
    EnumerationMetadata metadata;
    metadata["fpga_uuid"] = std::string(HSB_LITE_UUID);
    metadata["peer_ip"] = std::string("192.168.0.7");
    metadata["serial_number"] = std::string("hsb-lite-006");
    metadata["compat_id"] = static_cast<std::int64_t>(0x2603);
    metadata["data_plane"] = static_cast<std::int64_t>(1);

    std::thread enum_thread([metadata]() mutable {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        Adapter::get_adapter().enumerate(std::move(metadata));
    });
    EnumerationMetadata enumerated = Adapter::get_adapter().wait_for_channel(
        "192.168.0.7", std::chrono::seconds(1));
    enum_thread.join();

    // Independently retarget a fresh metadata to sensor 1 via use_sensor.
    EnumerationMetadata configured;
    configured["fpga_uuid"] = std::string(HSB_LITE_UUID);
    configured["compat_id"] = static_cast<std::int64_t>(0x2603);
    Adapter::get_adapter().use_sensor(configured, 1);

    for (const char* key : { "sensor_number", "sensor", "vp_mask",
             "sif_address", "vp_address", "hif_address", "data_channel",
             "frame_end_event" }) {
        ASSERT_TRUE(enumerated.contains(key)) << "enumerated missing " << key;
        ASSERT_TRUE(configured.contains(key)) << "configured missing " << key;
        EXPECT_EQ(enumerated.get<int64_t>(key), configured.get<int64_t>(key))
            << "field mismatch on '" << key << "'";
    }
}

TEST(HololinkAdapterHsbLite, UseSensorRejectsOutOfRange)
{
    using hololink::module::Adapter;
    using hololink::module::EnumerationMetadata;

    Adapter::get_adapter().set_module_directory(test_module_dir());

    EnumerationMetadata metadata;
    metadata["fpga_uuid"] = std::string(HSB_LITE_UUID);
    metadata["compat_id"] = static_cast<std::int64_t>(0x2603);

    // HSB-Lite supports sensors 0..2; sensor 3 is out of range. The
    // unified use_sensor validates (the old set_data_plane_metadata
    // enumerate path did not).
    EXPECT_THROW(Adapter::get_adapter().use_sensor(metadata, 3),
        std::runtime_error);
}

TEST(HololinkAdapterHsbLite, FrameMetadataSingletonIsReachable)
{
    using hololink::module::Adapter;
    using hololink::module::EnumerationMetadata;
    using hololink::module::FrameMetadataInterfaceV1;

    Adapter::get_adapter().set_module_directory(test_module_dir());

    EnumerationMetadata metadata;
    metadata["fpga_uuid"] = std::string(HSB_LITE_UUID);
    metadata["peer_ip"] = std::string("192.168.0.4");
    metadata["serial_number"] = std::string("hsb-lite-003");
    metadata["compat_id"] = static_cast<std::int64_t>(0x2603);
    metadata["data_plane"] = static_cast<std::int64_t>(0); // bootp supplies this
    Adapter::get_adapter().enumerate(metadata);

    // load_module by absolute path returns the same shared_ptr<Module>
    // that enumerate populated under the cache.
    std::filesystem::path module_so = test_module_dir();
    module_so.append(std::string("hololink_") + HSB_LITE_UUID + "_2603.so");
    auto module = Adapter::get_adapter().load_module(module_so);
    auto frame_metadata = FrameMetadataInterfaceV1::get_service(module);
    ASSERT_NE(frame_metadata, nullptr);

    // Decode a known 48-byte block to prove the singleton is the
    // actual FrameMetadataV1 from module/core/.
    std::vector<std::uint8_t> block(48, 0);
    block[0] = 0xAA; // flags top byte
    FrameMetadataInterfaceV1::FrameMetadata out {};
    EXPECT_EQ(frame_metadata->decode(block.data(), block.size(), out),
        HOLOLINK_MODULE_OK);
    EXPECT_EQ(out.flags, 0xAA000000u);
}
