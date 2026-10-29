/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Drives the Leopard VB1940 modules end-to-end: feeds an
 * EnumerationMetadata with the Leopard VB1940 UUID and a per-revision
 * compat_id through Adapter::enumerate, expects the adapter to load
 * the matching .so (module/leopard_vb1940/ as the compat-suffixed
 * hololink_<UUID>_2603.so for the current FPGA, or
 * module/leopard_vb1940_2510/ as the bare-name hololink_<UUID>.so for
 * the pre-0x2602 FPGA, which reports compat_id 0), run its
 * EnumerationInterfaceV1::update_metadata, and store the enriched
 * result. Also covers the 2510 module's FPGA-version gate — it
 * declines silicon newer than it understands, and deliberately accepts
 * anything older, since it is the catch-all for the UUID. Finally,
 * pins the Leopard per-sensor addressing: enumerate (update_metadata)
 * and Adapter::use_sensor route through one formula, so the two paths
 * agree field-for-field; out-of-range sensors are rejected.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>

#include "hololink/module/adapter.hpp"
#include "hololink/module/enumeration_metadata.hpp"

static std::filesystem::path test_module_dir()
{
    const char* dir = std::getenv("HOLOLINK_MODULE_TEST_MODULE_DIR");
    return dir ? std::filesystem::path(dir) : std::filesystem::current_path();
}

// Real Leopard VB1940 UUID — both module/leopard_vb1940/ and
// module/leopard_vb1940_2510/ claim it, distinguished by the
// per-compat-id .so suffix.
static constexpr const char* LEOPARD_VB1940_UUID
    = "f1627640-b4dc-48af-a360-c55b09b3d230";

/* Same staging discipline as hololink_module_hsb_lite_test: every test
 * here drives enumeration through synthetic metadata handed straight to
 * Adapter::enumerate, and the staged peer IPs are RFC 5737 TEST-NET-1
 * (192.0.2.0/24) so a real board on the CI runner's network can never
 * claim one. The listener is additionally rebound to an ephemeral port,
 * so nothing off the wire (boards broadcast to 12267) reaches this
 * process at all.
 */
class HololinkModuleLeopardVb1940 : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        auto& adapter = hololink::module::Adapter::get_adapter();
        adapter.stop_bootp_listener();
        adapter.start_bootp_listener(0);
    }

    void SetUp() override
    {
        hololink::module::Adapter::get_adapter().set_module_directory(
            test_module_dir());
    }

    // Enumerate `metadata` off-thread and return the enriched record the
    // adapter announced for `peer_ip`.
    static hololink::module::EnumerationMetadata enumerate_and_wait(
        hololink::module::EnumerationMetadata metadata,
        const std::string& peer_ip)
    {
        using hololink::module::Adapter;
        std::thread enum_thread([metadata]() mutable {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            Adapter::get_adapter().enumerate(std::move(metadata));
        });
        auto found = Adapter::get_adapter().wait_for_channel(
            peer_ip, std::chrono::seconds(1));
        enum_thread.join();
        return found;
    }
};

TEST_F(HololinkModuleLeopardVb1940, EnumerateLoads2603Module)
{
    using hololink::module::Adapter;
    using hololink::module::EnumerationMetadata;

    EnumerationMetadata metadata;
    metadata["fpga_uuid"] = std::string(LEOPARD_VB1940_UUID);
    metadata["peer_ip"] = std::string("192.0.2.20");
    metadata["serial_number"] = std::string("leopard-001");
    metadata["compat_id"] = static_cast<std::int64_t>(0x2603);
    metadata["hsb_ip_version"] = static_cast<std::int64_t>(0x2603);
    metadata["data_plane"] = static_cast<std::int64_t>(0); // bootp supplies this

    EnumerationMetadata found = enumerate_and_wait(metadata, "192.0.2.20");

    // compat_id 0x2603 resolves the compat-suffixed .so.
    ASSERT_TRUE(found.contains("module_name"));
    EXPECT_EQ(found.get<std::string>("module_name"), "leopard_vb1940");
    EXPECT_EQ(found.get<int64_t>("compat_id"), 0x2603);

    auto module_handle = Adapter::get_adapter().get_module(found);
    ASSERT_NE(module_handle, nullptr);
}

TEST_F(HololinkModuleLeopardVb1940, ZeroCompatIdFallsThroughTo2510Module)
{
    using hololink::module::Adapter;
    using hololink::module::EnumerationMetadata;

    // The pre-0x2602 board reports compat_id 0. There is no
    // hololink_<UUID>_0000.so, so the loader falls through to the bare
    // hololink_<UUID>.so — module/leopard_vb1940_2510/.
    EnumerationMetadata metadata;
    metadata["fpga_uuid"] = std::string(LEOPARD_VB1940_UUID);
    metadata["peer_ip"] = std::string("192.0.2.21");
    metadata["serial_number"] = std::string("leopard-002");
    metadata["compat_id"] = static_cast<std::int64_t>(0);
    metadata["hsb_ip_version"] = static_cast<std::int64_t>(0x2510);
    metadata["data_plane"] = static_cast<std::int64_t>(0);

    EnumerationMetadata found = enumerate_and_wait(metadata, "192.0.2.21");

    ASSERT_TRUE(found.contains("module_name"));
    EXPECT_EQ(found.get<std::string>("module_name"), "leopard_vb1940_2510");
    EXPECT_EQ(found.get<int64_t>("compat_id"), 0); // bootp value preserved

    auto module_handle = Adapter::get_adapter().get_module(found);
    ASSERT_NE(module_handle, nullptr);
}

TEST_F(HololinkModuleLeopardVb1940, MissingCompatIdFallsThroughTo2510Module)
{
    using hololink::module::EnumerationMetadata;

    // No compat_id at all takes the same no-compat-id fallback path.
    EnumerationMetadata metadata;
    metadata["fpga_uuid"] = std::string(LEOPARD_VB1940_UUID);
    metadata["peer_ip"] = std::string("192.0.2.22");
    metadata["serial_number"] = std::string("leopard-003");
    metadata["data_plane"] = static_cast<std::int64_t>(0);

    EnumerationMetadata found = enumerate_and_wait(metadata, "192.0.2.22");

    ASSERT_TRUE(found.contains("module_name"));
    EXPECT_EQ(found.get<std::string>("module_name"), "leopard_vb1940_2510");
}

TEST_F(HololinkModuleLeopardVb1940, EnumerateSkipsNewerIpVersion)
{
    using hololink::module::Adapter;
    using hololink::module::EnumerationMetadata;

    // The bare .so is the catch-all for this UUID, so it sees newer
    // silicon this build predates. 0x2700 is past the 0x2603 the module
    // understands, so update_metadata returns
    // HOLOLINK_MODULE_ENUMERATION_SKIPPED.
    EnumerationMetadata metadata;
    metadata["fpga_uuid"] = std::string(LEOPARD_VB1940_UUID);
    metadata["peer_ip"] = std::string("192.0.2.23");
    metadata["serial_number"] = std::string("leopard-004");
    metadata["compat_id"] = static_cast<std::int64_t>(0);
    metadata["hsb_ip_version"] = static_cast<std::int64_t>(0x2700);
    metadata["data_plane"] = static_cast<std::int64_t>(0);

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

TEST_F(HololinkModuleLeopardVb1940, EnumerateAcceptsOlderIpVersion)
{
    using hololink::module::EnumerationMetadata;

    // Deliberately no lower bound on the gate: a 2510 VB1940 is known
    // only to report some version below 0x2603, and this module is the
    // board's catch-all, so a low version must still enumerate. (The
    // HSB-Lite 2510 module, which is not its UUID's only module, does
    // decline below 0x2510.)
    EnumerationMetadata metadata;
    metadata["fpga_uuid"] = std::string(LEOPARD_VB1940_UUID);
    metadata["peer_ip"] = std::string("192.0.2.24");
    metadata["serial_number"] = std::string("leopard-005");
    metadata["compat_id"] = static_cast<std::int64_t>(0);
    metadata["hsb_ip_version"] = static_cast<std::int64_t>(0x2400);
    metadata["data_plane"] = static_cast<std::int64_t>(0);

    EnumerationMetadata found = enumerate_and_wait(metadata, "192.0.2.24");

    ASSERT_TRUE(found.contains("module_name"));
    EXPECT_EQ(found.get<std::string>("module_name"), "leopard_vb1940_2510");
}

TEST_F(HololinkModuleLeopardVb1940, UseSensorStampsAioLayout)
{
    using hololink::module::Adapter;
    using hololink::module::EnumerationMetadata;

    // use_sensor loads the module on demand from fpga_uuid + compat_id;
    // no prior enumerate needed.
    EnumerationMetadata metadata;
    metadata["fpga_uuid"] = std::string(LEOPARD_VB1940_UUID);
    metadata["compat_id"] = static_cast<std::int64_t>(0x2603);

    // Every sensor sits on the single physical data plane 0, one SIF
    // each, with its own camera I2C bus.
    Adapter::get_adapter().use_sensor(metadata, 0);
    EXPECT_EQ(metadata.get<int64_t>("hif_address"), 0x02000300);
    EXPECT_EQ(metadata.get<int64_t>("sensor_number"), 0);
    EXPECT_EQ(metadata.get<int64_t>("sensor"), 0);
    EXPECT_EQ(metadata.get<int64_t>("i2c_bus"), 1);

    Adapter::get_adapter().use_sensor(metadata, 1);
    EXPECT_EQ(metadata.get<int64_t>("hif_address"), 0x02000300);
    EXPECT_EQ(metadata.get<int64_t>("sensor_number"), 1);
    EXPECT_EQ(metadata.get<int64_t>("sensor"), 1); // sensor_number * sifs(1)
    EXPECT_EQ(metadata.get<int64_t>("sif_address"), 0x01010000);
    EXPECT_EQ(metadata.get<int64_t>("data_channel"), 1);
    EXPECT_EQ(metadata.get<int64_t>("i2c_bus"), 2);
}

TEST_F(HololinkModuleLeopardVb1940, EnumerateMatchesUseSensorFieldForField)
{
    using hololink::module::Adapter;
    using hololink::module::EnumerationMetadata;

    // Enumerate with bootp data_plane = 1. update_metadata delegates to
    // use_sensor(1), so the enriched metadata must equal what an explicit
    // use_sensor(1) produces — the two paths can no longer drift.
    EnumerationMetadata metadata;
    metadata["fpga_uuid"] = std::string(LEOPARD_VB1940_UUID);
    metadata["peer_ip"] = std::string("192.0.2.25");
    metadata["serial_number"] = std::string("leopard-006");
    metadata["compat_id"] = static_cast<std::int64_t>(0x2603);
    metadata["data_plane"] = static_cast<std::int64_t>(1);

    EnumerationMetadata enumerated = enumerate_and_wait(metadata, "192.0.2.25");

    EnumerationMetadata configured;
    configured["fpga_uuid"] = std::string(LEOPARD_VB1940_UUID);
    configured["compat_id"] = static_cast<std::int64_t>(0x2603);
    configured["data_plane"] = static_cast<std::int64_t>(1);
    Adapter::get_adapter().use_sensor(configured, 1);

    for (const char* key : { "sensor_number", "sensor", "vp_mask",
             "sif_address", "vp_address", "hif_address", "data_channel",
             "i2c_bus", "frame_end_event" }) {
        ASSERT_TRUE(enumerated.contains(key)) << "enumerated missing " << key;
        ASSERT_TRUE(configured.contains(key)) << "configured missing " << key;
        EXPECT_EQ(enumerated.get<int64_t>(key), configured.get<int64_t>(key))
            << "field mismatch on '" << key << "'";
    }
}

TEST_F(HololinkModuleLeopardVb1940, UseSensorRejectsOutOfRange)
{
    using hololink::module::Adapter;
    using hololink::module::EnumerationMetadata;

    EnumerationMetadata metadata;
    metadata["fpga_uuid"] = std::string(LEOPARD_VB1940_UUID);
    metadata["compat_id"] = static_cast<std::int64_t>(0x2603);

    // The AIO carrier wires three camera positions; sensor 3 is out of
    // range on both modules.
    EXPECT_THROW(Adapter::get_adapter().use_sensor(metadata, 3),
        std::runtime_error);

    EnumerationMetadata fallback;
    fallback["fpga_uuid"] = std::string(LEOPARD_VB1940_UUID);
    fallback["compat_id"] = static_cast<std::int64_t>(0);
    EXPECT_THROW(Adapter::get_adapter().use_sensor(fallback, 3),
        std::runtime_error);
}
