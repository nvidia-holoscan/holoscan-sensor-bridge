/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Drives the Adapter's enumeration pipeline manually: feeds a
 * synthetic EnumerationMetadata in, expects Adapter::enumerate to
 * load a stub .so by UUID, route the metadata through the stub's
 * EnumerationInterfaceV1::update_metadata, and observe the enriched
 * result through Adapter::wait_for_channel.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "hololink/module/adapter.hpp"
#include "hololink/module/enumeration_metadata.hpp"

static std::filesystem::path test_module_dir()
{
    const char* dir = std::getenv("HOLOLINK_MODULE_TEST_MODULE_DIR");
    return dir ? std::filesystem::path(dir) : std::filesystem::current_path();
}

// Reserved test UUID per the plan; never matches a real device.
static constexpr const char* TEST_UUID = "01020304-0506-0708-090a-0b0c0d0e0f00";
static constexpr const char* PEER_IP = "10.0.0.42";

TEST(HololinkAdapterEnumeration, ManualEnumerateLoadsModuleAndEnriches)
{
    using hololink::module::Adapter;
    using hololink::module::EnumerationMetadata;

    Adapter::get_adapter().set_module_directory(test_module_dir());

    EnumerationMetadata metadata;
    metadata["fpga_uuid"] = std::string(TEST_UUID);
    metadata["peer_ip"] = std::string(PEER_IP);
    metadata["serial_number"] = std::string("test-serial-001");

    // wait_for_channel discards any previously cached announcement
    // before waiting, so the call to enumerate must happen after
    // wait_for_channel is already blocked. The helper thread sleeps
    // briefly to let the main thread reach the cv.wait inside
    // wait_for_channel before the synthetic enumerate fires.
    std::thread enum_thread([metadata]() mutable {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        Adapter::get_adapter().enumerate(std::move(metadata));
    });

    EnumerationMetadata found = Adapter::get_adapter().wait_for_channel(
        PEER_IP, std::chrono::seconds(1));
    enum_thread.join();

    // Original fields preserved.
    ASSERT_TRUE(found.contains("serial_number"));
    EXPECT_EQ(found.get<std::string>("serial_number"), "test-serial-001");

    ASSERT_TRUE(found.contains("fpga_uuid"));
    EXPECT_EQ(found.get<std::string>("fpga_uuid"), TEST_UUID);

    // Stub's update_metadata stamped these in.
    ASSERT_TRUE(found.contains("enriched_by_stub"));
    EXPECT_EQ(found.get<std::string>("enriched_by_stub"), "yes");

    ASSERT_TRUE(found.contains("control_port"));
    EXPECT_EQ(found.get<int64_t>("control_port"), 8192);

    // Adapter stamps the absolute path of the loaded .so so callers
    // can identify which module enriched the metadata.
    ASSERT_TRUE(found.contains("module_filename"));
    const std::string module_filename
        = found.get<std::string>("module_filename");
    EXPECT_NE(module_filename.find(std::string("hololink_") + TEST_UUID),
        std::string::npos);
    EXPECT_TRUE(std::filesystem::path(module_filename).is_absolute());
}

TEST(HololinkAdapterEnumeration, EnumerateRejectsMissingUuid)
{
    using hololink::module::Adapter;
    using hololink::module::EnumerationMetadata;

    Adapter::get_adapter().set_module_directory(test_module_dir());

    EnumerationMetadata metadata;
    metadata["peer_ip"] = std::string("10.0.0.99");
    EXPECT_THROW(Adapter::get_adapter().enumerate(metadata), std::runtime_error);
}

TEST(HololinkAdapterEnumeration, EnumerateStoresUnknownUuidUnenriched)
{
    using hololink::module::Adapter;
    using hololink::module::EnumerationMetadata;

    Adapter::get_adapter().set_module_directory(test_module_dir());

    // No .so exists for this UUID under the test module dir. Per the
    // plan, enumerate must still publish the bootp metadata unchanged
    // (no enrichment) rather than throw — applications can pick up
    // the unmodified record via wait_for_channel(peer_ip, ...) and
    // decide what to do.
    constexpr const char* unknown_uuid = "ffffffff-ffff-ffff-ffff-ffffffffffff";
    constexpr const char* unknown_peer_ip = "10.0.0.100";

    EnumerationMetadata metadata;
    metadata["fpga_uuid"] = std::string(unknown_uuid);
    metadata["peer_ip"] = std::string(unknown_peer_ip);

    // enumerate runs from a helper thread so wait_for_channel
    // observes it as a fresh announcement (its pre-wait erase would
    // otherwise discard a synchronously-posted entry).
    std::thread enum_thread([metadata]() mutable {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        EXPECT_NO_THROW(Adapter::get_adapter().enumerate(std::move(metadata)));
    });

    EnumerationMetadata observed = Adapter::get_adapter().wait_for_channel(
        unknown_peer_ip, std::chrono::seconds(1));
    enum_thread.join();
    EXPECT_EQ(observed.get<std::string>("fpga_uuid", std::string {}), unknown_uuid);
    EXPECT_EQ(observed.get<std::string>("peer_ip", std::string {}), unknown_peer_ip);
    // No module ran update_metadata so the test stub's enrichment
    // fields (e.g. "module_name") are absent.
    EXPECT_FALSE(observed.contains("module_name"));
}

TEST(HololinkAdapterEnumeration, WaitForChannelTimesOutOnUnknownIp)
{
    using hololink::module::Adapter;
    EXPECT_THROW(Adapter::get_adapter().wait_for_channel(
                     "198.51.100.1", std::chrono::milliseconds(50)),
        std::runtime_error);
}

// The bootp deserializer always writes a 16-byte hardware_address
// blob plus the meaningful prefix length. enumerate() must trim the
// blob to the length before any subscriber sees it.
TEST(HololinkAdapterEnumeration, EnumerateTrimsHardwareAddressToLength)
{
    using hololink::module::Adapter;
    using hololink::module::EnumerationCallbackHandle;
    using hololink::module::EnumerationMetadata;

    Adapter::get_adapter().set_module_directory(test_module_dir());

    std::vector<uint8_t> wire_blob(16, 0);
    wire_blob[0] = 0x02;
    wire_blob[1] = 0x34;
    wire_blob[2] = 0x56;
    wire_blob[3] = 0x78;
    wire_blob[4] = 0x9a;
    wire_blob[5] = 0xbc;

    EnumerationMetadata metadata;
    metadata["fpga_uuid"] = std::string(TEST_UUID);
    metadata["peer_ip"] = std::string("10.0.0.60");
    metadata["hardware_address"] = wire_blob;
    metadata["hardware_address_length"] = static_cast<int64_t>(6);

    std::optional<std::vector<uint8_t>> seen;
    EnumerationCallbackHandle handle = Adapter::get_adapter().register_raw_all(
        [&](const EnumerationMetadata& m) {
            if (m.contains("hardware_address")) {
                seen = m.get<std::vector<uint8_t>>("hardware_address");
            }
        });

    Adapter::get_adapter().enumerate(metadata);
    Adapter::get_adapter().unregister(handle);

    ASSERT_TRUE(seen.has_value());
    const std::vector<uint8_t> expected
        = { 0x02, 0x34, 0x56, 0x78, 0x9a, 0xbc };
    EXPECT_EQ(*seen, expected);
}

// register_raw_* fires before the module's update_metadata enrichment;
// the regular register_* subscribers fire after, with the enriched
// metadata. Both must observe the same announcement.
TEST(HololinkAdapterEnumeration, RawSubscriberSeesPreEnrichmentMetadata)
{
    using hololink::module::Adapter;
    using hololink::module::EnumerationCallbackHandle;
    using hololink::module::EnumerationMetadata;

    Adapter::get_adapter().set_module_directory(test_module_dir());

    EnumerationMetadata metadata;
    metadata["fpga_uuid"] = std::string(TEST_UUID);
    metadata["peer_ip"] = std::string("10.0.0.50");
    metadata["serial_number"] = std::string("test-serial-raw");

    std::mutex mutex;
    std::vector<std::string> order; // "raw" / "enriched", in fire order
    std::optional<EnumerationMetadata> raw_seen;
    std::optional<EnumerationMetadata> enriched_seen;

    EnumerationCallbackHandle raw_handle = Adapter::get_adapter().register_raw_all(
        [&](const EnumerationMetadata& m) {
            std::lock_guard<std::mutex> guard(mutex);
            order.push_back("raw");
            raw_seen = m;
        });
    EnumerationCallbackHandle enriched_handle = Adapter::get_adapter().register_all(
        [&](const EnumerationMetadata& m) {
            std::lock_guard<std::mutex> guard(mutex);
            order.push_back("enriched");
            enriched_seen = m;
        });

    Adapter::get_adapter().enumerate(metadata);
    Adapter::get_adapter().unregister(raw_handle);
    Adapter::get_adapter().unregister(enriched_handle);

    ASSERT_EQ(order.size(), 2u);
    EXPECT_EQ(order[0], "raw");
    EXPECT_EQ(order[1], "enriched");

    ASSERT_TRUE(raw_seen.has_value());
    // Pre-enrichment: the stub's update_metadata never ran on this
    // view and the adapter hasn't stamped module_filename yet.
    EXPECT_FALSE(raw_seen->contains("enriched_by_stub"));
    EXPECT_FALSE(raw_seen->contains("module_filename"));
    EXPECT_EQ(raw_seen->get<std::string>("fpga_uuid", std::string {}), TEST_UUID);

    ASSERT_TRUE(enriched_seen.has_value());
    // Post-enrichment: the stub stamped this in and the adapter
    // recorded which .so it loaded.
    EXPECT_EQ(enriched_seen->get<std::string>("enriched_by_stub", std::string {}), "yes");
    EXPECT_TRUE(enriched_seen->contains("module_filename"));
}
