/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * End-to-end test for the Adapter's bootp listener: hand-craft a
 * bootp v2 packet with a stub UUID, send it to the listener via
 * sendto(127.0.0.1, port), and verify that the stub module's
 * EnumerationInterfaceV1::update_metadata fired with the parsed
 * fields and the original raw packet bytes.
 */

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "hololink/module/adapter.hpp"
#include "hololink/module/enumeration_metadata.hpp"

#include "hololink_module_bootp_capture.hpp"

// Fixed UUID for this test's stub module. Bytes appear in this order
// in the bootp packet's v2 fpga_uuid field; next_uuid_as_string
// formats them as 8-4-4-4-12.
static const std::uint8_t STUB_UUID_BYTES[16] = {
    0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
    0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00
};
static constexpr const char* STUB_UUID_STRING
    = "11223344-5566-7788-99aa-bbccddeeff00";

// Outside the default Linux ephemeral range (32768-60999) so kernel
// auto-allocations don't collide.
static constexpr std::uint32_t TEST_PORT = 18267;

// 16-bit wire value; matches the COMPAT 2603 the bootp-stub module
// is built with, since the .so filename uses the 4-digit hex
// rendering "2603".
static constexpr std::uint16_t TEST_COMPAT_ID = 0x2603;
static constexpr std::uint8_t TEST_DATA_PLANE = 0;

// HSB-Lite serial numbers are 7 raw bytes — the on-the-wire field is
// binary, not text. Real boards stamp their actual hardware serial.
// The legacy parser's `next_buffer_as_string` formats each byte as
// two lowercase hex characters and concatenates, so the 14-character
// hex string below is what the supplement (and the test) observe in
// the metadata.
static constexpr std::array<std::uint8_t, 7> TEST_SERIAL_WIRE_BYTES = {
    0x74, 0x73, 0x74, 0x2d, 0x30, 0x30, 0x31
};
static constexpr const char* TEST_SERIAL_DECODED = "7473742d303031";

static std::filesystem::path test_module_dir()
{
    const char* dir = std::getenv("HOLOLINK_MODULE_TEST_MODULE_DIR");
    return dir ? std::filesystem::path(dir) : std::filesystem::current_path();
}

static void put_u8(std::vector<std::uint8_t>& p, std::uint8_t v)
{
    p.push_back(v);
}
static void put_u16_be(std::vector<std::uint8_t>& p, std::uint16_t v)
{
    p.push_back(static_cast<std::uint8_t>(v >> 8));
    p.push_back(static_cast<std::uint8_t>(v & 0xff));
}
static void put_u32_be(std::vector<std::uint8_t>& p, std::uint32_t v)
{
    p.push_back(static_cast<std::uint8_t>((v >> 24) & 0xff));
    p.push_back(static_cast<std::uint8_t>((v >> 16) & 0xff));
    p.push_back(static_cast<std::uint8_t>((v >> 8) & 0xff));
    p.push_back(static_cast<std::uint8_t>(v & 0xff));
}
static void put_u16_le(std::vector<std::uint8_t>& p, std::uint16_t v)
{
    p.push_back(static_cast<std::uint8_t>(v & 0xff));
    p.push_back(static_cast<std::uint8_t>((v >> 8) & 0xff));
}
static void put_u32_le(std::vector<std::uint8_t>& p, std::uint32_t v)
{
    p.push_back(static_cast<std::uint8_t>(v & 0xff));
    p.push_back(static_cast<std::uint8_t>((v >> 8) & 0xff));
    p.push_back(static_cast<std::uint8_t>((v >> 16) & 0xff));
    p.push_back(static_cast<std::uint8_t>((v >> 24) & 0xff));
}
static void put_zeros(std::vector<std::uint8_t>& p, std::size_t n)
{
    p.insert(p.end(), n, 0);
}

static std::vector<std::uint8_t> build_bootp_v2(
    const std::uint8_t (&uuid_bytes)[16],
    const std::array<std::uint8_t, 7>& serial,
    std::uint16_t compat_id,
    std::uint8_t data_plane)
{
    std::vector<std::uint8_t> p;
    p.reserve(277);

    // BOOTP fixed header (28 bytes).
    put_u8(p, 1); // op = BOOTREQUEST
    put_u8(p, 1); // hardware_type = ethernet
    put_u8(p, 6); // hardware_address_length
    put_u8(p, 0); // hops
    put_u32_be(p, 0xCAFEBABE); // transaction_id
    put_u16_be(p, 0); // seconds
    put_u16_be(p, 0); // flags
    put_u32_be(p, 0); // client_ip
    put_u32_be(p, 0); // your_ip
    put_u32_be(p, 0); // server_ip
    put_u32_be(p, 0); // gateway_ip

    // hardware_address (16 bytes: 6-byte MAC + 10 zero pad).
    put_u8(p, 0x02);
    put_u8(p, 0x00);
    put_u8(p, 0x00);
    put_u8(p, 0x00);
    put_u8(p, 0x00);
    put_u8(p, 0x01);
    put_zeros(p, 10);

    put_zeros(p, 64); // server_hostname
    put_zeros(p, 128); // boot_filename

    // Vendor section (8 bytes).
    put_u8(p, 0xE0); // vendor_tag (parser asserts this exact value)
    put_u8(p, 33); // vendor_tag_length (informational; parser doesn't enforce)
    put_u32_be(p, 0x4E564441); // vendor_id 'NVDA' (parser asserts)
    put_u8(p, data_plane);
    put_u8(p, 2); // enum_version = 2

    // BOOTP v2 fields (33 bytes).
    put_u16_le(p, compat_id);
    for (std::uint8_t b : uuid_bytes) {
        put_u8(p, b);
    }
    put_u32_le(p, 0); // transmitted_packet_count
    for (std::uint8_t byte : serial) {
        put_u8(p, byte);
    }
    put_u16_le(p, 0); // hsb_ip_version
    put_u16_le(p, 0); // fpga_crc

    return p;
}

TEST(HololinkAdapterBootpEndToEnd, ParsesV2PacketAndDeliversThroughEnumerate)
{
    using hololink::module::Adapter;
    using hololink::module::Module;

    auto& adapter = Adapter::get_adapter();
    adapter.set_module_directory(test_module_dir());

    // Pre-load the stub by its absolute path so the BootpCaptureV1
    // service is reachable before the listener fires. Adapter caches
    // by absolute path; the bootp callback's load_module_for(uuid)
    // resolves to the same Module via the UUID-keyed lookup against
    // the same module directory.
    std::filesystem::path stub_so = test_module_dir();
    stub_so.append(std::string("hololink_") + STUB_UUID_STRING + "_2603.so");
    auto module = adapter.load_module(stub_so);
    ASSERT_NE(module, nullptr);

    auto capture = test::BootpCaptureV1::get_service(module, "");
    ASSERT_NE(capture, nullptr);

    // The constructor already started the listener on the default
    // port; stop it so we can bind TEST_PORT (which the synthetic
    // sender writes to).
    adapter.stop_bootp_listener();
    adapter.start_bootp_listener(TEST_PORT);

    // enumerate() skips the module .so load + update_metadata
    // enrichment when no post-enrichment subscriber is registered.
    // Register a no-op register_all so the enrichment path runs and
    // the stub's update_metadata fires.
    auto registration = adapter.register_all(
        [](const hololink::module::EnumerationMetadata&) {});

    std::vector<std::uint8_t> packet = build_bootp_v2(
        STUB_UUID_BYTES, TEST_SERIAL_WIRE_BYTES, TEST_COMPAT_ID, TEST_DATA_PLANE);

    int sender = ::socket(AF_INET, SOCK_DGRAM, 0);
    ASSERT_GE(sender, 0);
    sockaddr_in dest {};
    dest.sin_family = AF_INET;
    dest.sin_port = htons(static_cast<std::uint16_t>(TEST_PORT));
    dest.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    const ssize_t sent = ::sendto(sender, packet.data(), packet.size(), 0,
        reinterpret_cast<sockaddr*>(&dest), sizeof(dest));
    EXPECT_EQ(sent, static_cast<ssize_t>(packet.size()));
    ::close(sender);

    EXPECT_TRUE(capture->wait_for_metadata(/*timeout_ms=*/2000));

    EXPECT_EQ(capture->fpga_uuid(), STUB_UUID_STRING);
    EXPECT_EQ(capture->serial_number(), TEST_SERIAL_DECODED);
    EXPECT_EQ(capture->compat_id(), TEST_COMPAT_ID);
    EXPECT_EQ(capture->data_plane(), TEST_DATA_PLANE);
    EXPECT_EQ(capture->peer_ip(), "127.0.0.1");
    EXPECT_EQ(capture->raw_packet(), packet);

    adapter.unregister(registration);
    adapter.stop_bootp_listener();
}
