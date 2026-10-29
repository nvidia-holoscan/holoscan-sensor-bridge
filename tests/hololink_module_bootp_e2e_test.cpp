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
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <system_error>
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

namespace {

using hololink::module::Adapter;
using hololink::module::EnumerationCallbackHandle;
using hololink::module::EnumerationMetadata;

// Fd numbers of the sockets this process has open. Pipes, files and the
// directory fd used to read /proc/self/fd are excluded, so the difference
// between two snapshots is the sockets opened in between.
std::set<int> open_socket_fds()
{
    std::set<int> fds;
    for (const auto& entry : std::filesystem::directory_iterator("/proc/self/fd")) {
        std::error_code ec;
        const auto target = std::filesystem::read_symlink(entry.path(), ec);
        if (!ec && target.string().rfind("socket:", 0) == 0) {
            fds.insert(std::stoi(entry.path().filename().string()));
        }
    }
    return fds;
}

bool fd_is_open(int fd)
{
    return ::fcntl(fd, F_GETFD) != -1;
}

bool wait_until_closed(int fd, std::chrono::milliseconds timeout)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (fd_is_open(fd)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return true;
}

// Restarts the listener on TEST_PORT and returns its fd, or -1 if the
// listener's socket could not be identified.
int restart_listener(Adapter& adapter)
{
    adapter.stop_bootp_listener();
    const std::set<int> before = open_socket_fds();
    adapter.start_bootp_listener(TEST_PORT);
    std::set<int> added;
    for (int fd : open_socket_fds()) {
        if (before.count(fd) == 0) {
            added.insert(fd);
        }
    }
    return added.size() == 1 ? *added.begin() : -1;
}

void send_to_listener(const std::vector<std::uint8_t>& packet)
{
    int sender = ::socket(AF_INET, SOCK_DGRAM, 0);
    ASSERT_GE(sender, 0);
    sockaddr_in dest {};
    dest.sin_family = AF_INET;
    dest.sin_port = htons(static_cast<std::uint16_t>(TEST_PORT));
    dest.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    EXPECT_EQ(::sendto(sender, packet.data(), packet.size(), 0,
                  reinterpret_cast<sockaddr*>(&dest), sizeof(dest)),
        static_cast<ssize_t>(packet.size()));
    ::close(sender);
}

std::vector<std::uint8_t> test_packet()
{
    return build_bootp_v2(
        STUB_UUID_BYTES, TEST_SERIAL_WIRE_BYTES, TEST_COMPAT_ID, TEST_DATA_PLANE);
}

} // namespace

// Stopping the listener while an announcement is being handled must not
// close the socket under the handler; the socket closes once the handler
// returns.
TEST(HololinkAdapterBootpEndToEnd, StopDuringDispatchKeepsSocketOpen)
{
    auto& adapter = Adapter::get_adapter();
    const int listener_fd = restart_listener(adapter);
    ASSERT_GE(listener_fd, 0);

    struct State {
        std::once_flag once;
        std::promise<void> entered;
        std::promise<void> release;
        std::once_flag released;
    };
    auto state = std::make_shared<State>();
    auto entered = state->entered.get_future();
    auto release = state->release.get_future().share();

    // Unblocks the handler and unregisters on every exit path, so a failed
    // assertion cannot leave the reactor thread blocked.
    struct Cleanup {
        Adapter& adapter;
        std::shared_ptr<State> state;
        EnumerationCallbackHandle handle;
        void release()
        {
            std::call_once(state->released, [&]() { state->release.set_value(); });
        }
        ~Cleanup()
        {
            release();
            adapter.unregister(handle);
        }
    } cleanup { adapter, state,
        adapter.register_raw_all([state, release](const EnumerationMetadata&) {
            std::call_once(state->once, [&]() {
                state->entered.set_value();
                release.wait();
            });
        }) };

    send_to_listener(test_packet());
    ASSERT_EQ(entered.wait_for(std::chrono::seconds(5)), std::future_status::ready);

    // A call to the bootp callback is running on the reactor thread now.
    adapter.stop_bootp_listener();
    EXPECT_TRUE(fd_is_open(listener_fd));

    cleanup.release();
    EXPECT_TRUE(wait_until_closed(listener_fd, std::chrono::seconds(5)));
}

// An enumeration subscriber, which runs inside the bootp callback on the
// reactor thread, may stop the listener.
TEST(HololinkAdapterBootpEndToEnd, StopFromSubscriberClosesSocket)
{
    auto& adapter = Adapter::get_adapter();
    const int listener_fd = restart_listener(adapter);
    ASSERT_GE(listener_fd, 0);

    auto once = std::make_shared<std::once_flag>();
    auto stopped = std::make_shared<std::promise<bool>>();
    auto open_after_stop = stopped->get_future();

    struct Cleanup {
        Adapter& adapter;
        EnumerationCallbackHandle handle;
        ~Cleanup() { adapter.unregister(handle); }
    } cleanup { adapter,
        adapter.register_raw_all([&adapter, once, stopped, listener_fd](const EnumerationMetadata&) {
            std::call_once(*once, [&]() {
                adapter.stop_bootp_listener();
                stopped->set_value(fd_is_open(listener_fd));
            });
        }) };

    send_to_listener(test_packet());
    ASSERT_EQ(open_after_stop.wait_for(std::chrono::seconds(5)), std::future_status::ready);

    // Still being handled when stop returned, so still open then...
    EXPECT_TRUE(open_after_stop.get());
    // ...and closed once the handler is done.
    EXPECT_TRUE(wait_until_closed(listener_fd, std::chrono::seconds(5)));
}

// The Ethernet HSB firmware sends a 56-byte vendor area whose tail (offsets 41..57) carries the
// board's PTP time, a packet sequence number and a status byte (ip_integration.mdx, Table 2).
// The parser is transport-agnostic and stops at fpga_crc: a full-length record yields none of
// the tail keys (they belong to the module that claims the UUID -- hsb_pcie decodes them from
// the raw packet for its endpoints) and the header path is the same for every htype: peer_ip
// is the IPv4 source and mac_id the colon-hex hardware address.
TEST(HololinkAdapterBootpEndToEnd, LeavesTheVendorTailToTheModule)
{
    using hololink::module::Adapter;
    using hololink::module::EnumerationMetadata;

    auto& adapter = Adapter::get_adapter();
    adapter.stop_bootp_listener();
    adapter.start_bootp_listener(TEST_PORT);

    struct Capture {
        std::mutex mutex;
        std::condition_variable cv;
        std::optional<EnumerationMetadata> metadata;
    };
    auto capture = std::make_shared<Capture>();
    // Told apart from the previous test's announcement by its data plane.
    constexpr std::uint8_t TAIL_TEST_DATA_PLANE = TEST_DATA_PLANE + 1;
    auto handle = adapter.register_raw_all([capture](const EnumerationMetadata& md) {
        if (md.get<std::string>("fpga_uuid", "") != STUB_UUID_STRING
            || md.get<int64_t>("data_plane", -1) != TAIL_TEST_DATA_PLANE) {
            return;
        }
        std::lock_guard<std::mutex> guard(capture->mutex);
        if (!capture->metadata) {
            capture->metadata = md;
            capture->cv.notify_all();
        }
    });

    std::vector<std::uint8_t> packet = build_bootp_v2(
        STUB_UUID_BYTES, TEST_SERIAL_WIRE_BYTES, TEST_COMPAT_ID, TAIL_TEST_DATA_PLANE);
    ASSERT_EQ(packet.size(), 277u);
    packet[237] = 56; // the firmware's tag length
    put_u32_be(packet, 0); // vendor 41..44: eeprom_misc, tied to 0
    // vendor 45..50: PTP seconds BE48 = 0x0000_6543_21FE
    for (std::uint8_t byte : { 0x00, 0x00, 0x65, 0x43, 0x21, 0xFE }) {
        put_u8(packet, byte);
    }
    put_u32_be(packet, 123456789u); // vendor 51..54: PTP nanoseconds
    put_u16_be(packet, 0x1234); // vendor 55..56: PSN
    put_u8(packet, 0x01); // vendor 57: PTP enabled and a SYNC received
    put_zeros(packet, 6); // vendor 58..63
    ASSERT_EQ(packet.size(), 300u);

    send_to_listener(packet);

    {
        std::unique_lock<std::mutex> lock(capture->mutex);
        ASSERT_TRUE(capture->cv.wait_for(lock, std::chrono::seconds(3), [&] { return capture->metadata.has_value(); }));
    }
    adapter.unregister(handle);
    adapter.stop_bootp_listener();
    const EnumerationMetadata& md = *capture->metadata;

    EXPECT_EQ(md.get<int64_t>("hardware_type"), 1);
    EXPECT_EQ(md.get<std::string>("peer_ip"), "127.0.0.1") << "the IPv4 source";
    EXPECT_EQ(md.get<std::string>("mac_id"), "02:00:00:00:00:01");
    EXPECT_EQ(md.get<int64_t>("data_plane"), TAIL_TEST_DATA_PLANE);
    EXPECT_EQ(md.get<std::string>("serial_number"), TEST_SERIAL_DECODED);
    EXPECT_FALSE(md.contains("hardware_address_text"));
    EXPECT_FALSE(md.contains("ptp_timestamp_s"));
    EXPECT_FALSE(md.contains("ptp_timestamp_ns"));
    EXPECT_FALSE(md.contains("bootp_psn"));
    EXPECT_FALSE(md.contains("bootp_status"));
}
