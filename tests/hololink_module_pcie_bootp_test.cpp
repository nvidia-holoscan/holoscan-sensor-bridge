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

/*
 * Pins the hololink_pcie kernel driver's bootp announcement to the host parser, without a
 * kernel: the 300-byte payload is built with the very inline filler the driver uses
 * (hololink_pcie_bootp.h, mirrored byte-identical into the installed module headers), sent to
 * the Adapter's bootp listener over loopback as the driver's netif_rx() frame arrives (a UDP
 * broadcast to 127.255.255.255:<port>, which needs SO_BROADCAST from userspace and reaches the
 * listener because it binds INADDR_ANY; with one difference: the driver sources the datagram
 * from the BDF packed as an IPv4 address, the ciaddr value, which a userspace socket cannot --
 * 0.0.0.0/8 is refused as a source -- so this sender uses 127.0.0.1, which the module accepts
 * equally), and checked twice:
 *
 *  1. raw (pre-module): every field the transport-agnostic parser derives -- hardware_type 0x50,
 *     the 4-byte chaddr in hardware_address and as colon-hex in mac_id, peer_ip = the datagram's
 *     IPv4 source, client_ip_address = the BDF as a dotted quad, transaction_id, fpga_uuid,
 *     hsb_ip_version, data_plane, enum_version, compat_id, packet count, serial (placeholder or
 *     DSN) -- and nothing PCIe-specific: no BDF text, no vendor tail;
 *  2. through the real hsb_pcie module .so: its update_metadata renders the BDF from chaddr into
 *     peer_ip / mac_id / hardware_address_text and decodes the vendor tail (ptp_timestamp_s/ns,
 *     bootp_psn, bootp_status) before Adapter::wait_for_channel("0005:01:00.0") resolves; the
 *     module stamps the pcie_* fields, keeps a DSN serial and replaces the placeholder.
 */

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <hololink/module/adapter.hpp>
#include <hololink/module/enumeration_metadata.hpp>
#include <hololink/module/pcie/hololink_pcie_bootp.h>
#include <hololink/module/pcie/pcie_sysfs.hpp>

namespace {

using hololink::module::Adapter;
using hololink::module::EnumerationMetadata;

constexpr std::uint32_t TEST_PORT = 18277; // not 12267: real boards on the bench must not reach us
constexpr const char* BDF = "0005:01:00.0";
constexpr const char* BDF_DOTTED = "0.5.1.0";
// The hsb_pcie module's UUID (the board's read-only UUID), in the driver's byte order.
constexpr std::uint8_t BOARD_UUID[16] = {
    0x58, 0xc1, 0x41, 0x9b, 0x1e, 0x7a, 0x41, 0x20, 0x95, 0xfd, 0x4e, 0xfc, 0x15, 0x61, 0x87, 0xc1
};
constexpr const char* BOARD_UUID_TEXT = "58c1419b-1e7a-4120-95fd-4efc156187c1";

std::filesystem::path test_module_dir()
{
    const char* dir = std::getenv("HOLOLINK_MODULE_TEST_MODULE_DIR");
    if (!dir) {
        throw std::runtime_error("HOLOLINK_MODULE_TEST_MODULE_DIR is not set");
    }
    return std::filesystem::path(dir);
}

// A Device Serial Number as pci_get_dsn() returns it (EUI-64, MSB first); the announcement
// carries its 7 least-significant bytes LSB first, so the serial reads 08 07 06 05 04 03 02.
constexpr std::uint64_t TEST_DSN = 0x0102030405060708ULL;
constexpr const char* TEST_DSN_SERIAL = "08070605040302";
constexpr std::uint64_t TEST_PTP_SEC = 0x0123456789ABULL; // 48 bits
constexpr std::uint32_t TEST_PTP_NSEC = 999999999u; // 0x3B9AC9FF

hl_pcie_bootp_id board_id(std::uint32_t pkt_count, std::uint16_t secs, std::uint64_t dsn = 0,
    bool ptp_valid = false)
{
    hl_pcie_bootp_id id {};
    id.domain = 0x0005;
    id.bus = 0x01;
    id.devfn = 0x00;
    std::memcpy(id.uuid, BOARD_UUID, sizeof(BOARD_UUID));
    id.hsb_ip_version = 0x2606;
    id.compat_id = 0;
    id.fpga_crc = 0;
    id.data_plane = 0;
    id.pkt_count = pkt_count;
    id.secs = secs;
    id.dsn = dsn;
    id.ptp_valid = ptp_valid ? 1 : 0;
    if (ptp_valid) {
        id.ptp_sec = TEST_PTP_SEC;
        id.ptp_nsec = TEST_PTP_NSEC;
    }
    return id;
}

/* Send the driver's payload to the listener the way the driver's frame arrives: broadcast on the
 * loopback subnet (HL_PCIE_BOOTP_DADDR, 127.255.255.255, which needs SO_BROADCAST from userspace
 * and reaches the listener because it binds INADDR_ANY), source port 12268 when it is free, to
 * TEST_PORT. The one thing userspace cannot reproduce is the driver's IP source (the packed BDF,
 * in 0.0.0.0/8); this sends from 127.0.0.1, which the module treats the same way since it takes
 * the BDF from ciaddr. */
void send_announcement(const hl_pcie_bootp_pkt& pkt)
{
    int sender = ::socket(AF_INET, SOCK_DGRAM, 0);
    ASSERT_GE(sender, 0);
    const int one = 1;
    ASSERT_EQ(::setsockopt(sender, SOL_SOCKET, SO_BROADCAST, &one, sizeof(one)), 0);
    sockaddr_in src {};
    src.sin_family = AF_INET;
    src.sin_port = htons(HL_PCIE_BOOTP_SPORT);
    src.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    (void)::bind(sender, reinterpret_cast<sockaddr*>(&src), sizeof(src)); // best effort
    sockaddr_in dest {};
    dest.sin_family = AF_INET;
    dest.sin_port = htons(static_cast<std::uint16_t>(TEST_PORT));
    dest.sin_addr.s_addr = htonl(HL_PCIE_BOOTP_DADDR);
    const ssize_t sent = ::sendto(sender, &pkt, sizeof(pkt), 0,
        reinterpret_cast<sockaddr*>(&dest), sizeof(dest));
    EXPECT_EQ(sent, static_cast<ssize_t>(sizeof(pkt)));
    ::close(sender);
}

class KernelBootp : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        Adapter& adapter = Adapter::get_adapter();
        adapter.stop_bootp_listener();
        adapter.start_bootp_listener(TEST_PORT);
        adapter.set_module_directory(test_module_dir());
    }
    static void TearDownTestSuite() { Adapter::get_adapter().stop_bootp_listener(); }
};

} // namespace

TEST(KernelBootpLayout, PayloadIsThreeHundredBytesInWireOrder)
{
    static_assert(sizeof(hl_pcie_bootp_pkt) == HL_PCIE_BOOTP_LEN, "RFC 951 payload size");
    hl_pcie_bootp_pkt pkt;
    const hl_pcie_bootp_id id = board_id(/*pkt_count=*/0x010203, /*secs=*/7);
    hl_pcie_bootp_fill(&pkt, &id);
    const auto* b = reinterpret_cast<const std::uint8_t*>(&pkt);

    EXPECT_EQ(b[0], 1u); // op
    EXPECT_EQ(b[1], 0x50u); // htype: a PCIe HSB / eHSB endpoint ('P'), not Ethernet's 1
    EXPECT_EQ(b[1], HL_PCIE_BOOTP_HTYPE_PCIE);
    EXPECT_EQ(b[2], 4u); // hlen: chaddr is the BDF
    EXPECT_EQ(b[2], HL_PCIE_BOOTP_HLEN);
    EXPECT_EQ(b[3], 0u); // hops
    EXPECT_EQ((b[4] << 24 | b[5] << 16 | b[6] << 8 | b[7]), 0x00050100u) << "xid = domain<<16 | requester_id";
    EXPECT_EQ((b[8] << 8 | b[9]), 7u); // secs BE
    EXPECT_EQ((b[12] << 24 | b[13] << 16 | b[14] << 8 | b[15]), 0x00050100u) << "ciaddr = bdf32";
    EXPECT_EQ(b[28], 0x00u); // chaddr: domain hi
    EXPECT_EQ(b[29], 0x05u); // domain lo
    EXPECT_EQ(b[30], 0x01u); // bus
    EXPECT_EQ(b[31], 0x00u); // devfn
    for (size_t i = 32; i < 236; i++) {
        ASSERT_EQ(b[i], 0u) << "chaddr tail, sname and file are zero at " << i;
    }
    EXPECT_EQ(b[236], 0xE0u); // vendor tag
    EXPECT_EQ(b[237], 56u) << "the Ethernet HSB IP's tag length: the tail through the status byte is present";
    EXPECT_EQ(b[237], HL_PCIE_BOOTP_VEND_TAG_LEN);
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(b + 238), 4), "NVDA");
    EXPECT_EQ(b[242], 0u); // data_plane
    EXPECT_EQ(b[243], 2u); // enum_version
    EXPECT_EQ((b[244] | b[245] << 8), 0u); // compat_id LE
    EXPECT_EQ(std::memcmp(b + 246, BOARD_UUID, 16), 0);
    EXPECT_EQ((b[262] << 16 | b[263] << 8 | b[264]), 0x010203u) << "packet count BE24";
    EXPECT_EQ(b[265], 0u); // pad
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(b + 266), 4), "PCIE");
    EXPECT_EQ(b[270], 0x05u);
    EXPECT_EQ(b[271], 0x01u);
    EXPECT_EQ(b[272], 0x00u);
    EXPECT_EQ((b[273] | b[274] << 8), 0x2606u) << "hsb_ip_version LE";
    EXPECT_EQ((b[275] | b[276] << 8), 0u); // fpga_crc LE
    // No DSN, no PTP: the tail carries only the PSN; the status byte says neither is present.
    for (size_t i = 277; i < 291; i++) {
        ASSERT_EQ(b[i], 0u) << "reserved and PTP bytes are zero at " << i;
    }
    EXPECT_EQ((b[291] << 8 | b[292]), 0x0203u) << "psn BE16 = pkt_count & 0xFFFF";
    EXPECT_EQ(b[293], 0u) << "status: no PTP snapshot, placeholder serial";
    for (size_t i = 294; i < 300; i++) {
        ASSERT_EQ(b[i], 0u) << "vend tail is zero at " << i;
    }

    // The two helpers the module uses to get back from the parser's rendering to the BDF.
    std::uint32_t bdf32 = 0;
    ASSERT_TRUE(hololink::module::pcie::dotted_quad_to_u32(BDF_DOTTED, bdf32));
    EXPECT_EQ(bdf32, 0x00050100u);
    EXPECT_EQ(hololink::module::pcie::bdf32_to_string(bdf32), BDF);
    EXPECT_EQ(hololink::module::pcie::bdf32_to_string(0x00008a0fu), "0000:8a:01.7");
    EXPECT_FALSE(hololink::module::pcie::dotted_quad_to_u32("0.5.1", bdf32));
    EXPECT_FALSE(hololink::module::pcie::dotted_quad_to_u32("0.5.1.256", bdf32));
    EXPECT_FALSE(hololink::module::pcie::dotted_quad_to_u32("not-an-address", bdf32));
}

// With a Device Serial Number and a PTP snapshot: the serial bytes are the DSN's low seven,
// LSB first (the Ethernet firmware's little-endian serial order), the timestamp lands at vendor
// 45..54 big-endian, and the status byte reports both.
TEST(KernelBootpLayout, DsnSerialAndPtpSnapshotLandInTheVendorTail)
{
    hl_pcie_bootp_pkt pkt;
    const hl_pcie_bootp_id id = board_id(/*pkt_count=*/0x010203, /*secs=*/7, TEST_DSN, /*ptp_valid=*/true);
    hl_pcie_bootp_fill(&pkt, &id);
    const auto* b = reinterpret_cast<const std::uint8_t*>(&pkt);

    const std::uint8_t serial[7] = { 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02 };
    EXPECT_EQ(std::memcmp(b + 266, serial, sizeof(serial)), 0) << "DSN low 7 bytes, LSB first";
    const std::uint8_t ptp_sec[6] = { 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB };
    EXPECT_EQ(std::memcmp(b + 281, ptp_sec, sizeof(ptp_sec)), 0) << "PTP seconds BE48 at vendor 45";
    const std::uint8_t ptp_nsec[4] = { 0x3B, 0x9A, 0xC9, 0xFF };
    EXPECT_EQ(std::memcmp(b + 287, ptp_nsec, sizeof(ptp_nsec)), 0) << "PTP nanoseconds BE32 at vendor 51";
    EXPECT_EQ((b[291] << 8 | b[292]), 0x0203u) << "psn";
    EXPECT_EQ(b[293], HL_PCIE_BOOTP_STATUS_PTP_VALID | HL_PCIE_BOOTP_STATUS_SERIAL_DSN);
    for (size_t i = 277; i < 281; i++) {
        ASSERT_EQ(b[i], 0u) << "reserved bytes at " << i;
    }
    for (size_t i = 294; i < 300; i++) {
        ASSERT_EQ(b[i], 0u) << "vend tail is zero at " << i;
    }

    // An unprogrammed DSN (all-ones) is "no DSN" as much as a missing capability (0).
    hl_pcie_bootp_fill(&pkt, &(const hl_pcie_bootp_id&)board_id(1, 0, ~0ULL, false));
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(b + 266), 4), "PCIE");
    EXPECT_EQ(b[293], 0u);
    EXPECT_FALSE(hl_pcie_bootp_dsn_valid(0));
    EXPECT_FALSE(hl_pcie_bootp_dsn_valid(~0ULL));
    EXPECT_TRUE(hl_pcie_bootp_dsn_valid(TEST_DSN));
}

TEST_F(KernelBootp, HostParserDeliversTheDriversAnnouncementTransportAgnostically)
{
    struct Capture {
        std::mutex mutex;
        std::condition_variable cv;
        std::optional<EnumerationMetadata> metadata;
    };
    auto capture = std::make_shared<Capture>();
    // Raw subscribers see the parser's output before any module touches it.
    auto handle = Adapter::get_adapter().register_raw_all([capture](const EnumerationMetadata& md) {
        if (md.get<std::string>("fpga_uuid", "") != BOARD_UUID_TEXT) {
            return;
        }
        std::lock_guard<std::mutex> guard(capture->mutex);
        if (!capture->metadata) {
            capture->metadata = md;
            capture->cv.notify_all();
        }
    });

    hl_pcie_bootp_pkt pkt;
    const hl_pcie_bootp_id id = board_id(/*pkt_count=*/42, /*secs=*/3);
    hl_pcie_bootp_fill(&pkt, &id);
    send_announcement(pkt);

    {
        std::unique_lock<std::mutex> lock(capture->mutex);
        ASSERT_TRUE(capture->cv.wait_for(lock, std::chrono::seconds(3), [&] { return capture->metadata.has_value(); }));
    }
    Adapter::get_adapter().unregister(handle);
    const EnumerationMetadata& md = *capture->metadata;

    // The parser knows nothing about htype 0x50: it reports the header as it is -- peer_ip is
    // the datagram's IPv4 source (this test's 127.0.0.1; the driver's 0.5.1.0, the ciaddr value),
    // mac_id the chaddr bytes in the Ethernet colon-hex convention -- and leaves the BDF
    // rendering to the module that claims the UUID (the next tests).
    EXPECT_EQ(md.get<int64_t>("hardware_type"), 0x50);
    EXPECT_EQ(md.get<std::string>("peer_ip"), "127.0.0.1") << "the IPv4 source, as for any htype";
    EXPECT_FALSE(md.contains("source_ip"));
    EXPECT_EQ(md.get<std::string>("client_ip_address"), BDF_DOTTED) << "ciaddr carries the BDF";
    EXPECT_EQ(md.get<std::string>("mac_id"), "00:05:01:00") << "chaddr, colon-hex";
    EXPECT_FALSE(md.contains("hardware_address_text")) << "the module's rendering; not the parser's";
    EXPECT_EQ(md.get<int64_t>("hardware_address_length"), 4);
    const std::vector<uint8_t> chaddr { 0x00, 0x05, 0x01, 0x00 };
    EXPECT_EQ(md.get<std::vector<uint8_t>>("hardware_address"), chaddr) << "trimmed to hlen";
    EXPECT_EQ(md.get<int64_t>("transaction_id"), 0x00050100);
    EXPECT_EQ(md.get<int64_t>("seconds"), 3);
    EXPECT_EQ(md.get<std::string>("fpga_uuid"), BOARD_UUID_TEXT);
    EXPECT_EQ(md.get<int64_t>("hsb_ip_version"), 0x2606);
    EXPECT_EQ(md.get<int64_t>("data_plane"), 0);
    EXPECT_EQ(md.get<int64_t>("enum_version"), 2);
    EXPECT_EQ(md.get<int64_t>("compat_id"), 0);
    EXPECT_EQ(md.get<int64_t>("transmitted_packet_count"), 42);
    EXPECT_EQ(md.get<std::string>("serial_number"), "50434945050100") << "the placeholder, 7 raw bytes as hex";
    EXPECT_EQ(md.get<std::string>("interface"), "lo");
    EXPECT_EQ(md.get<int64_t>("control_port"), 8192) << "parser constant; the PCIe module ignores it";
    // The parser stops at fpga_crc: the vendor tail is the module's to decode.
    EXPECT_FALSE(md.contains("ptp_timestamp_s"));
    EXPECT_FALSE(md.contains("ptp_timestamp_ns"));
    EXPECT_FALSE(md.contains("bootp_psn"));
    EXPECT_FALSE(md.contains("bootp_status"));
}

namespace {

/* Announce `pkt` at the driver's cadence until Adapter::wait_for_channel(BDF) resolves with
 * THIS announcement -- matched on its sequence number (vendor 55..56), since a late duplicate
 * from an earlier test can still be queued in the listener when the wait starts. */
EnumerationMetadata wait_for_bdf(const hl_pcie_bootp_pkt& pkt)
{
    const int64_t psn = (int64_t(pkt.vend[HL_PCIE_BOOTP_VEND_PSN]) << 8) | pkt.vend[HL_PCIE_BOOTP_VEND_PSN + 1];
    struct Result {
        std::mutex mutex;
        std::optional<EnumerationMetadata> found;
        std::string error;
    };
    auto result = std::make_shared<Result>();
    std::thread waiter([result, psn] {
        try {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (std::chrono::steady_clock::now() < deadline) {
                EnumerationMetadata md = Adapter::get_adapter().wait_for_channel(BDF, std::chrono::seconds(5));
                if (md.get<int64_t>("bootp_psn", -1) != psn) {
                    continue; // somebody else's announcement
                }
                std::lock_guard<std::mutex> guard(result->mutex);
                result->found = md;
                return;
            }
            std::lock_guard<std::mutex> guard(result->mutex);
            result->error = "no announcement with psn " + std::to_string(psn) + " within 5 s";
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> guard(result->mutex);
            result->error = e.what();
        }
    });
    // wait_for_channel only counts announcements after it started waiting; give it a moment,
    // then announce at the driver's cadence until it resolves.
    for (int i = 0; i < 6; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        send_announcement(pkt);
        std::lock_guard<std::mutex> guard(result->mutex);
        if (result->found || !result->error.empty()) {
            break;
        }
    }
    waiter.join();
    if (!result->found) {
        throw std::runtime_error(result->error.empty() ? "wait_for_channel did not resolve" : result->error);
    }
    return *result->found;
}

} // namespace

TEST_F(KernelBootp, WaitForChannelResolvesTheBdfThroughTheModule)
{
    // The module .so (from HOLOLINK_MODULE_TEST_MODULE_DIR) renders the htype-0x50 chaddr as the
    // BDF into peer_ip in its update_metadata, before register_ip / wait_for_channel subscribers
    // are matched -- so the wait keyed on the BDF resolves -- decodes the vendor tail, stamps the
    // pcie_* fields and replaces the placeholder serial.
    hl_pcie_bootp_pkt pkt;
    const hl_pcie_bootp_id id = board_id(/*pkt_count=*/1, /*secs=*/0);
    hl_pcie_bootp_fill(&pkt, &id);
    EnumerationMetadata md;
    ASSERT_NO_THROW(md = wait_for_bdf(pkt));

    EXPECT_EQ(md.get<std::string>("peer_ip"), BDF) << "from chaddr, whatever the IP source";
    EXPECT_EQ(md.get<std::string>("pcie_bdf"), BDF);
    EXPECT_FALSE(md.contains("pcie_source_ip")) << "only an older driver's htype-1 record carries a separate source";
    EXPECT_EQ(md.get<std::string>("serial_number"), std::string("pcie-") + BDF) << "placeholder replaced";
    EXPECT_EQ(md.get<std::string>("pcie_serial_source"), "bdf");
    EXPECT_EQ(md.get<std::string>("transport"), "pcie");
    EXPECT_EQ(md.get<std::string>("module_name"), "hsb_pcie");
    EXPECT_EQ(md.get<std::string>("mac_id"), BDF) << "the hardware address in its htype's convention";
    EXPECT_EQ(md.get<std::string>("hardware_address_text"), BDF);
    EXPECT_EQ(md.get<int64_t>("pcie_domain"), 5);
    EXPECT_EQ(md.get<int64_t>("pcie_bus"), 1);
    EXPECT_EQ(md.get<int64_t>("pcie_device"), 0);
    EXPECT_EQ(md.get<int64_t>("pcie_function"), 0);
    // The vendor tail (the Ethernet firmware's positions): no PTP, placeholder serial, psn.
    EXPECT_EQ(md.get<int64_t>("ptp_timestamp_s"), 0);
    EXPECT_EQ(md.get<int64_t>("ptp_timestamp_ns"), 0);
    EXPECT_EQ(md.get<int64_t>("bootp_psn"), 1);
    EXPECT_EQ(md.get<int64_t>("bootp_status"), 0) << "no PTP snapshot, placeholder serial";
}

TEST_F(KernelBootp, ModuleKeepsTheDeviceSerialNumber)
{
    // The same announcement from an endpoint with a Device Serial Number and bootp_ptp=1: the
    // module decodes the tail, and the serial rule keys on its status byte.
    hl_pcie_bootp_pkt pkt;
    const hl_pcie_bootp_id id = board_id(/*pkt_count=*/0x1BEEF, /*secs=*/0, TEST_DSN, /*ptp_valid=*/true);
    hl_pcie_bootp_fill(&pkt, &id);
    EnumerationMetadata md;
    ASSERT_NO_THROW(md = wait_for_bdf(pkt));
    EXPECT_EQ(md.get<std::string>("peer_ip"), BDF);
    EXPECT_EQ(md.get<std::string>("serial_number"), TEST_DSN_SERIAL) << "the DSN's low 7 bytes, LSB first, as hex; kept";
    EXPECT_EQ(md.get<std::string>("pcie_serial_source"), "dsn");
    EXPECT_EQ(md.get<int64_t>("ptp_timestamp_s"), static_cast<int64_t>(TEST_PTP_SEC));
    EXPECT_EQ(md.get<int64_t>("ptp_timestamp_ns"), static_cast<int64_t>(TEST_PTP_NSEC));
    EXPECT_EQ(md.get<int64_t>("bootp_psn"), 0xBEEF) << "pkt_count & 0xFFFF";
    EXPECT_EQ(md.get<int64_t>("bootp_status"), HL_PCIE_BOOTP_STATUS_PTP_VALID | HL_PCIE_BOOTP_STATUS_SERIAL_DSN)
        << "the raw status byte is the only flag published: PTP snapshot live, serial is the DSN";
    EXPECT_EQ(md.get<int64_t>("transmitted_packet_count"), 0x1BEEF) << "the full 24-bit count is still there";
    EXPECT_EQ(md.get<std::string>("transport"), "pcie");
}
