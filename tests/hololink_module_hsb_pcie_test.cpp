/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Drives the hsb_pcie module (hololink_module/module/hsb_pcie/) end-to-end through the
 * framework: a synthetic PCIe-shaped EnumerationMetadata (peer_ip = the BDF "0000:00:00.0",
 * mac_id = the all-zero MAC, pcie_minor = 0, pcie_backend = loopback) goes through
 * Adapter::enumerate, which dlopens hololink_58c1419b-1e7a-4120-95fd-4efc156187c1.so and runs
 * its update_metadata. Then the per-board HololinkInterfaceV1 is resolved, started over the
 * loopback endpoint and read; the I2C child resolves; the RoCE data channel is declined; the
 * PCIe data channel (PcieDataChannelInterfaceV1) programs the endpoint's PCIE_DP block, MSI-X
 * entry and DP_PKT words for a fake receiver attachment and arms / disarms only its own
 * VIP_MASK bit -- all read back through the Hololink; a bootp-shaped announcement carrying
 * this module's UUID is skipped, as is a /dev-style peer that does not say transport = pcie;
 * and a /dev-style peer that does is accepted.
 */

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "hololink/module/adapter.hpp"
#include "hololink/module/data_channel.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/hsb_pcie/pcie_data_channel.hpp"
#include "hololink/module/i2c.hpp"
#include "hololink/module/roce_data_channel.hpp"
#include "pcie_regs.hpp"

namespace {

std::filesystem::path test_module_dir()
{
    const char* dir = std::getenv("HOLOLINK_MODULE_TEST_MODULE_DIR");
    return dir ? std::filesystem::path(dir) : std::filesystem::current_path();
}

// The fixed UUID every hsb_pcie board is provisioned with (plan decision 3).
constexpr const char* HSB_PCIE_UUID = "58c1419b-1e7a-4120-95fd-4efc156187c1";
// The channel key: the endpoint's PCI function address. The loopback backend takes the
// minor from pcie_minor instead of sysfs, so this BDF need not exist on the test host.
constexpr const char* PEER = "0000:00:00.0";
constexpr const char* NULL_MAC = "00:00:00:00:00:00";
constexpr const char* DEVICE_NODE_PEER = "/dev/hololink_pcie0";

hololink::module::EnumerationMetadata pcie_metadata(const std::string& serial,
    const std::string& peer = PEER)
{
    hololink::module::EnumerationMetadata md;
    md["fpga_uuid"] = std::string(HSB_PCIE_UUID);
    md["peer_ip"] = peer;
    md["mac_id"] = std::string(NULL_MAC);
    md["pcie_bdf"] = std::string(PEER);
    md["pcie_minor"] = static_cast<int64_t>(0);
    md["serial_number"] = serial;
    md["control_port"] = static_cast<int64_t>(0);
    md["data_plane"] = static_cast<int64_t>(0);
    md["hsb_ip_version"] = static_cast<int64_t>(0x2606);
    md["pcie_backend"] = std::string("loopback");
    return md;
}

/* Same guard as hololink_module_hsb_lite_test: the Adapter opens a live bootp listener on
 * the default port, and real boards on the CI network would race the synthetic
 * announcements. Rebind it to an ephemeral port so nothing off the wire reaches this
 * process. The PCIe peer key is a PCI function address, which no bootp packet can carry. */
class HololinkAdapterHsbPcie : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        auto& adapter = hololink::module::Adapter::get_adapter();
        adapter.stop_bootp_listener();
        adapter.start_bootp_listener(0);
    }

    void SetUp() override
    {
        hololink::module::Adapter::get_adapter().set_module_directory(test_module_dir());
    }
};

/* What a PcieReceiver would hand the channel, as constants: a 3-slot ring of 0x278F80-byte
 * windows at 0x2800e00000 (so BUF_BASE_HI = 0x28, BUF_START = 0x00e00000), requester id
 * 0x0600, and the MSI-X message the driver would report. `vector` is per instance so one
 * fake attaches to sensor 0 and another to sensor 1. */
class FakeAttachment : public hololink::module::hsb_pcie::PcieReceiverAttachmentV1 {
public:
    static constexpr uint64_t IOVA = 0x2800e00000ull;
    static constexpr uint64_t WINDOW = 0x278F80ull;
    // align_up(0x278F80, 128) + 128 = 0x279000, padded to 4 KiB = 0x279000; in 128-byte pages.
    static constexpr uint32_t BUF_INC_PAGES = 0x279000u / 128u;
    static constexpr uint32_t NUM_BUFFERS = 3;
    static constexpr uint32_t REQUESTER_ID = 0x0600;

    explicit FakeAttachment(uint32_t vector)
        : vector_(vector)
    {
    }

    uint64_t iova() const override { return IOVA; }
    uint64_t window_bytes() const override { return WINDOW; }
    uint32_t buf_inc_pages() const override { return BUF_INC_PAGES; }
    uint32_t num_buffers() const override { return NUM_BUFFERS; }
    uint32_t requester_id() const override { return REQUESTER_ID; }
    uint32_t vector() const override { return vector_; }
    bool disable_metadata() const override { return false; }

private:
    uint32_t vector_;
};

/// One BAR0 word read back through the Hololink.
uint32_t read_word(hololink::module::HololinkInterfaceV1& hololink, uint32_t address)
{
    std::vector<uint32_t> values;
    if (hololink.read_uint32({ address }, values) != HOLOLINK_MODULE_OK || values.size() != 1) {
        throw std::runtime_error("read_uint32 failed");
    }
    return values[0];
}

/// enumerate() on a helper thread while this thread waits for `peer`.
hololink::module::EnumerationMetadata enumerate_and_wait(hololink::module::EnumerationMetadata metadata,
    const std::string& peer)
{
    std::thread enum_thread([metadata]() mutable {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        hololink::module::Adapter::get_adapter().enumerate(std::move(metadata));
    });
    // wait_for_channel throws on timeout; join on that path too, or the joinable thread
    // terminates the test binary while unwinding (the helper thread is bounded).
    struct JoinOnExit {
        std::thread& thread;
        ~JoinOnExit()
        {
            if (thread.joinable()) {
                thread.join();
            }
        }
    } join_enum { enum_thread };
    return hololink::module::Adapter::get_adapter().wait_for_channel(peer, std::chrono::seconds(2));
}

} // namespace

TEST_F(HololinkAdapterHsbPcie, EnumerateLoadsModuleAndEnriches)
{
    using hololink::module::Adapter;
    using hololink::module::EnumerationMetadata;

    EnumerationMetadata found = enumerate_and_wait(pcie_metadata("pcie-0000:00:00.0"), PEER);

    ASSERT_TRUE(found.contains("module_name"));
    EXPECT_EQ(found.get<std::string>("module_name"), "hsb_pcie");
    ASSERT_TRUE(found.contains("module_filename"));
    const std::string filename = found.get<std::string>("module_filename");
    EXPECT_NE(filename.find(HSB_PCIE_UUID), std::string::npos) << filename;
    EXPECT_NE(filename.find(std::string("hololink_") + HSB_PCIE_UUID + ".so"), std::string::npos) << filename;

    // Default sensor = data_plane 0: PCIE_DP block 0, MSI-X vector 0.
    EXPECT_EQ(found.get<int64_t>("sensor_number"), 0);
    EXPECT_EQ(found.get<int64_t>("sensor"), 0);
    EXPECT_EQ(found.get<int64_t>("vp_address"), 0x1000);
    EXPECT_EQ(found.get<int64_t>("sif_address"), 0x01000000);
    EXPECT_EQ(found.get<int64_t>("data_channel"), 0);
    EXPECT_EQ(found.get<int64_t>("pcie_vector"), 0);
    EXPECT_EQ(found.get<std::string>("transport"), "pcie");
    // Inputs preserved: the BDF stays the key, the placeholder MAC is untouched.
    EXPECT_EQ(found.get<std::string>("peer_ip"), PEER);
    EXPECT_EQ(found.get<std::string>("mac_id"), NULL_MAC);
    EXPECT_EQ(found.get<std::string>("pcie_backend"), "loopback");

    // get_module(metadata) resolves to the same cached Module enumerate() loaded.
    auto module_handle = Adapter::get_adapter().get_module(found);
    ASSERT_NE(module_handle, nullptr);
}

TEST_F(HololinkAdapterHsbPcie, UseSensorRetargetsAndRejectsOutOfRange)
{
    using hololink::module::Adapter;
    using hololink::module::EnumerationMetadata;

    EnumerationMetadata metadata = pcie_metadata("pcie-use-sensor");
    Adapter::get_adapter().use_sensor(metadata, 0);
    EXPECT_EQ(metadata.get<int64_t>("vp_address"), 0x1000);
    EXPECT_EQ(metadata.get<int64_t>("pcie_vector"), 0);

    Adapter::get_adapter().use_sensor(metadata, 1);
    EXPECT_EQ(metadata.get<int64_t>("sensor_number"), 1);
    EXPECT_EQ(metadata.get<int64_t>("sensor"), 1); // one SIF per sensor
    EXPECT_EQ(metadata.get<int64_t>("vp_address"), 0x1040);
    EXPECT_EQ(metadata.get<int64_t>("vp_mask"), 0x2);
    EXPECT_EQ(metadata.get<int64_t>("sif_address"), 0x01010000);
    EXPECT_EQ(metadata.get<int64_t>("hif_address"), 0x02000300); // data plane 0 stays
    EXPECT_EQ(metadata.get<int64_t>("data_channel"), 1);
    EXPECT_EQ(metadata.get<int64_t>("pcie_vector"), 1);
    EXPECT_EQ(metadata.get<int64_t>("frame_end_event"), 17);

    // Two sensors only.
    EXPECT_THROW(Adapter::get_adapter().use_sensor(metadata, 2), std::runtime_error);
    EXPECT_THROW(Adapter::get_adapter().use_sensor(metadata, -1), std::runtime_error);
}

TEST_F(HololinkAdapterHsbPcie, HololinkStartsAndReadsOverLoopback)
{
    using hololink::module::Adapter;
    using hololink::module::EnumerationMetadata;
    using hololink::module::HololinkInterfaceV1;
    using hololink::module::I2cInterfaceV1;
    using hololink::module::RoceDataChannelInterfaceV1;

    EnumerationMetadata metadata = pcie_metadata("pcie-start-0000:00:00.0");
    Adapter::get_adapter().use_sensor(metadata, 0);

    auto hololink = HololinkInterfaceV1::get_service(metadata);
    ASSERT_NE(hololink, nullptr);
    EXPECT_EQ(hololink->enumeration_metadata().get<std::string>("peer_ip"), PEER);
    // Configured with data_plane, so the default DataChannel exists.
    EXPECT_NE(hololink->default_data_channel(), nullptr);

    // start() runs configure_hsb() over the loopback endpoint: no sockets, no daemon.
    EXPECT_EQ(hololink->start(), HOLOLINK_MODULE_OK);

    std::vector<uint32_t> values;
    EXPECT_EQ(hololink->read_uint32({ 0x80 }, values), HOLOLINK_MODULE_OK);
    ASSERT_EQ(values.size(), 1u);
    EXPECT_EQ(values[0], 0x2606u);

    // Block read of the identity words, then a write/readback on a forwarded BAR0 offset.
    EXPECT_EQ(hololink->read_uint32({ 0x80, 0x84, 0x88 }, values), HOLOLINK_MODULE_OK);
    ASSERT_EQ(values.size(), 3u);
    EXPECT_EQ(values[0], 0x2606u);
    EXPECT_EQ(values[1], 20260219u);
    EXPECT_EQ(values[2], 0x484C5043u);

    EXPECT_EQ(hololink->write_uint32({ 0x1000, 0x1004 }, { 0xCAFEF00Du, 0x12345678u }), HOLOLINK_MODULE_OK);
    EXPECT_EQ(hololink->read_uint32({ 0x1000, 0x1004 }, values), HOLOLINK_MODULE_OK);
    ASSERT_EQ(values.size(), 2u);
    EXPECT_EQ(values[0], 0xCAFEF00Du);
    EXPECT_EQ(values[1], 0x12345678u);

    EXPECT_EQ(hololink->or_uint32(0x1004, 0x80000000u), HOLOLINK_MODULE_OK);
    EXPECT_EQ(hololink->read_uint32({ 0x1004 }, values), HOLOLINK_MODULE_OK);
    EXPECT_EQ(values[0], 0x92345678u);

    // The I2C child constructs against the configured PCIe Hololink. Not transacted: the
    // loopback has no sequencer behind it.
    auto i2c = hololink->get_i2c<I2cInterfaceV1>(0, 0x1A);
    EXPECT_NE(i2c, nullptr);

    // Ethernet data-plane configuration is declined on this transport.
    auto roce = RoceDataChannelInterfaceV1::get_service(metadata, /*allow_null=*/true);
    EXPECT_EQ(roce, nullptr);

    EXPECT_EQ(hololink->stop(), HOLOLINK_MODULE_OK);
}

// The removed engine-selection key must not break enumeration of older metadata: it is
// ignored (with a warning) and the software sequencer remains the only engine.
TEST_F(HololinkAdapterHsbPcie, LegacyEngineSelectionKeyIsIgnored)
{
    using hololink::module::Adapter;
    using hololink::module::EnumerationMetadata;
    using hololink::module::HololinkInterfaceV1;

    EnumerationMetadata metadata = pcie_metadata("pcie-legacy-engine-key");
    metadata["pcie_apb_mode"] = std::string("fpga");
    Adapter::get_adapter().use_sensor(metadata, 0);

    std::shared_ptr<HololinkInterfaceV1> hololink;
    ASSERT_NO_THROW(hololink = HololinkInterfaceV1::get_service(metadata));
    ASSERT_NE(hololink, nullptr);
    EXPECT_EQ(hololink->start(), HOLOLINK_MODULE_OK);
    std::vector<uint32_t> values;
    EXPECT_EQ(hololink->read_uint32({ 0x80 }, values), HOLOLINK_MODULE_OK);
    ASSERT_EQ(values.size(), 1u);
    EXPECT_EQ(values[0], 0x2606u);
    EXPECT_EQ(hololink->stop(), HOLOLINK_MODULE_OK);
}

TEST_F(HololinkAdapterHsbPcie, EthernetShapedAndUntaggedNodeMetadataAreDeclined)
{
    using hololink::module::Adapter;
    using hololink::module::EnumerationMetadata;

    bool enriched_fired = false;
    auto handle = Adapter::get_adapter().register_all(
        [&enriched_fired](const EnumerationMetadata&) { enriched_fired = true; });

    // Carries this module's UUID but is Ethernet-shaped: an IPv4 peer with no bootp client
    // address to carry a PCI function. Declined.
    EnumerationMetadata bootp;
    bootp["fpga_uuid"] = std::string(HSB_PCIE_UUID);
    bootp["peer_ip"] = std::string("192.0.2.9");
    bootp["serial_number"] = std::string("bootp-009");
    bootp["control_port"] = static_cast<int64_t>(8192);
    bootp["data_plane"] = static_cast<int64_t>(0);
    EXPECT_NO_THROW(Adapter::get_adapter().enumerate(bootp));
    EXPECT_FALSE(enriched_fired);

    // Still declined with transport = pcie stamped on it, and with a malformed client address.
    bootp["transport"] = std::string("pcie");
    EXPECT_NO_THROW(Adapter::get_adapter().enumerate(bootp));
    EXPECT_FALSE(enriched_fired);
    bootp["client_ip_address"] = std::string("not-an-address");
    EXPECT_NO_THROW(Adapter::get_adapter().enumerate(bootp));
    EXPECT_FALSE(enriched_fired);

    // A /dev/hololink_pcieN peer is only a convenience spelling; without transport = pcie
    // it is not recognised as this module's channel.
    EnumerationMetadata node_peer = pcie_metadata("pcie-node-untagged", DEVICE_NODE_PEER);
    ASSERT_FALSE(node_peer.contains("transport"));
    EXPECT_NO_THROW(Adapter::get_adapter().enumerate(node_peer));
    EXPECT_FALSE(enriched_fired);

    // mac_id is no longer consulted: a BDF peer without it enumerates.
    EnumerationMetadata no_mac = pcie_metadata("pcie-no-mac");
    no_mac.erase("mac_id");
    EXPECT_NO_THROW(Adapter::get_adapter().enumerate(no_mac));
    EXPECT_TRUE(enriched_fired);

    Adapter::get_adapter().unregister(handle);
}

// An OLDER hololink_pcie driver's announcement (htype 1) after the host parser: the driver
// packed the BDF ((domain << 16) | (bus << 8) | devfn) into both the datagram's IPv4 source and
// ciaddr, so peer_ip and client_ip_address arrive as the same dotted quad. The module restores
// peer_ip to the textual BDF before the Adapter matches subscribers, so wait_for_channel(bdf)
// resolves, keeps the source in pcie_source_ip, and replaces the placeholder serial with
// "pcie-<bdf>". (Current drivers announce with htype 0x50; see KernelBootpBdfPeerIsStamped.)
TEST_F(HololinkAdapterHsbPcie, KernelBootpAnnouncementIsRestoredToTheBdf)
{
    using hololink::module::Adapter;
    using hololink::module::EnumerationMetadata;
    using hololink::module::HololinkInterfaceV1;

    constexpr const char* BDF = "0005:01:00.0";
    EnumerationMetadata bootp = pcie_metadata("ignored-by-module", "0.5.1.0");
    bootp["client_ip_address"] = std::string("0.5.1.0");
    bootp["mac_id"] = std::string("02:50:00:05:01:00");
    bootp["transaction_id"] = static_cast<int64_t>((5u << 16) | 0x0100u);
    bootp["serial_number"] = std::string("50434945050100"); // the 7 raw bytes as the parser renders them
    bootp["interface"] = std::string("lo");
    bootp["control_port"] = static_cast<int64_t>(8192);
    bootp.erase("transport");

    // Matched on the restored BDF, not on the packed dotted quad.
    EnumerationMetadata found = enumerate_and_wait(bootp, BDF);
    ASSERT_TRUE(found.contains("module_name"));
    EXPECT_EQ(found.get<std::string>("module_name"), "hsb_pcie");
    EXPECT_EQ(found.get<std::string>("peer_ip"), BDF);
    EXPECT_EQ(found.get<std::string>("pcie_bdf"), BDF);
    EXPECT_EQ(found.get<std::string>("pcie_source_ip"), "0.5.1.0");
    EXPECT_EQ(found.get<std::string>("serial_number"), std::string("pcie-") + BDF);
    EXPECT_EQ(found.get<std::string>("transport"), "pcie");
    EXPECT_EQ(found.get<int64_t>("pcie_domain"), 5);
    EXPECT_EQ(found.get<int64_t>("pcie_bus"), 1);
    EXPECT_EQ(found.get<int64_t>("pcie_device"), 0);
    EXPECT_EQ(found.get<int64_t>("pcie_function"), 0);
    EXPECT_EQ(found.get<std::string>("mac_id"), "02:50:00:05:01:00") << "informational, left as parsed";

    // The restored record backs a working Hololink (loopback backend from pcie_backend).
    Adapter::get_adapter().use_sensor(found, 0);
    auto hololink = HololinkInterfaceV1::get_service(found);
    ASSERT_NE(hololink, nullptr);
    EXPECT_EQ(hololink->start(), HOLOLINK_MODULE_OK);
    std::vector<uint32_t> values;
    EXPECT_EQ(hololink->read_uint32({ 0x80 }, values), HOLOLINK_MODULE_OK);
    ASSERT_EQ(values.size(), 1u);
    EXPECT_EQ(values[0], 0x2606u);
    // This UUID is the CertusPro-NX Versa's: start() programmed the I2C divider for its
    // 60 MHz APB clock (400 kHz SCL -> CLK_CNT 75), read back through the HSB I2C alias.
    EXPECT_EQ(hololink->read_uint32({ 0x0300020C }, values), HOLOLINK_MODULE_OK);
    ASSERT_EQ(values.size(), 1u);
    EXPECT_EQ(values[0], 75u);
    EXPECT_EQ(hololink->stop(), HOLOLINK_MODULE_OK);
}

// The current driver's announcement (htype 0x50) as hand-built metadata that already spells
// peer_ip and mac_id as the BDF and carries a decoded bootp_status (no raw packet, so the
// module's own tail decode has nothing to read and keeps what is there). The module stamps
// pcie_bdf / pcie_domain / pcie_bus / pcie_device / pcie_function / transport and applies the
// serial rule: a Device Serial Number (bootp_status bit1) is kept, the placeholder becomes
// "pcie-<bdf>". The last block is the record as the transport-agnostic parser actually
// delivers it -- peer_ip the IPv4 source, mac_id the chaddr bytes colon-hex -- which the
// module's update_metadata renders to the BDF before anything is matched on peer_ip.
TEST_F(HololinkAdapterHsbPcie, KernelBootpBdfPeerIsStamped)
{
    using hololink::module::EnumerationMetadata;

    constexpr const char* BDF = "0005:01:00.0";
    struct Case {
        int64_t bootp_status;
        const char* announced_serial;
        const char* expected_serial;
        const char* expected_source;
    };
    const Case cases[] = {
        { 0, "50434945050100", "pcie-0005:01:00.0", "bdf" }, // placeholder: no DSN capability
        { 2, "08070605040302", "08070605040302", "dsn" }, // Device Serial Number
        { 3, "08070605040302", "08070605040302", "dsn" }, // ... with a PTP snapshot too
    };
    for (const Case& c : cases) {
        EnumerationMetadata bootp = pcie_metadata("ignored-by-module", BDF);
        bootp.erase("pcie_bdf");
        bootp.erase("transport");
        bootp["hardware_type"] = static_cast<int64_t>(0x50);
        bootp["hardware_address_length"] = static_cast<int64_t>(4);
        bootp["mac_id"] = std::string(BDF);
        bootp["client_ip_address"] = std::string("0.5.1.0");
        bootp["serial_number"] = std::string(c.announced_serial);
        bootp["bootp_status"] = c.bootp_status;

        EnumerationMetadata found = enumerate_and_wait(bootp, BDF);
        EXPECT_EQ(found.get<std::string>("module_name"), "hsb_pcie");
        EXPECT_EQ(found.get<std::string>("peer_ip"), BDF);
        EXPECT_EQ(found.get<std::string>("pcie_bdf"), BDF);
        EXPECT_FALSE(found.contains("pcie_source_ip"));
        EXPECT_EQ(found.get<int64_t>("pcie_domain"), 5);
        EXPECT_EQ(found.get<int64_t>("pcie_bus"), 1);
        EXPECT_EQ(found.get<int64_t>("pcie_device"), 0);
        EXPECT_EQ(found.get<int64_t>("pcie_function"), 0);
        EXPECT_EQ(found.get<std::string>("transport"), "pcie");
        EXPECT_EQ(found.get<std::string>("serial_number"), c.expected_serial) << "status " << c.bootp_status;
        EXPECT_EQ(found.get<std::string>("pcie_serial_source"), c.expected_source) << "status " << c.bootp_status;
        EXPECT_EQ(found.get<std::string>("mac_id"), BDF) << "left as parsed";
    }

    // A record whose ciaddr disagrees with chaddr is accepted on chaddr (logged).
    EnumerationMetadata odd = pcie_metadata("ignored-by-module", BDF);
    odd.erase("transport");
    odd["hardware_type"] = static_cast<int64_t>(0x50);
    odd["client_ip_address"] = std::string("0.6.1.0");
    odd["serial_number"] = std::string("50434945050100");
    EnumerationMetadata found = enumerate_and_wait(odd, BDF);
    EXPECT_EQ(found.get<std::string>("peer_ip"), BDF);
    EXPECT_EQ(found.get<int64_t>("pcie_domain"), 5);
    EXPECT_EQ(found.get<std::string>("serial_number"), std::string("pcie-") + BDF);

    // As the parser delivers it: htype 0x50, chaddr in hardware_address (trimmed to hlen),
    // mac_id colon-hex, peer_ip the IPv4 source. The module renders the BDF from chaddr into
    // peer_ip, mac_id and hardware_address_text; the source is not kept separately (it is the
    // packed BDF client_ip_address already carries).
    for (const char* source : { "0.5.1.0", "127.0.0.1" }) {
        EnumerationMetadata parsed = pcie_metadata("ignored-by-module", source);
        parsed.erase("pcie_bdf");
        parsed.erase("transport");
        parsed["hardware_type"] = static_cast<int64_t>(0x50);
        parsed["hardware_address_length"] = static_cast<int64_t>(4);
        parsed["hardware_address"] = std::vector<uint8_t> { 0x00, 0x05, 0x01, 0x00 };
        parsed["mac_id"] = std::string("00:05:01:00");
        parsed["client_ip_address"] = std::string("0.5.1.0");
        parsed["serial_number"] = std::string("50434945050100");

        EnumerationMetadata rendered = enumerate_and_wait(parsed, BDF);
        EXPECT_EQ(rendered.get<std::string>("peer_ip"), BDF) << source;
        EXPECT_EQ(rendered.get<std::string>("mac_id"), BDF) << source;
        EXPECT_EQ(rendered.get<std::string>("hardware_address_text"), BDF) << source;
        EXPECT_EQ(rendered.get<std::string>("pcie_bdf"), BDF) << source;
        EXPECT_FALSE(rendered.contains("pcie_source_ip")) << source;
        EXPECT_EQ(rendered.get<int64_t>("pcie_domain"), 5) << source;
        EXPECT_EQ(rendered.get<std::string>("serial_number"), std::string("pcie-") + BDF) << source;
        EXPECT_EQ(rendered.get<std::string>("transport"), "pcie") << source;
    }
}

// client_ip_address is the authority for the BDF on the legacy (htype 1) shape. An
// announcement sourced from 127.0.0.1 -- an older driver, or a userspace sender, which cannot
// source from a 0.0.0.0/8 address -- is restored the same way, and so is one whose source
// disagrees with ciaddr (logged, ciaddr wins). In every case pcie_source_ip keeps what arrived.
TEST_F(HololinkAdapterHsbPcie, BootpSourceOtherThanThePackedBdfStillRestoresFromCiaddr)
{
    using hololink::module::EnumerationMetadata;

    constexpr const char* BDF = "0005:01:00.0";
    for (const char* source : { "127.0.0.1", "0.6.1.0" }) {
        EnumerationMetadata bootp = pcie_metadata("ignored-by-module", source);
        bootp["client_ip_address"] = std::string("0.5.1.0");
        bootp["serial_number"] = std::string("50434945050100");
        bootp.erase("transport");

        EnumerationMetadata found = enumerate_and_wait(bootp, BDF);
        EXPECT_EQ(found.get<std::string>("peer_ip"), BDF) << source;
        EXPECT_EQ(found.get<std::string>("pcie_bdf"), BDF) << source;
        EXPECT_EQ(found.get<std::string>("pcie_source_ip"), source);
        EXPECT_EQ(found.get<std::string>("serial_number"), std::string("pcie-") + BDF) << source;
    }
}

// The convenience spelling: peer_ip = /dev/hololink_pcieN is accepted when the metadata says
// transport = pcie, and the loopback backing takes its minor from pcie_minor.
TEST_F(HololinkAdapterHsbPcie, DeviceNodePeerIsAcceptedWithTransportPcie)
{
    using hololink::module::Adapter;
    using hololink::module::EnumerationMetadata;
    using hololink::module::HololinkInterfaceV1;

    EnumerationMetadata metadata = pcie_metadata("pcie-node-tagged", DEVICE_NODE_PEER);
    metadata["transport"] = std::string("pcie");
    EnumerationMetadata found = enumerate_and_wait(metadata, DEVICE_NODE_PEER);
    ASSERT_TRUE(found.contains("module_name"));
    EXPECT_EQ(found.get<std::string>("module_name"), "hsb_pcie");
    EXPECT_EQ(found.get<std::string>("peer_ip"), DEVICE_NODE_PEER);

    Adapter::get_adapter().use_sensor(found, 0);
    auto hololink = HololinkInterfaceV1::get_service(found);
    ASSERT_NE(hololink, nullptr);
    EXPECT_EQ(hololink->start(), HOLOLINK_MODULE_OK);
    std::vector<uint32_t> values;
    EXPECT_EQ(hololink->read_uint32({ 0x80 }, values), HOLOLINK_MODULE_OK);
    ASSERT_EQ(values.size(), 1u);
    EXPECT_EQ(values[0], 0x2606u);
    EXPECT_EQ(hololink->stop(), HOLOLINK_MODULE_OK);
}

// The PCIe data channel: the module-side owner of the endpoint's receive registers. Two
// channels on one board (sensor 0 and sensor 1) each program their own PCIE_DP block and
// MSI-X entry from a fake receiver attachment and read-modify-write only their own bit of the
// shared DP_PKT_VIP_MASK; everything is read back through the same HololinkInterfaceV1, so
// the address policy the channel is subject to is the one under test too.
TEST_F(HololinkAdapterHsbPcie, PcieDataChannelProgramsAndArmsPerSensor)
{
    using hololink::module::Adapter;
    using hololink::module::DataChannelInterfaceV1;
    using hololink::module::EnumerationMetadata;
    using hololink::module::HololinkInterfaceV1;
    using hololink::module::hsb_pcie::PcieChannelDiagnostics;
    using hololink::module::hsb_pcie::PcieDataChannelInterfaceV1;
    namespace regs = ::hololink::pcie;

    EnumerationMetadata md0 = pcie_metadata("pcie-data-channel");
    Adapter::get_adapter().use_sensor(md0, 0);
    EnumerationMetadata md1 = md0;
    Adapter::get_adapter().use_sensor(md1, 1);

    auto hololink = HololinkInterfaceV1::get_service(md0);
    ASSERT_NE(hololink, nullptr);
    ASSERT_EQ(hololink->start(), HOLOLINK_MODULE_OK);

    // Metadata-form lookup configures the channel, its anchor and the Hololink in one go;
    // the two sensors resolve to two channels under two (serial, data_channel) keys, each
    // wrapping the anchor published under the same key.
    auto channel0 = PcieDataChannelInterfaceV1::get_service(md0);
    auto channel1 = PcieDataChannelInterfaceV1::get_service(md1);
    ASSERT_NE(channel0, nullptr);
    ASSERT_NE(channel1, nullptr);
    EXPECT_NE(channel0, channel1);
    EXPECT_EQ(channel0, PcieDataChannelInterfaceV1::get_service(md0)) << "cached per key";
    EXPECT_EQ(channel0->sensor(), 0u);
    EXPECT_EQ(channel0->vector(), 0u);
    EXPECT_EQ(channel0->vp_mask(), 0x1u);
    EXPECT_EQ(channel1->sensor(), 1u);
    EXPECT_EQ(channel1->vector(), 1u);
    EXPECT_EQ(channel1->vp_mask(), 0x2u);
    auto anchor1 = DataChannelInterfaceV1::get_service(md1);
    ASSERT_NE(anchor1, nullptr);
    EXPECT_EQ(anchor1->hololink(), hololink);

    const uint32_t vip_mask = regs::DP_PKT_BASE + regs::DP_PKT_VIP_MASK; // 0x3030C
    const uint32_t dp_pkt_len = regs::DP_PKT_BASE + regs::DP_PKT_LEN; // 0x30304
    ASSERT_EQ(vip_mask, 0x3030Cu);
    EXPECT_EQ(read_word(*hololink, vip_mask), 0u) << "nothing armed yet";

    // Nothing attached: arm() is refused, disarm() and detach_receiver() are harmless.
    EXPECT_NE(channel0->arm(), HOLOLINK_MODULE_OK);
    EXPECT_EQ(channel0->disarm(), HOLOLINK_MODULE_OK);
    EXPECT_EQ(channel0->detach_receiver(), HOLOLINK_MODULE_OK);
    EXPECT_EQ(read_word(*hololink, vip_mask), 0u);

    // An attachment on the wrong vector is rejected before any register is touched.
    EXPECT_NE(channel0->attach_receiver(std::make_shared<FakeAttachment>(1), 2), HOLOLINK_MODULE_OK);
    EXPECT_EQ(read_word(*hololink, regs::pcie_dp_reg(0, regs::PCIE_DP_REQUESTER_ID)), 0u);
    // So is an unrepresentable one (chunk_pages beyond the 8-bit DP_PKT_LEN).
    EXPECT_NE(channel0->attach_receiver(std::make_shared<FakeAttachment>(0), 256), HOLOLINK_MODULE_OK);
    EXPECT_EQ(read_word(*hololink, regs::pcie_dp_reg(0, regs::PCIE_DP_REQUESTER_ID)), 0u);

    // The MSI-X table is the kernel's: the channel must never write an entry. Snapshot both
    // sensors' entries (0x8000, 0x8010) before anything is attached and hold them to it.
    const uint32_t entry0 = regs::msix_table_entry(0);
    const uint32_t entry1 = regs::msix_table_entry(1);
    ASSERT_EQ(entry0, 0x8000u);
    ASSERT_EQ(entry1, 0x8010u);
    const auto snapshot_entry = [&](uint32_t entry) {
        return std::array<uint32_t, 4> {
            read_word(*hololink, entry + regs::MSIX_MESSAGE_ADDR_LO),
            read_word(*hololink, entry + regs::MSIX_MESSAGE_ADDR_HI),
            read_word(*hololink, entry + regs::MSIX_MESSAGE_DATA),
            read_word(*hololink, entry + regs::MSIX_VECTOR_CONTROL),
        };
    };
    const auto entry0_before = snapshot_entry(entry0);
    const auto entry1_before = snapshot_entry(entry1);

    // Sensor 0: program + arm. Every PCIE_DP word of block 0 (0x1000), DP_PKT_LEN and VIP
    // bit 0 -- and nothing in the MSI-X table.
    auto attachment0 = std::make_shared<FakeAttachment>(0);
    ASSERT_EQ(channel0->attach_receiver(attachment0, 2), HOLOLINK_MODULE_OK);
    const auto dp0 = [](uint32_t reg) { return regs::pcie_dp_reg(0, reg); };
    ASSERT_EQ(dp0(regs::PCIE_DP_BUF_BASE_HI), 0x1000u);
    EXPECT_EQ(read_word(*hololink, dp0(regs::PCIE_DP_BUF_BASE_HI)), 0x28u);
    EXPECT_EQ(read_word(*hololink, dp0(regs::PCIE_DP_REQUESTER_ID)), FakeAttachment::REQUESTER_ID);
    EXPECT_EQ(read_word(*hololink, dp0(regs::PCIE_DP_BUF_INC)), FakeAttachment::BUF_INC_PAGES);
    EXPECT_EQ(read_word(*hololink, dp0(regs::PCIE_DP_BUF_START)), 0x00e00000u);
    EXPECT_EQ(read_word(*hololink, dp0(regs::PCIE_DP_BUF_LEN)), 0x278F80u);
    EXPECT_EQ(read_word(*hololink, dp0(regs::PCIE_DP_BUF_PTR_RANGE)), FakeAttachment::NUM_BUFFERS - 1);
    EXPECT_EQ(read_word(*hololink, dp0(regs::PCIE_DP_METADATA_CTRL)), 0u);
    EXPECT_EQ(read_word(*hololink, dp_pkt_len), 2u);
    EXPECT_EQ(snapshot_entry(entry0), entry0_before) << "MSI-X entry 0 is the kernel's";
    EXPECT_EQ(read_word(*hololink, vip_mask), 0x1u);

    // Sensor 1 alongside: its own block (0x1040); sensor 0 keeps streaming.
    auto attachment1 = std::make_shared<FakeAttachment>(1);
    ASSERT_EQ(channel1->attach_receiver(attachment1, 2), HOLOLINK_MODULE_OK);
    const auto dp1 = [](uint32_t reg) { return regs::pcie_dp_reg(1, reg); };
    ASSERT_EQ(dp1(regs::PCIE_DP_BUF_BASE_HI), 0x1040u);
    EXPECT_EQ(read_word(*hololink, dp1(regs::PCIE_DP_BUF_BASE_HI)), 0x28u);
    EXPECT_EQ(read_word(*hololink, dp1(regs::PCIE_DP_BUF_START)), 0x00e00000u);
    EXPECT_EQ(read_word(*hololink, dp1(regs::PCIE_DP_BUF_LEN)), 0x278F80u);
    EXPECT_EQ(read_word(*hololink, dp1(regs::PCIE_DP_BUF_PTR_RANGE)), 2u);
    EXPECT_EQ(snapshot_entry(entry1), entry1_before) << "MSI-X entry 1 is the kernel's";
    EXPECT_EQ(snapshot_entry(entry0), entry0_before);
    EXPECT_EQ(read_word(*hololink, vip_mask), 0x3u) << "both armed";

    // A second attach without a detach is an error and changes nothing.
    EXPECT_NE(channel1->attach_receiver(std::make_shared<FakeAttachment>(1), 2), HOLOLINK_MODULE_OK);
    EXPECT_EQ(read_word(*hololink, vip_mask), 0x3u);

    // diagnostics() reads the programmed values back.
    PcieChannelDiagnostics d {};
    ASSERT_EQ(channel1->diagnostics(d), HOLOLINK_MODULE_OK);
    EXPECT_EQ(d.vip_mask, 0x3u);
    EXPECT_EQ(d.dp_pkt_len, 2u);
    EXPECT_EQ(d.fixed_msb, 0x28u);
    EXPECT_EQ(d.requester_id, FakeAttachment::REQUESTER_ID);
    EXPECT_EQ(d.buf_inc, FakeAttachment::BUF_INC_PAGES);
    EXPECT_EQ(d.buf_start, 0x00e00000u);
    EXPECT_EQ(d.buf_len, 0x278F80u);
    EXPECT_EQ(d.ptr_range, 2u);
    EXPECT_EQ(d.metadata_ctrl, 0u);
    // The MSI-X words are read back as diagnostics only: whatever the table holds.
    EXPECT_EQ(d.msix_addr_lo, entry1_before[0]);
    EXPECT_EQ(d.msix_addr_hi, entry1_before[1]);
    EXPECT_EQ(d.msix_data, entry1_before[2]);
    EXPECT_EQ(d.msix_vector_control, entry1_before[3]);
    EXPECT_EQ(d.scratch, read_word(*hololink, regs::GLB_SCRATCH));

    // Disarm / re-arm sensor 1 only.
    EXPECT_EQ(channel1->disarm(), HOLOLINK_MODULE_OK);
    EXPECT_EQ(read_word(*hololink, vip_mask), 0x1u);
    EXPECT_EQ(channel1->arm(), HOLOLINK_MODULE_OK);
    EXPECT_EQ(read_word(*hololink, vip_mask), 0x3u);

    // Detach sensor 1: its VIP bit clears; sensor 0 is untouched; the MSI-X table still is.
    EXPECT_EQ(channel1->detach_receiver(), HOLOLINK_MODULE_OK);
    EXPECT_EQ(read_word(*hololink, vip_mask), 0x1u);
    EXPECT_EQ(snapshot_entry(entry1), entry1_before);
    EXPECT_EQ(snapshot_entry(entry0), entry0_before);
    EXPECT_EQ(read_word(*hololink, dp1(regs::PCIE_DP_BUF_START)), 0x00e00000u) << "destination words are left as programmed";

    // A detached channel can attach again.
    ASSERT_EQ(channel1->attach_receiver(attachment1, 2), HOLOLINK_MODULE_OK);
    EXPECT_EQ(read_word(*hololink, vip_mask), 0x3u);
    EXPECT_EQ(channel1->detach_receiver(), HOLOLINK_MODULE_OK);

    // DP_PKT_LEN is one word for every sensor: while sensor 0 streams with 2 pages, sensor 1
    // may not retarget it to 1. The attach is refused, the word and sensor 0's arm stay.
    EXPECT_EQ(channel1->attach_receiver(attachment1, 1), HOLOLINK_MODULE_INVALID_PARAMETER);
    EXPECT_EQ(read_word(*hololink, dp_pkt_len), 2u);
    EXPECT_EQ(read_word(*hololink, vip_mask), 0x1u);
    // With sensor 0 detached nothing else depends on the word, so it may change.
    EXPECT_EQ(channel0->detach_receiver(), HOLOLINK_MODULE_OK);
    ASSERT_EQ(channel1->attach_receiver(attachment1, 1), HOLOLINK_MODULE_OK);
    EXPECT_EQ(read_word(*hololink, dp_pkt_len), 1u);
    EXPECT_EQ(read_word(*hololink, vip_mask), 0x2u);
    EXPECT_EQ(channel1->detach_receiver(), HOLOLINK_MODULE_OK);
    EXPECT_EQ(read_word(*hololink, vip_mask), 0u);
    EXPECT_EQ(snapshot_entry(entry0), entry0_before) << "never written, from first attach to last detach";

    EXPECT_EQ(hololink->stop(), HOLOLINK_MODULE_OK);
}
