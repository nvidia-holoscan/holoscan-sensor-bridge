/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Hardware-free tests for the hsb_pcie module's sysfs/ioctl enumerator
 * (hololink_module/tools/pcie_device_info.hpp, private to hololink_pcie_identity): UUID text round trips, the
 * standardized section-F metadata table built from a hand-built
 * PcieDeviceInfo (peer_ip = BDF), scan_pcie_devices() against a fake
 * /sys/class/hololink_pcie tree in a temp dir, and the header-only BDF <->
 * /dev/hololink_pcieN resolvers (pcie_sysfs.hpp) over
 * both the /sys/bus/pci/devices/<bdf>/hololink_pcie/ fast path and the
 * class-walk fallback. The fake dev dir is empty, so the IDENTITY ioctl path
 * is skipped and the sysfs-attribute fallback is what gets exercised. Also
 * (the kernel-uevent parser is gone: discovery is bootp-only, published by the
 * kernel driver) the
 * Adapter's hotplug monitor feeds, over literal kernel-style datagrams --
 * the netlink socket itself is not exercised here.
 */

#include <gtest/gtest.h>

#include <cerrno>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "hololink/module/enumeration_metadata.hpp"
#include "pcie_device_info.hpp"

namespace pcie = hololink::module::pcie;
using hololink::module::EnumerationMetadata;

namespace {

// Byte order of the module UUID as the driver reports it: uuid[0..3] = HSB_UUID_3 MSB-first.
constexpr pcie::Uuid MODULE_UUID_BYTES = {
    0x58, 0xc1, 0x41, 0x9b, 0x1e, 0x7a, 0x41, 0x20,
    0x95, 0xfd, 0x4e, 0xfc, 0x15, 0x61, 0x87, 0xc1
};

void write_file(const std::filesystem::path& path, const std::string& text)
{
    std::ofstream out(path);
    ASSERT_TRUE(out) << "cannot create " << path;
    out << text;
}

// A fake sysfs / dev tree that mimics what the hololink_pcie driver populates.
class FakeTree {
public:
    FakeTree()
    {
        // temp_directory_path() honors TMPDIR, so a sandboxed run lands in its scratch dir.
        const std::string base
            = (std::filesystem::temp_directory_path() / "hololink_pcie_enum_test.XXXXXX").string();
        std::vector<char> buf(base.begin(), base.end());
        buf.push_back('\0');
        const char* made = ::mkdtemp(buf.data());
        if (made == nullptr) {
            throw std::runtime_error("mkdtemp failed");
        }
        root_ = made;
        sysfs_class_ = root_ / "sys" / "class" / "hololink_pcie";
        pci_devices_ = root_ / "sys" / "bus" / "pci" / "devices";
        dev_dir_ = root_ / "dev";
        std::filesystem::create_directories(sysfs_class_);
        std::filesystem::create_directories(pci_devices_);
        std::filesystem::create_directories(dev_dir_);
    }

    ~FakeTree()
    {
        std::error_code ec;
        std::filesystem::remove_all(root_, ec);
    }

    const std::filesystem::path& root() const { return root_; }
    const std::filesystem::path& sysfs_class() const { return sysfs_class_; }
    const std::filesystem::path& dev_dir() const { return dev_dir_; }

    /// /sys/devices/<root_bus>/<chain...>/<bdf>, returned canonical, plus the bus view
    /// /sys/bus/pci/devices/<bdf> -> that directory (the last chain element is the BDF).
    std::filesystem::path add_pci_device(const std::string& root_bus,
        const std::vector<std::string>& chain)
    {
        std::filesystem::path p = root_ / "sys" / "devices" / root_bus;
        for (const auto& c : chain) {
            p /= c;
        }
        std::filesystem::create_directories(p);
        const std::filesystem::path canonical = std::filesystem::canonical(p);
        std::filesystem::create_directory_symlink(canonical, pci_devices_ / chain.back());
        return canonical;
    }

    /// hololink_pcieN with a `device` link and optional attribute files. With `with_pci_child`
    /// the class device also appears under the PCI device, as device_create(class,
    /// &pdev->dev, ...) makes it: <device_dir>/hololink_pcie/hololink_pcieN.
    std::filesystem::path add_class_device(unsigned minor, const std::filesystem::path& device_dir,
        bool with_bdf_attr, const std::string& bdf, const std::string& uuid_text,
        bool with_pci_child = true)
    {
        const std::filesystem::path entry = sysfs_class_ / ("hololink_pcie" + std::to_string(minor));
        std::filesystem::create_directories(entry);
        std::filesystem::create_directory_symlink(device_dir, entry / "device");
        if (with_pci_child) {
            std::filesystem::create_directories(
                device_dir / "hololink_pcie" / ("hololink_pcie" + std::to_string(minor)));
        }
        if (with_bdf_attr) {
            write_file(entry / "bdf", bdf + "\n");
        }
        if (!uuid_text.empty()) {
            write_file(entry / "uuid", uuid_text + "\n");
            write_file(entry / "hsb_version", "0x00002606\n");
            write_file(entry / "hsb_date", "0x20260901\n");
            write_file(entry / "requester_id", "0x0600\n");
            write_file(entry / "msix_vectors", "64\n");
        }
        return entry;
    }

private:
    std::filesystem::path root_;
    std::filesystem::path sysfs_class_;
    std::filesystem::path pci_devices_;
    std::filesystem::path dev_dir_;
};

} // namespace

TEST(PcieEnumerator, UuidTextRoundTrip)
{
    EXPECT_EQ(pcie::uuid_to_text(MODULE_UUID_BYTES), pcie::HSB_PCIE_MODULE_UUID);

    pcie::Uuid parsed {};
    ASSERT_TRUE(pcie::uuid_from_text(pcie::HSB_PCIE_MODULE_UUID, parsed));
    EXPECT_EQ(parsed, MODULE_UUID_BYTES);

    // Upper-case and bare 32-digit forms parse to the same bytes; output is always canonical.
    ASSERT_TRUE(pcie::uuid_from_text("58C1419B-1E7A-4120-95FD-4EFC156187C1", parsed));
    EXPECT_EQ(parsed, MODULE_UUID_BYTES);
    ASSERT_TRUE(pcie::uuid_from_text("58c1419b1e7a412095fd4efc156187c1", parsed));
    EXPECT_EQ(parsed, MODULE_UUID_BYTES);
    EXPECT_EQ(pcie::uuid_to_text(parsed), pcie::HSB_PCIE_MODULE_UUID);

    // Arbitrary bytes round-trip through text.
    pcie::Uuid arbitrary {};
    for (size_t i = 0; i < arbitrary.size(); i++) {
        arbitrary[i] = static_cast<uint8_t>(0x10 * i + i);
    }
    pcie::Uuid back {};
    ASSERT_TRUE(pcie::uuid_from_text(pcie::uuid_to_text(arbitrary), back));
    EXPECT_EQ(back, arbitrary);

    // Malformed inputs are rejected and leave the output untouched.
    pcie::Uuid untouched = MODULE_UUID_BYTES;
    EXPECT_FALSE(pcie::uuid_from_text("", untouched));
    EXPECT_FALSE(pcie::uuid_from_text("58c1419b-1e7a-4120-95fd", untouched));
    EXPECT_FALSE(pcie::uuid_from_text("58c1419b_1e7a_4120_95fd_4efc156187c1", untouched));
    EXPECT_FALSE(pcie::uuid_from_text("g8c1419b-1e7a-4120-95fd-4efc156187c1", untouched));
    EXPECT_EQ(untouched, MODULE_UUID_BYTES);

    // Validity rule matches the driver's HL_PCIE_ID_F_UUID_VALID.
    EXPECT_TRUE(pcie::uuid_is_valid(MODULE_UUID_BYTES));
    pcie::Uuid zeros {};
    EXPECT_FALSE(pcie::uuid_is_valid(zeros));
    pcie::Uuid ones {};
    ones.fill(0xFF);
    EXPECT_FALSE(pcie::uuid_is_valid(ones));
}

TEST(PcieEnumerator, ScanFakeSysfsTree)
{
    FakeTree tree;
    const std::filesystem::path device_dir = tree.add_pci_device("pci0005:00",
        { "0005:00:00.0", "0005:06:00.0" });
    tree.add_class_device(0, device_dir, /*with_bdf_attr=*/true, "0005:06:00.0",
        pcie::HSB_PCIE_MODULE_UUID);

    const std::vector<pcie::PcieDeviceInfo> devices
        = pcie::scan_pcie_devices(tree.sysfs_class(), tree.dev_dir());
    ASSERT_EQ(devices.size(), 1u);
    const pcie::PcieDeviceInfo& d = devices[0];
    EXPECT_EQ(d.minor, 0);
    EXPECT_EQ(d.device_node, (tree.dev_dir() / "hololink_pcie0").string());
    EXPECT_EQ(d.bdf, "0005:06:00.0");
    EXPECT_EQ(d.domain, 5);
    EXPECT_EQ(d.bus, 6);
    EXPECT_EQ(d.device, 0);
    EXPECT_EQ(d.function, 0);
    EXPECT_EQ(d.root_complex, "pci0005:00");
    EXPECT_EQ(d.sysfs_path, device_dir.string());
    EXPECT_EQ(d.uuid, MODULE_UUID_BYTES);
    EXPECT_EQ(d.uuid_text, pcie::HSB_PCIE_MODULE_UUID);
    EXPECT_EQ(d.hsb_version, 0x2606u);
    EXPECT_EQ(d.hsb_date, 0x20260901u);
    EXPECT_EQ(d.requester_id, 0x0600u);
    EXPECT_EQ(d.msix_vectors, 64u);
    // Not served by sysfs; stay at their defaults.
    EXPECT_EQ(d.bar0_len, 0u);
    EXPECT_EQ(d.dma_address_bits, 0u);
    // Flags synthesized from the values: version and uuid valid, no magic in hsb_status.
    EXPECT_NE(d.flags & 0x1u, 0u); // HL_PCIE_ID_F_VERSION_VALID
    EXPECT_NE(d.flags & 0x10u, 0u); // HL_PCIE_ID_F_UUID_VALID
    EXPECT_EQ(d.flags & 0x8u, 0u); // HL_PCIE_ID_F_MAGIC_VALID

    // The scan is keyed on the BDF; the node rides along.
    EXPECT_EQ(d.bdf, "0005:06:00.0");
    EXPECT_EQ(d.uuid_text, pcie::HSB_PCIE_MODULE_UUID);

    // The resolvers find the minor through the same tree -- here via the fast path
    // /sys/bus/pci/devices/<bdf>/hololink_pcie/hololink_pcie0 -- case-insensitively.
    EXPECT_EQ(pcie::minor_for_bdf("0005:06:00.0", tree.sysfs_class()), std::optional<uint16_t>(0));
    EXPECT_EQ(pcie::minor_for_bdf("0000:01:00.0", tree.sysfs_class()), std::nullopt);
    EXPECT_EQ(pcie::minor_for_bdf("not-a-bdf", tree.sysfs_class()), std::nullopt);
    EXPECT_EQ(pcie::device_node_for_bdf("0005:06:00.0", tree.sysfs_class(), tree.dev_dir()),
        std::optional<std::string>((tree.dev_dir() / "hololink_pcie0").string()));
    EXPECT_EQ(pcie::device_node_for_bdf("0000:01:00.0", tree.sysfs_class(), tree.dev_dir()), std::nullopt);

    // Convenience spellings resolve back to the BDF through the class device's bdf attribute.
    EXPECT_EQ(pcie::bdf_for_peer("0005:06:00.0", tree.sysfs_class(), tree.dev_dir()),
        std::optional<std::string>("0005:06:00.0"));
    EXPECT_EQ(pcie::normalize_bdf("0005:0A:00.0"), "0005:0a:00.0");
    EXPECT_EQ(pcie::bdf_for_peer((tree.dev_dir() / "hololink_pcie0").string(), tree.sysfs_class(), tree.dev_dir()),
        std::optional<std::string>("0005:06:00.0"));
    EXPECT_EQ(pcie::bdf_for_peer((tree.dev_dir() / "hololink_pcie9").string(), tree.sysfs_class(), tree.dev_dir()),
        std::nullopt);
    EXPECT_EQ(pcie::bdf_for_peer("192.0.2.9", tree.sysfs_class(), tree.dev_dir()), std::nullopt);
    EXPECT_EQ(pcie::device_node_for_peer((tree.dev_dir() / "hololink_pcie0").string(), tree.sysfs_class(),
                  tree.dev_dir()),
        std::optional<std::string>((tree.dev_dir() / "hololink_pcie0").string()));
}

TEST(PcieEnumerator, ScanFallsBackToDeviceLinkForBdf)
{
    // Older driver: no `bdf` attribute and no identity attributes -- BDF must come from the
    // resolved `device` link; identity stays all-zero (no /dev node either).
    // No hololink_pcie/ child under the PCI device either, so the resolvers take the
    // class-walk fallback (bdf attribute absent -> device link).
    FakeTree tree;
    const std::filesystem::path device_dir = tree.add_pci_device("pci0000:00",
        { "0000:00:01.0", "0000:01:00.0" });
    tree.add_class_device(2, device_dir, /*with_bdf_attr=*/false, "", "", /*with_pci_child=*/false);

    const auto devices = pcie::scan_pcie_devices(tree.sysfs_class(), tree.dev_dir());
    ASSERT_EQ(devices.size(), 1u);
    EXPECT_EQ(devices[0].minor, 2);
    EXPECT_EQ(devices[0].bdf, "0000:01:00.0");
    EXPECT_EQ(devices[0].root_complex, "pci0000:00");
    EXPECT_EQ(devices[0].requester_id, 0x0100u); // derived from the BDF
    EXPECT_EQ(devices[0].uuid, pcie::Uuid {});
    EXPECT_EQ(devices[0].flags, 0u);
    EXPECT_EQ(pcie::minor_for_bdf("0000:01:00.0", tree.sysfs_class()), std::optional<uint16_t>(2));
    EXPECT_EQ(pcie::device_node_for_bdf("0000:01:00.0", tree.sysfs_class(), tree.dev_dir()),
        std::optional<std::string>((tree.dev_dir() / "hololink_pcie2").string()));
    EXPECT_EQ(pcie::bdf_for_peer((tree.dev_dir() / "hololink_pcie2").string(), tree.sysfs_class(), tree.dev_dir()),
        std::optional<std::string>("0000:01:00.0"));
}

TEST(PcieEnumerator, MinorForBdfPrefersThePciDeviceChildDirectory)
{
    // Only the fast path can answer here: the class entry has no bdf attribute and a dangling
    // device link, but the PCI device directory carries hololink_pcie/hololink_pcie5.
    FakeTree tree;
    const std::filesystem::path device_dir = tree.add_pci_device("pci0007:00",
        { "0007:00:00.0", "0007:03:00.0" });
    std::filesystem::create_directories(device_dir / "hololink_pcie" / "hololink_pcie5");
    const std::filesystem::path entry = tree.sysfs_class() / "hololink_pcie5";
    std::filesystem::create_directories(entry);
    std::filesystem::create_directory_symlink(tree.root() / "sys" / "devices" / "missing", entry / "device");

    EXPECT_EQ(pcie::minor_for_bdf("0007:03:00.0", tree.sysfs_class()), std::optional<uint16_t>(5));
    EXPECT_EQ(pcie::device_node_for_bdf("0007:03:00.0", tree.sysfs_class(), tree.dev_dir()),
        std::optional<std::string>((tree.dev_dir() / "hololink_pcie5").string()));
    // The class walk alone cannot: the entry's BDF is unknowable.
    EXPECT_EQ(pcie::bdf_for_minor(5, tree.sysfs_class()), std::nullopt);
    // And the bus view without a hololink_pcie child (unbound function) resolves nothing.
    tree.add_pci_device("pci0007:00", { "0007:00:00.0", "0007:04:00.0" });
    EXPECT_EQ(pcie::minor_for_bdf("0007:04:00.0", tree.sysfs_class()), std::nullopt);
}

TEST(PcieEnumerator, ResolvesTheByBdfUdevAlias)
{
    // /dev/hololink/by-bdf/<bdf> -> hololink_pcieN, as the optional udev rule creates it.
    FakeTree tree;
    const std::filesystem::path device_dir = tree.add_pci_device("pci0005:00",
        { "0005:00:00.0", "0005:06:00.0" });
    tree.add_class_device(1, device_dir, /*with_bdf_attr=*/true, "0005:06:00.0", "");
    write_file(tree.dev_dir() / "hololink_pcie1", ""); // stands in for the node
    const std::filesystem::path alias_dir = tree.dev_dir() / "hololink" / "by-bdf";
    std::filesystem::create_directories(alias_dir);
    std::filesystem::create_symlink("../../hololink_pcie1", alias_dir / "0005:06:00.0");

    const std::string alias = (alias_dir / "0005:06:00.0").string();
    EXPECT_EQ(pcie::bdf_for_peer(alias, tree.sysfs_class(), tree.dev_dir()),
        std::optional<std::string>("0005:06:00.0"));
    EXPECT_EQ(pcie::device_node_for_peer(alias, tree.sysfs_class(), tree.dev_dir()),
        std::optional<std::string>((tree.dev_dir() / "hololink_pcie1").string()));
    // A dangling alias still yields the BDF its name carries.
    std::filesystem::create_symlink("../../hololink_pcie7", alias_dir / "0005:07:00.0");
    EXPECT_EQ(pcie::bdf_for_peer((alias_dir / "0005:07:00.0").string(), tree.sysfs_class(), tree.dev_dir()),
        std::optional<std::string>("0005:07:00.0"));
    // The bare predicate.
    EXPECT_TRUE(pcie::is_bdf("0005:06:00.0"));
    EXPECT_TRUE(pcie::is_bdf("FFFF:FF:1F.7"));
    EXPECT_FALSE(pcie::is_bdf("0005:06:00.8"));
    EXPECT_FALSE(pcie::is_bdf("05:06:00.0"));
    EXPECT_FALSE(pcie::is_bdf("/dev/hololink_pcie0"));
}

TEST(PcieEnumerator, ScanSkipsBrokenEntriesWithoutThrowing)
{
    FakeTree tree;
    const std::filesystem::path good_dir = tree.add_pci_device("pci0005:00",
        { "0005:00:00.0", "0005:06:00.0" });
    tree.add_class_device(0, good_dir, true, "0005:06:00.0", pcie::HSB_PCIE_MODULE_UUID);

    // hololink_pcie1: dangling `device` link, no bdf attribute -> skipped.
    const std::filesystem::path broken = tree.sysfs_class() / "hololink_pcie1";
    std::filesystem::create_directories(broken);
    std::filesystem::create_directory_symlink(tree.root() / "sys" / "devices" / "missing",
        broken / "device");
    // Something that is not a hololink_pcieN entry at all -> ignored.
    std::filesystem::create_directories(tree.sysfs_class() / "other_device");
    // hololink_pcie7: malformed bdf attribute -> skipped.
    const std::filesystem::path malformed = tree.sysfs_class() / "hololink_pcie7";
    std::filesystem::create_directories(malformed);
    write_file(malformed / "bdf", "not-a-bdf\n");

    std::vector<pcie::PcieDeviceInfo> devices;
    ASSERT_NO_THROW(devices = pcie::scan_pcie_devices(tree.sysfs_class(), tree.dev_dir()));
    ASSERT_EQ(devices.size(), 1u);
    EXPECT_EQ(devices[0].minor, 0);
    EXPECT_EQ(devices[0].bdf, "0005:06:00.0");

    // A missing class directory altogether is simply empty, not an error.
    ASSERT_NO_THROW(devices = pcie::scan_pcie_devices(tree.root() / "does-not-exist", tree.dev_dir()));
    EXPECT_TRUE(devices.empty());
}

TEST(PcieEnumerator, DeviceHelpersFailCleanlyWithoutNode)
{
    FakeTree tree;
    std::string error;
    uint32_t value = 0;
    EXPECT_FALSE(pcie::bar_read32(0, 0x80, value, error, tree.dev_dir()));
    EXPECT_FALSE(error.empty());
    error.clear();
    EXPECT_FALSE(pcie::bar_write32(0, 0x00, 0x1, error, tree.dev_dir()));
    EXPECT_FALSE(error.empty());
    error.clear();
    EXPECT_FALSE(pcie::program_uuid(0, MODULE_UUID_BYTES, error, tree.dev_dir()));
    EXPECT_FALSE(error.empty());
    error.clear();
    pcie::PcieDeviceInfo info;
    EXPECT_FALSE(pcie::read_device_info(0, info, error, tree.sysfs_class(), tree.dev_dir()));
    EXPECT_FALSE(error.empty());
}
