/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standardized enumeration of Hololink PCIe endpoints for the hololink_module framework.
 *
 * Hololink boards on Ethernet announce themselves over bootp; a PCIe endpoint is announced
 * the same way by the hololink_pcie kernel driver (htype 0x50, chaddr = the BDF), and the
 * hsb_pcie module's update_metadata completes the parser's transport-agnostic record (the BDF
 * from the 4-byte hardware address, the vendor tail) into the EnumerationMetadata the
 * framework expects. This header
 * holds the sysfs-side helpers around it: the metadata is keyed on the PCI function --
 * `peer_ip` is the BDF ("DDDD:BB:DD.F" -- the FPGA function whose BAR0 is the control-plane
 * destination, the PCIe counterpart of an Ethernet board's IP), `mac_id` is the same BDF
 * text (the hardware address in its htype's convention), `interface` is `lo`, and the
 * module is selected through `fpga_uuid` exactly like any other board. The character device
 * node (/dev/hololink_pcieN) is derived from the BDF through sysfs at use time
 * (pcie/pcie_sysfs.hpp) and also stamped as `pcie_device_node` for reference; minors follow
 * probe order, so they are not the key.
 *
 * Identity comes from the kernel driver in order of preference: the HL_PCIE_IOC_IDENTITY
 * ioctl, the read-only sysfs attributes the driver exports, and finally raw BAR0 reads of
 * the identity block through HL_PCIE_IOC_READ (oldest drivers).
 *
 * This header is host-side and core-free: it depends only on the module headers, the uapi
 * copy of hololink_pcie_uapi.h, pcie_regs.hpp and the header-only BDF <-> node resolvers in
 * pcie/pcie_sysfs.hpp (shared with the hsb_pcie module .so, which cannot link this library).
 * Ports the identity / UUID handling of
 * hsb_pcie/driver/src/hsb_pcie/msix_pcie.{hpp,cpp} (read_uuid / program_uuid / UUID_REGS)
 * to the enumeration side.
 */

#ifndef HOLOLINK_MODULE_PCIE_ENUMERATOR_HPP
#define HOLOLINK_MODULE_PCIE_ENUMERATOR_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "hololink/module/enumeration_metadata.hpp"
// DEFAULT_SYSFS_CLASS / DEFAULT_DEV_DIR, is_bdf(), minor_for_bdf(), device_node_for_bdf(),
// bdf_for_peer(), device_node_for_peer(): header-only so the module .so shares them.
#include "hololink/module/pcie/pcie_sysfs.hpp"

namespace hololink::module::pcie {

/// The UUID the hsb_pcie endpoint bitfile reports in BAR0 HSB_UUID_3..0. The FPGA holds it
/// read-only, so it is the board identity the framework keys on; it selects the
/// hololink_58c1419b-1e7a-4120-95fd-4efc156187c1.so module.
constexpr const char* HSB_PCIE_MODULE_UUID = "58c1419b-1e7a-4120-95fd-4efc156187c1";

/// 128-bit board UUID, byte order as in hl_pcie_identity_info::uuid (see uuid_to_text()).
using Uuid = std::array<uint8_t, 16>;

/**
 * @brief Everything the enumerator learns about one /dev/hololink_pcieN.
 *
 * Fields the source could not supply stay at their defaults (0 / empty); `flags` carries the
 * HL_PCIE_ID_F_* bits, synthesized from the values themselves when they did not come from
 * the IDENTITY ioctl.
 */
struct PcieDeviceInfo {
    uint16_t minor { 0 }; ///< N in hololink_pcieN (`pcie_minor`; follows probe order)
    std::string device_node; ///< "/dev/hololink_pcieN" (`pcie_device_node`; derived from the BDF)
    std::string bdf; ///< "DDDD:BB:DD.F" lowercase -- the channel key (`peer_ip`, `pcie_bdf`)
    uint16_t domain { 0 }; ///< PCI segment
    uint8_t bus { 0 };
    uint8_t device { 0 };
    uint8_t function { 0 };
    uint32_t requester_id { 0 }; ///< (bus << 8) | devfn, as programmed into PCIE_DP
    std::string root_complex; ///< sysfs root-bus id, e.g. "pci0005:00" (`interface`)
    std::string sysfs_path; ///< realpath of the class device's `device` link (`interface_address`)
    uint32_t hsb_version { 0 }; ///< BAR0 HSB_VER
    uint32_t hsb_date { 0 }; ///< BAR0 HSB_DATE
    uint32_t hsb_status { 0 }; ///< BAR0 HSB_STAT
    uint32_t flags { 0 }; ///< HL_PCIE_ID_F_* bits
    uint32_t msix_vectors { 0 }; ///< MSI-X vectors allocated on the device
    uint32_t dma_address_bits { 0 }; ///< DMA mask width the driver was loaded with
    uint64_t bar0_len { 0 }; ///< BAR0 length in bytes
    Uuid uuid {}; ///< board UUID, driver byte order
    std::string uuid_text; ///< canonical lowercase hyphenated rendering of `uuid`
    /// False when the endpoint's identity could not be read as a live board: BAR0 reads
    /// came back all-ones (PCI memory decode disabled -- `lspci` shows `Control: Mem-` --
    /// or the link is down) or no identity source answered at all. Such an endpoint is
    /// still announced so listings show it, with `pcie_identity_valid = 0` and
    /// `pcie_identity_error` explaining why; no module can match its UUID.
    bool responding { false };
    std::string identity_error; ///< empty when `responding`
    /// Why the enumerator could not open `device_node` (`pcie_node_error`); empty when it
    /// could. Typically EACCES for a non-root process without the udev rule: the identity
    /// then comes from the world-readable sysfs attributes, but no application can use the
    /// endpoint until the node is accessible.
    std::string node_error;
};

/**
 * @brief Render a UUID as canonical 8-4-4-4-12 lowercase hex.
 *
 * Bytes are printed in array order: uuid[0..3] is HSB_UUID_3 (BAR0 0xB8) MSB-first,
 * uuid[4..7] HSB_UUID_2, uuid[8..11] HSB_UUID_1, uuid[12..15] HSB_UUID_0 -- the order the
 * driver's hl_pcie_identity_info::uuid and hsb_pcie's msix_pcie.hpp UUID_REGS use, so the
 * text matches identity.py and the `uuid` sysfs attribute.
 */
std::string uuid_to_text(const Uuid& uuid);

/**
 * @brief Parse canonical (hyphenated, any case) or bare 32-digit hex UUID text.
 * @return false (and `uuid` untouched) when `text` is malformed.
 */
bool uuid_from_text(const std::string& text, Uuid& uuid);

/// True when `uuid` is neither all-zero nor all-ones (the driver's UUID_VALID rule).
bool uuid_is_valid(const Uuid& uuid);

/**
 * @brief Walk `sysfs_class` and describe every hololink_pcieN endpoint found.
 *
 * The BDF comes from the `bdf` attribute when the driver exports it, else from the basename
 * of the resolved `device` link; the root complex is the "pciDDDD:BB" component of that
 * resolved path. Identity is read through HL_PCIE_IOC_IDENTITY on `dev_dir`/hololink_pcieN
 * when the node can be opened, else from the sysfs attributes, else from BAR0 via
 * HL_PCIE_IOC_READ. A device that cannot be described is skipped with a warning; this never
 * throws for a single bad entry. Both roots are parameters so tests can point at a fake tree.
 */
std::vector<PcieDeviceInfo> scan_pcie_devices(
    const std::filesystem::path& sysfs_class = DEFAULT_SYSFS_CLASS,
    const std::filesystem::path& dev_dir = DEFAULT_DEV_DIR);

/*
 * BDF <-> node resolution -- minor_for_bdf(bdf, sysfs_class), device_node_for_bdf(bdf,
 * sysfs_class, dev_dir), bdf_for_peer(peer, ...) and device_node_for_peer(peer, ...) -- is
 * declared and defined in pcie/pcie_sysfs.hpp (included above). bdf_for_peer() also accepts,
 * for convenience, a peer that is already "/dev/hololink_pcieN" or a "/dev/hololink/by-bdf/
 * <bdf>" udev alias, resolving it to the BDF through the class device's `bdf` attribute.
 */

/**
 * @brief Describe one device by minor, opening `dev_dir`/hololink_pcieN.
 * @return false with `error` filled when the node cannot be opened or identity cannot be read.
 */
bool read_device_info(uint16_t minor, PcieDeviceInfo& info, std::string& error,
    const std::filesystem::path& sysfs_class = DEFAULT_SYSFS_CLASS,
    const std::filesystem::path& dev_dir = DEFAULT_DEV_DIR);

/// Single aligned 32-bit BAR0 read through HL_PCIE_IOC_READ on `dev_dir`/hololink_pcieN.
bool bar_read32(uint16_t minor, uint32_t offset, uint32_t& value, std::string& error,
    const std::filesystem::path& dev_dir = DEFAULT_DEV_DIR);

/// Single aligned 32-bit BAR0 write through HL_PCIE_IOC_WRITE on `dev_dir`/hololink_pcieN.
bool bar_write32(uint16_t minor, uint32_t offset, uint32_t value, std::string& error,
    const std::filesystem::path& dev_dir = DEFAULT_DEV_DIR);

/// Read the board UUID from BAR0 HSB_UUID_3..0 (0xB8, 0xB4, 0xB0, 0xAC) via HL_PCIE_IOC_READ.
bool read_uuid(uint16_t minor, Uuid& uuid, std::string& error,
    const std::filesystem::path& dev_dir = DEFAULT_DEV_DIR);

/**
 * @brief Program the board UUID into BAR0 and verify the readback.
 *
 * Writes HSB_UUID_3, HSB_UUID_2, HSB_UUID_1, HSB_UUID_0 (0xB8, 0xB4, 0xB0, 0xAC) MSW-first
 * as big-endian words -- the semantics of MSIXPCIE::program_uuid in hsb_pcie's
 * msix_pcie.cpp -- then reads the four registers back and compares.
 *
 * Temporary provisioning aid until the FPGA advertises the module UUID by itself.
 *
 * @return false with `error` filled on any ioctl failure or readback mismatch.
 */
bool program_uuid(uint16_t minor, const Uuid& uuid, std::string& error,
    const std::filesystem::path& dev_dir = DEFAULT_DEV_DIR);

} // namespace hololink::module::pcie

#endif // HOLOLINK_MODULE_PCIE_ENUMERATOR_HPP
