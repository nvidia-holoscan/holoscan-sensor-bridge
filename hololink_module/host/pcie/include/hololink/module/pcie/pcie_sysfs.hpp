/**
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

#ifndef HOLOLINK_MODULE_PCIE_PCIE_SYSFS_HPP
#define HOLOLINK_MODULE_PCIE_PCIE_SYSFS_HPP

/**
 * @file
 * @brief Header-only sysfs helpers mapping a Hololink PCIe endpoint's PCI function address
 * ("DDDD:BB:DD.F" -- the `peer_ip` of a PCIe channel) to the /dev/hololink_pcieN node the
 * hololink_pcie driver bound to it, and back.
 *
 * Shared by three sides that cannot share a library: the host-side enumerator
 * (hololink_module/host/src/pcie_enumerator.cpp, in hololink::module), the hsb_pcie module
 * (.so -- it links only module_headers / module_runtime / module_core, never the host library,
 * whose Adapter singleton must exist once per process) and the module-native receive path
 * (hololink::module::pcie_receiver, core-free). Standard library only: no fmt, no HSB_LOG.
 *
 * Why the BDF is the key: the FPGA function's BAR0 is the control-plane destination, exactly
 * as an Ethernet board's IP is; /dev/hololink_pcieN minors follow probe order and change
 * when boards are added or the driver reloads. The node is derived from the BDF at use time.
 *
 * Resolution order in minor_for_bdf(): the class device is a child of the PCI device
 * (`device_create(class, &pdev->dev, ...)`), so /sys/bus/pci/devices/<bdf>/hololink_pcie/
 * hololink_pcieN names the minor directly; when that directory is absent (older driver, or a
 * tree without the bus view) the class is walked and the `bdf` attribute, then the `device`
 * link, are matched. Every function takes the sysfs / dev roots as parameters so tests can
 * point at a fake tree.
 */

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>

#include "hololink/module/pcie/hololink_pcie_uapi.h"

namespace hololink::module::pcie {

/// Default sysfs class directory the kernel driver populates with hololink_pcieN entries.
constexpr const char* DEFAULT_SYSFS_CLASS = "/sys/class/hololink_pcie";
/// Default directory holding the /dev/hololink_pcieN character device nodes.
constexpr const char* DEFAULT_DEV_DIR = "/dev";
/// Class / character-device name the driver registers ("hololink_pcie").
constexpr const char* PCIE_DEVICE_NAME = HL_PCIE_DEVICE_NAME;
/// Optional udev alias directory below the dev dir: /dev/hololink/by-bdf/<bdf> -> hololink_pcieN.
constexpr const char* BY_BDF_ALIAS_SUBDIR = "hololink/by-bdf";
/// The `mac_id` the PCIe enumerator stamps. The PCI function is the address; nothing reads this.
constexpr const char* PCIE_NULL_MAC_ID = "00:00:00:00:00:00";

/// True when `text` is a PCI function address: ^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$
inline bool is_bdf(const std::string& text)
{
    if (text.size() != 12 || text[4] != ':' || text[7] != ':' || text[10] != '.') {
        return false;
    }
    const auto hex = [&text](size_t begin, size_t count) {
        for (size_t i = begin; i < begin + count; i++) {
            if (std::isxdigit(static_cast<unsigned char>(text[i])) == 0) {
                return false;
            }
        }
        return true;
    };
    return hex(0, 4) && hex(5, 2) && hex(8, 2) && text[11] >= '0' && text[11] <= '7';
}

/// Lowercase copy, the form sysfs and the enumerator use.
inline std::string normalize_bdf(std::string bdf)
{
    std::transform(bdf.begin(), bdf.end(), bdf.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return bdf;
}

/// "<dev_dir>/hololink_pcieN".
inline std::string device_node_path(const std::filesystem::path& dev_dir, uint16_t minor)
{
    return (dev_dir / (std::string(PCIE_DEVICE_NAME) + std::to_string(minor))).string();
}

/// Minor from a class-device (or node) name "hololink_pcieN"; std::nullopt for anything else.
inline std::optional<uint16_t> minor_from_class_name(const std::string& name)
{
    const std::string prefix = PCIE_DEVICE_NAME;
    if (name.size() <= prefix.size() || name.compare(0, prefix.size(), prefix) != 0) {
        return std::nullopt;
    }
    const std::string digits = name.substr(prefix.size());
    if (!std::all_of(digits.begin(), digits.end(),
            [](unsigned char c) { return std::isdigit(c) != 0; })) {
        return std::nullopt;
    }
    const unsigned long value = std::strtoul(digits.c_str(), nullptr, 10);
    if (value > 0xFFFFul) {
        return std::nullopt;
    }
    return static_cast<uint16_t>(value);
}

/// Minor when `path` is exactly "<dev_dir>/hololink_pcieN"; std::nullopt for any other form.
inline std::optional<uint16_t> minor_from_device_node(const std::string& path,
    const std::filesystem::path& dev_dir = DEFAULT_DEV_DIR)
{
    const std::filesystem::path p(path);
    if (p.parent_path().lexically_normal() != dev_dir.lexically_normal()) {
        return std::nullopt;
    }
    return minor_from_class_name(p.filename().string());
}

/// First line of a sysfs attribute, trailing whitespace stripped; std::nullopt when unreadable.
inline std::optional<std::string> read_sysfs_attribute(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        return std::nullopt;
    }
    std::string line;
    std::getline(in, line);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' ')) {
        line.pop_back();
    }
    return line;
}

/// The PCI bus view that pairs with `sysfs_class`: "/sys/class/hololink_pcie" ->
/// "/sys/bus/pci/devices". A fake tree rooted elsewhere follows the same shape.
inline std::filesystem::path sysfs_pci_devices_dir(const std::filesystem::path& sysfs_class)
{
    return sysfs_class.parent_path().parent_path() / "bus" / "pci" / "devices";
}

/// BDF of the class device at `entry` (lowercase): the driver's `bdf` attribute, else the
/// basename of the resolved `device` link. std::nullopt when neither is available.
inline std::optional<std::string> bdf_for_class_device(const std::filesystem::path& entry)
{
    if (const auto attr = read_sysfs_attribute(entry / "bdf"); attr && !attr->empty()) {
        return normalize_bdf(*attr);
    }
    std::error_code ec;
    const std::filesystem::path resolved = std::filesystem::canonical(entry / "device", ec);
    if (ec) {
        return std::nullopt;
    }
    return normalize_bdf(resolved.filename().string());
}

/// BDF of the endpoint bound at `minor`, from its class device.
inline std::optional<std::string> bdf_for_minor(uint16_t minor,
    const std::filesystem::path& sysfs_class = DEFAULT_SYSFS_CLASS)
{
    return bdf_for_class_device(sysfs_class / (std::string(PCIE_DEVICE_NAME) + std::to_string(minor)));
}

/**
 * @brief Resolve a BDF ("DDDD:BB:DD.F", case-insensitive) to its hololink_pcieN minor.
 *
 * Fast path: /sys/bus/pci/devices/<bdf>/hololink_pcie/hololink_pcieN (the class device is a
 * child of the PCI device). Fallback: walk `sysfs_class` matching the `bdf` attribute, then the
 * resolved `device` link.
 * @return std::nullopt when no class device carries that BDF (or `bdf` is malformed).
 */
inline std::optional<uint16_t> minor_for_bdf(const std::string& bdf,
    const std::filesystem::path& sysfs_class = DEFAULT_SYSFS_CLASS)
{
    if (!is_bdf(bdf)) {
        return std::nullopt;
    }
    const std::string wanted = normalize_bdf(bdf);

    std::error_code ec;
    const std::filesystem::path by_device = sysfs_pci_devices_dir(sysfs_class) / wanted / PCIE_DEVICE_NAME;
    std::filesystem::directory_iterator children(by_device, ec);
    if (!ec) {
        for (const auto& child : children) {
            if (const auto minor = minor_from_class_name(child.path().filename().string())) {
                return minor;
            }
        }
    }

    std::filesystem::directory_iterator entries(sysfs_class, ec);
    if (ec) {
        return std::nullopt;
    }
    for (const auto& entry : entries) {
        const auto minor = minor_from_class_name(entry.path().filename().string());
        if (!minor) {
            continue;
        }
        if (const auto have = bdf_for_class_device(entry.path()); have && *have == wanted) {
            return minor;
        }
    }
    return std::nullopt;
}

/// minor_for_bdf() rendered as the node path "<dev_dir>/hololink_pcieN".
inline std::optional<std::string> device_node_for_bdf(const std::string& bdf,
    const std::filesystem::path& sysfs_class = DEFAULT_SYSFS_CLASS,
    const std::filesystem::path& dev_dir = DEFAULT_DEV_DIR)
{
    const auto minor = minor_for_bdf(bdf, sysfs_class);
    if (!minor) {
        return std::nullopt;
    }
    return device_node_path(dev_dir, *minor);
}

/**
 * @brief Resolve any accepted spelling of a PCIe peer to its BDF (lowercase).
 *
 *  - "DDDD:BB:DD.F": itself (not checked against sysfs; minor_for_bdf() does that).
 *  - "<dev_dir>/hololink_pcieN": the class device's `bdf` attribute, else its `device` link.
 *  - "<dev_dir>/hololink/by-bdf/<bdf>" (udev alias): the node the link resolves to, as above;
 *    the alias basename when the link cannot be followed.
 * @return std::nullopt when `peer` is none of these or the class device is missing.
 */
inline std::optional<std::string> bdf_for_peer(const std::string& peer,
    const std::filesystem::path& sysfs_class = DEFAULT_SYSFS_CLASS,
    const std::filesystem::path& dev_dir = DEFAULT_DEV_DIR)
{
    if (is_bdf(peer)) {
        return normalize_bdf(peer);
    }
    if (const auto minor = minor_from_device_node(peer, dev_dir)) {
        return bdf_for_minor(*minor, sysfs_class);
    }
    const std::filesystem::path p(peer);
    if (p.parent_path().lexically_normal() == (dev_dir / BY_BDF_ALIAS_SUBDIR).lexically_normal()) {
        std::error_code ec;
        const std::filesystem::path target = std::filesystem::canonical(p, ec);
        if (!ec) {
            if (const auto minor = minor_from_class_name(target.filename().string())) {
                if (const auto bdf = bdf_for_minor(*minor, sysfs_class)) {
                    return bdf;
                }
            }
        }
        if (is_bdf(p.filename().string())) {
            return normalize_bdf(p.filename().string());
        }
    }
    return std::nullopt;
}

/**
 * @brief Parse a dotted-quad IPv4 string into its 32-bit big-endian value; false if malformed.
 *
 * The kernel driver's bootp announcement carries the BDF in the bootp client address, which the
 * host parser renders as a dotted quad ("0.5.1.0" for 0005:01:00.0); this reverses the rendering.
 */
inline bool dotted_quad_to_u32(const std::string& text, uint32_t& value)
{
    uint32_t result = 0;
    unsigned groups = 0;
    size_t i = 0;
    while (i <= text.size()) {
        size_t j = i;
        while (j < text.size() && std::isdigit(static_cast<unsigned char>(text[j]))) {
            j++;
        }
        if (j == i || j - i > 3 || (j < text.size() && text[j] != '.')) {
            return false;
        }
        const unsigned long octet = std::strtoul(text.substr(i, j - i).c_str(), nullptr, 10);
        if (octet > 255) {
            return false;
        }
        result = (result << 8) | static_cast<uint32_t>(octet);
        groups++;
        i = j + 1;
    }
    if (groups != 4) {
        return false;
    }
    value = result;
    return true;
}

/// The 32-bit BDF encoding the kernel driver uses: (domain << 16) | (bus << 8) | devfn.
inline std::string bdf32_to_string(uint32_t bdf32)
{
    char buffer[16];
    std::snprintf(buffer, sizeof(buffer), "%04x:%02x:%02x.%u", (bdf32 >> 16) & 0xFFFFu,
        (bdf32 >> 8) & 0xFFu, (bdf32 >> 3) & 0x1Fu, bdf32 & 0x7u);
    return buffer;
}

/// Inverse of bdf32_to_string(): "DDDD:BB:DD.F" (any case) -> (domain << 16) | (bus << 8) | devfn.
/// False when `text` is not a PCI function address (is_bdf()).
inline bool bdf_string_to_u32(const std::string& text, uint32_t& bdf32)
{
    if (!is_bdf(text)) {
        return false;
    }
    const uint32_t domain = static_cast<uint32_t>(std::stoul(text.substr(0, 4), nullptr, 16));
    const uint32_t bus = static_cast<uint32_t>(std::stoul(text.substr(5, 2), nullptr, 16));
    const uint32_t device = static_cast<uint32_t>(std::stoul(text.substr(8, 2), nullptr, 16));
    const uint32_t function = static_cast<uint32_t>(text[11] - '0');
    if (device > 0x1Fu) {
        return false;
    }
    bdf32 = (domain << 16) | (bus << 8) | (device << 3) | function;
    return true;
}

/// bdf_for_peer() followed by device_node_for_bdf(): the node for any accepted peer spelling.
inline std::optional<std::string> device_node_for_peer(const std::string& peer,
    const std::filesystem::path& sysfs_class = DEFAULT_SYSFS_CLASS,
    const std::filesystem::path& dev_dir = DEFAULT_DEV_DIR)
{
    const auto bdf = bdf_for_peer(peer, sysfs_class, dev_dir);
    if (!bdf) {
        return std::nullopt;
    }
    return device_node_for_bdf(*bdf, sysfs_class, dev_dir);
}

/**
 * @brief One-line diagnosis of a failed open() of a /dev/hololink_pcieN node.
 *
 * The node is created by the driver as 0600 root:root (device_create() with no devnode
 * callback), so by far the most common failure for a non-root process is EACCES -- which
 * has nothing to do with whether the driver is loaded. Name the actual cause and the fix.
 */
inline std::string device_node_open_error(const std::string& path, int err)
{
    std::string text = "open " + path + " failed: " + std::strerror(err);
    switch (err) {
    case EACCES:
    case EPERM:
        text += ". The driver creates the node 0600 root:root; either run as root or install "
                "the udev rule (see src/ehsb/kmod/README.md): KERNEL==\"hololink_pcie[0-9]*\", "
                "MODE=\"0660\", GROUP=\"hololink\" -- then add yourself to that group and "
                "re-login.";
        break;
    case ENOENT:
        text += ". No such node: load the hololink_pcie driver (build/ehsb/kmod/"
                "hololink_pcie_drv.ko) and make sure the endpoint is bound to it "
                "(/sys/bus/pci/devices/<bdf>/driver).";
        break;
    case ENXIO:
    case ENODEV:
        text += ". The node exists but no device answers behind it: the driver was unloaded "
                "or the endpoint removed since the node was created.";
        break;
    default:
        break;
    }
    return text;
}

} // namespace hololink::module::pcie

#endif /* HOLOLINK_MODULE_PCIE_PCIE_SYSFS_HPP */
