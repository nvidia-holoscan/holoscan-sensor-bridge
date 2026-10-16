/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standardized PCIe enumeration for hololink_module: sysfs class walk, identity via the
 * driver (IDENTITY ioctl / sysfs attributes / BAR0 reads), the section-F metadata table
 * (keyed on the BDF: peer_ip = "DDDD:BB:DD.F"), and the UUID text + provisioning helpers.
 * The BDF <-> /dev/hololink_pcieN resolvers live header-only in pcie/pcie_sysfs.hpp so the
 * hsb_pcie module .so can share them. Ports the UUID register handling of
 * hsb_pcie/driver/src/hsb_pcie/msix_pcie.cpp (read_uuid / program_uuid, UUID_REGS) onto the
 * core-free host library; every hardware access here is a plain ioctl on the character
 * device, never a hololink::core call.
 */

#include "hololink/module/pcie_enumerator.hpp"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <system_error>

#include <fmt/core.h>

#include "hololink/module/logging.hpp" // HSB_LOG_*
#include "hololink/module/pcie/hololink_pcie_uapi.h"
#include "hololink/module/pcie/pcie_regs.hpp"

namespace hololink::module::pcie {

namespace {

    constexpr const char* DEVICE_NAME = HL_PCIE_DEVICE_NAME; // "hololink_pcie"

    /// BAR0 registers holding the UUID, most-significant word first (msix_pcie.hpp UUID_REGS).
    constexpr uint32_t UUID_REGS[4] = {
        hololink::pcie::HSB_UUID_3, hololink::pcie::HSB_UUID_2,
        hololink::pcie::HSB_UUID_1, hololink::pcie::HSB_UUID_0
    };

    /// Pack uuid[byte_index .. byte_index+3] into one big-endian register word.
    constexpr uint32_t uuid_to_word(const Uuid& uuid, size_t byte_index)
    {
        return (static_cast<uint32_t>(uuid[byte_index]) << 24)
            | (static_cast<uint32_t>(uuid[byte_index + 1]) << 16)
            | (static_cast<uint32_t>(uuid[byte_index + 2]) << 8)
            | static_cast<uint32_t>(uuid[byte_index + 3]);
    }

    /// Unpack one big-endian register word into uuid[byte_index .. byte_index+3].
    void uuid_from_word(uint32_t word, Uuid& uuid, size_t byte_index)
    {
        uuid[byte_index] = static_cast<uint8_t>(word >> 24);
        uuid[byte_index + 1] = static_cast<uint8_t>(word >> 16);
        uuid[byte_index + 2] = static_cast<uint8_t>(word >> 8);
        uuid[byte_index + 3] = static_cast<uint8_t>(word);
    }

    // RAII guard that closes a file descriptor on scope exit.
    class FdGuard {
    public:
        explicit FdGuard(int fd)
            : fd_(fd)
        {
        }
        ~FdGuard()
        {
            if (fd_ >= 0) {
                ::close(fd_);
            }
        }
        FdGuard(const FdGuard&) = delete;
        FdGuard& operator=(const FdGuard&) = delete;

        int get() const { return fd_; }
        explicit operator bool() const { return fd_ >= 0; }

    private:
        int fd_;
    };

    /// Open the device node read-write first (BAR writes need it), read-only as a fallback.
    int open_device(const std::filesystem::path& dev_dir, uint16_t minor, bool writable,
        std::string& error)
    {
        const std::string path = device_node_path(dev_dir, minor);
        int fd = ::open(path.c_str(), (writable ? O_RDWR : O_RDONLY) | O_CLOEXEC);
        if (fd < 0) {
            error = fmt::format("open({}) failed: {}", path, std::strerror(errno));
        }
        return fd;
    }

    bool ioctl_read32(int fd, uint32_t offset, uint32_t& value, std::string& error)
    {
        struct hl_pcie_rw rw { };
        rw.bar = hololink::pcie::BAR0;
        rw.offset = offset;
        rw.width = hololink::pcie::REG_SIZE;
        rw.count = 1;
        rw.data = 0;
        if (::ioctl(fd, HL_PCIE_IOC_READ, &rw) < 0) {
            error = fmt::format("HL_PCIE_IOC_READ(0x{:x}) failed: {}", offset, std::strerror(errno));
            return false;
        }
        value = static_cast<uint32_t>(rw.data);
        return true;
    }

    bool ioctl_write32(int fd, uint32_t offset, uint32_t value, std::string& error)
    {
        struct hl_pcie_rw rw { };
        rw.bar = hololink::pcie::BAR0;
        rw.offset = offset;
        rw.width = hololink::pcie::REG_SIZE;
        rw.count = 1;
        rw.data = value;
        if (::ioctl(fd, HL_PCIE_IOC_WRITE, &rw) < 0) {
            error = fmt::format("HL_PCIE_IOC_WRITE(0x{:x}) failed: {}", offset, std::strerror(errno));
            return false;
        }
        return true;
    }

    bool ioctl_read_uuid(int fd, Uuid& uuid, std::string& error)
    {
        Uuid tmp {};
        for (size_t i = 0; i < 4; i++) {
            uint32_t word = 0;
            if (!ioctl_read32(fd, UUID_REGS[i], word, error)) {
                return false;
            }
            uuid_from_word(word, tmp, i * 4);
        }
        uuid = tmp;
        return true;
    }

    /// Parse a sysfs integer attribute ("0x2606", "64", ...); base auto-detected.
    std::optional<uint64_t> read_uint_attribute(const std::filesystem::path& path)
    {
        const auto text = read_sysfs_attribute(path);
        if (!text || text->empty()) {
            return std::nullopt;
        }
        char* end = nullptr;
        errno = 0;
        const unsigned long long value = std::strtoull(text->c_str(), &end, 0);
        if (errno != 0 || end == text->c_str()) {
            return std::nullopt;
        }
        return static_cast<uint64_t>(value);
    }

    bool is_hex(const std::string& s, size_t begin, size_t count)
    {
        if (begin + count > s.size()) {
            return false;
        }
        for (size_t i = begin; i < begin + count; i++) {
            if (std::isxdigit(static_cast<unsigned char>(s[i])) == 0) {
                return false;
            }
        }
        return true;
    }

    /// Parse "DDDD:BB:DD.F" (case-insensitive) into its fields; false when malformed.
    bool parse_bdf(const std::string& bdf, uint16_t& domain, uint8_t& bus, uint8_t& device,
        uint8_t& function)
    {
        // 0005:06:00.0
        if (bdf.size() != 12 || bdf[4] != ':' || bdf[7] != ':' || bdf[10] != '.') {
            return false;
        }
        if (!is_hex(bdf, 0, 4) || !is_hex(bdf, 5, 2) || !is_hex(bdf, 8, 2) || !is_hex(bdf, 11, 1)) {
            return false;
        }
        domain = static_cast<uint16_t>(std::strtoul(bdf.substr(0, 4).c_str(), nullptr, 16));
        bus = static_cast<uint8_t>(std::strtoul(bdf.substr(5, 2).c_str(), nullptr, 16));
        const unsigned long dev = std::strtoul(bdf.substr(8, 2).c_str(), nullptr, 16);
        const unsigned long fn = std::strtoul(bdf.substr(11, 1).c_str(), nullptr, 16);
        if (dev > 0x1f || fn > 7) {
            return false;
        }
        device = static_cast<uint8_t>(dev);
        function = static_cast<uint8_t>(fn);
        return true;
    }

    /// "pciDDDD:BB" -- the sysfs root-bus directory name under /sys/devices.
    bool is_root_bus_component(const std::string& component)
    {
        return component.size() == 10 && component.compare(0, 3, "pci") == 0
            && is_hex(component, 3, 4) && component[7] == ':' && is_hex(component, 8, 2);
    }

    std::string root_complex_of(const std::filesystem::path& sysfs_path)
    {
        for (const auto& component : sysfs_path) {
            const std::string text = component.string();
            if (is_root_bus_component(text)) {
                return text;
            }
        }
        return {};
    }

    /// Resolve `entry`/device to the PCI device directory; empty when the link is missing.
    std::filesystem::path resolve_device_link(const std::filesystem::path& entry)
    {
        std::error_code ec;
        const std::filesystem::path resolved = std::filesystem::canonical(entry / "device", ec);
        if (ec) {
            return {};
        }
        return resolved;
    }

    void fill_from_ioctl(const struct hl_pcie_identity_info& id, PcieDeviceInfo& info)
    {
        info.domain = id.domain;
        info.bus = id.bus;
        info.device = static_cast<uint8_t>(id.devfn >> 3);
        info.function = static_cast<uint8_t>(id.devfn & 0x7);
        info.requester_id = id.requester_id;
        info.hsb_version = id.hsb_version;
        info.hsb_date = id.hsb_date;
        info.hsb_status = id.hsb_status;
        info.flags = id.flags;
        info.msix_vectors = id.num_msix_vectors;
        info.dma_address_bits = id.dma_address_bits;
        info.bar0_len = id.bar0_len;
        std::memcpy(info.uuid.data(), id.uuid, info.uuid.size());
        if (info.bdf.empty()) {
            info.bdf = fmt::format("{:04x}:{:02x}:{:02x}.{}", id.domain, id.bus,
                id.devfn >> 3, id.devfn & 0x7);
        }
    }

    /// Identity through HL_PCIE_IOC_IDENTITY; false (error set) when the driver lacks it.
    bool identity_from_ioctl(int fd, PcieDeviceInfo& info, std::string& error)
    {
        struct hl_pcie_identity_info id { };
        if (::ioctl(fd, HL_PCIE_IOC_IDENTITY, &id) < 0) {
            error = fmt::format("HL_PCIE_IOC_IDENTITY failed: {}", std::strerror(errno));
            return false;
        }
        fill_from_ioctl(id, info);
        return true;
    }

    /// Identity from the driver's sysfs attributes; false when `uuid` (the key one) is absent.
    bool identity_from_sysfs(const std::filesystem::path& entry, PcieDeviceInfo& info)
    {
        const auto uuid_text = read_sysfs_attribute(entry / "uuid");
        if (!uuid_text) {
            return false;
        }
        Uuid uuid {};
        if (!uuid_from_text(*uuid_text, uuid)) {
            HSB_LOG_WARN("{}: unparsable uuid attribute \"{}\"", entry.string(), *uuid_text);
            return false;
        }
        info.uuid = uuid;
        if (const auto v = read_uint_attribute(entry / "hsb_version")) {
            info.hsb_version = static_cast<uint32_t>(*v);
        }
        if (const auto v = read_uint_attribute(entry / "hsb_date")) {
            info.hsb_date = static_cast<uint32_t>(*v);
        }
        if (const auto v = read_uint_attribute(entry / "requester_id")) {
            info.requester_id = static_cast<uint32_t>(*v);
        }
        if (const auto v = read_uint_attribute(entry / "msix_vectors")) {
            info.msix_vectors = static_cast<uint32_t>(*v);
        }
        return true;
    }

    /// After identity_from_sysfs(): read HSB_STAT (never a sysfs attribute) and HSB_VER /
    /// HSB_DATE when their attributes were absent (left at 0). Best effort; failures are
    /// logged at debug level and leave the fields as they were.
    void fill_sysfs_gaps_from_bar(int fd, PcieDeviceInfo& info)
    {
        std::string error;
        uint32_t value = 0;
        if (ioctl_read32(fd, hololink::pcie::HSB_STAT, value, error)) {
            info.hsb_status = value;
        } else {
            HSB_LOG_DEBUG("{}: {}", info.device_node, error);
        }
        if (info.hsb_version == 0) {
            if (ioctl_read32(fd, hololink::pcie::HSB_VER, value, error)) {
                info.hsb_version = value;
            } else {
                HSB_LOG_DEBUG("{}: {}", info.device_node, error);
            }
        }
        if (info.hsb_date == 0) {
            if (ioctl_read32(fd, hololink::pcie::HSB_DATE, value, error)) {
                info.hsb_date = value;
            } else {
                HSB_LOG_DEBUG("{}: {}", info.device_node, error);
            }
        }
    }

    /// Identity straight from BAR0 (HSB_VER / HSB_DATE / HSB_STAT / HSB_UUID_3..0).
    bool identity_from_bar(int fd, PcieDeviceInfo& info, std::string& error)
    {
        uint32_t version = 0;
        uint32_t date = 0;
        uint32_t status = 0;
        if (!ioctl_read32(fd, hololink::pcie::HSB_VER, version, error)
            || !ioctl_read32(fd, hololink::pcie::HSB_DATE, date, error)
            || !ioctl_read32(fd, hololink::pcie::HSB_STAT, status, error)) {
            return false;
        }
        Uuid uuid {};
        if (!ioctl_read_uuid(fd, uuid, error)) {
            return false;
        }
        info.hsb_version = version;
        info.hsb_date = date;
        info.hsb_status = status;
        info.uuid = uuid;
        return true;
    }

    /// Flags as the driver would compute them, for identities not served by the ioctl.
    uint32_t synthesize_flags(const PcieDeviceInfo& info)
    {
        uint32_t flags = 0;
        if (info.hsb_version != 0 && info.hsb_version != 0xFFFFFFFFu) {
            flags |= HL_PCIE_ID_F_VERSION_VALID;
        }
        if (info.hsb_status == hololink::pcie::HSB_MAGIC_EXPECT) {
            flags |= HL_PCIE_ID_F_MAGIC_VALID;
        }
        if (uuid_is_valid(info.uuid)) {
            flags |= HL_PCIE_ID_F_UUID_VALID;
        }
        return flags;
    }

    /// Derive requester_id from the BDF when nothing supplied it.
    void finish_info(PcieDeviceInfo& info, bool from_ioctl)
    {
        if (info.requester_id == 0) {
            info.requester_id = (static_cast<uint32_t>(info.bus) << 8)
                | (static_cast<uint32_t>(info.device) << 3) | info.function;
        }
        if (!from_ioctl) {
            info.flags = synthesize_flags(info);
        }
        info.uuid_text = uuid_to_text(info.uuid);

        // A live endpoint answers with a plausible HSB_VER and a UUID that is not all-ones.
        // All-ones is what a PCI read returns when nothing decodes the address: memory
        // decode disabled after a driver swap, or the link down. The IDENTITY ioctl serves
        // hsb_version from the driver's probe-time cache, so the UUID (re-read live) is the
        // deciding check there.
        const bool uuid_all_ones = std::all_of(info.uuid.begin(), info.uuid.end(),
            [](uint8_t b) { return b == 0xFF; });
        const bool version_dead = info.hsb_version == 0 || info.hsb_version == 0xFFFFFFFFu;
        if (uuid_all_ones || info.hsb_version == 0xFFFFFFFFu) {
            info.responding = false;
            info.identity_error = "BAR0 reads all-ones: PCI memory decode is disabled (lspci "
                                  "shows 'Control: Mem-') or the link is down; reload the "
                                  "hololink_pcie driver";
        } else if (version_dead) {
            info.responding = false;
            info.identity_error = "identity unavailable: no IDENTITY ioctl, sysfs attributes "
                                  "or BAR0 access answered";
        } else {
            info.responding = true;
            info.identity_error.clear();
        }
    }

    /**
     * Describe the class device at `entry`. Returns std::nullopt (after logging) when the
     * entry is not a hololink_pcieN directory or its BDF cannot be determined.
     */
    std::optional<PcieDeviceInfo> describe_entry(const std::filesystem::path& entry,
        const std::filesystem::path& dev_dir)
    {
        const std::string name = entry.filename().string();
        const auto minor = minor_from_class_name(name);
        if (!minor) {
            return std::nullopt;
        }

        PcieDeviceInfo info;
        info.minor = *minor;
        info.device_node = device_node_path(dev_dir, *minor);

        // Where the device sits: the resolved `device` link gives the sysfs path, the root
        // complex and (when the driver has no `bdf` attribute) the BDF itself.
        const std::filesystem::path sysfs_path = resolve_device_link(entry);
        if (!sysfs_path.empty()) {
            info.sysfs_path = sysfs_path.string();
            info.root_complex = root_complex_of(sysfs_path);
        }
        if (const auto bdf = read_sysfs_attribute(entry / "bdf"); bdf && !bdf->empty()) {
            info.bdf = normalize_bdf(*bdf);
        } else if (!sysfs_path.empty()) {
            info.bdf = normalize_bdf(sysfs_path.filename().string());
        }
        if (info.bdf.empty()) {
            HSB_LOG_WARN("{}: no bdf attribute and no resolvable device link; skipping", entry.string());
            return std::nullopt;
        }
        if (!parse_bdf(info.bdf, info.domain, info.bus, info.device, info.function)) {
            HSB_LOG_WARN("{}: malformed BDF \"{}\"; skipping", entry.string(), info.bdf);
            return std::nullopt;
        }

        // Identity: ioctl, then sysfs attributes, then raw BAR0 reads.
        std::string error;
        bool from_ioctl = false;
        bool identified = false;
        FdGuard fd(::open(info.device_node.c_str(), O_RDONLY | O_CLOEXEC));
        if (fd) {
            if (identity_from_ioctl(fd.get(), info, error)) {
                from_ioctl = true;
                identified = true;
            } else {
                HSB_LOG_DEBUG("{}: {}; falling back to sysfs attributes", info.device_node, error);
            }
        } else {
            info.node_error = device_node_open_error(info.device_node, errno);
            HSB_LOG_WARN("{}; identity taken from the sysfs attributes, but the endpoint cannot be "
                         "used from this process until the node is accessible",
                info.node_error);
        }
        if (!identified) {
            identified = identity_from_sysfs(entry, info);
            // The sysfs attributes carry no HSB_STAT (and a driver may lack hsb_version /
            // hsb_date); fill those gaps straight from BAR0 when the node is open.
            if (identified && fd) {
                fill_sysfs_gaps_from_bar(fd.get(), info);
            }
        }
        if (!identified && fd) {
            if (identity_from_bar(fd.get(), info, error)) {
                identified = true;
            } else {
                HSB_LOG_WARN("{}: {}", info.device_node, error);
            }
        }
        if (!identified) {
            HSB_LOG_WARN("{}: identity unavailable (no IDENTITY ioctl, sysfs attributes or BAR0 access); "
                         "announcing with an all-zero UUID",
                info.device_node);
        }
        finish_info(info, from_ioctl);
        return info;
    }

    std::vector<std::filesystem::path> class_entries(const std::filesystem::path& sysfs_class)
    {
        std::vector<std::filesystem::path> entries;
        std::error_code ec;
        std::filesystem::directory_iterator it(sysfs_class, ec);
        if (ec) {
            HSB_LOG_DEBUG("{}: {}", sysfs_class.string(), ec.message());
            return entries;
        }
        for (const auto& entry : it) {
            entries.push_back(entry.path());
        }
        // Deterministic order: by minor, so hololink_pcie2 sorts after hololink_pcie10.
        std::sort(entries.begin(), entries.end(),
            [](const std::filesystem::path& a, const std::filesystem::path& b) {
                const auto ma = minor_from_class_name(a.filename().string());
                const auto mb = minor_from_class_name(b.filename().string());
                if (ma && mb && *ma != *mb) {
                    return *ma < *mb;
                }
                return a.filename().string() < b.filename().string();
            });
        return entries;
    }

} // namespace

std::string uuid_to_text(const Uuid& u)
{
    return fmt::format(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
        u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
}

bool uuid_from_text(const std::string& text, Uuid& uuid)
{
    std::string hex;
    hex.reserve(32);
    if (text.size() == 36) {
        for (size_t i = 0; i < text.size(); i++) {
            const char c = text[i];
            if (i == 8 || i == 13 || i == 18 || i == 23) {
                if (c != '-') {
                    return false;
                }
                continue;
            }
            hex.push_back(c);
        }
    } else if (text.size() == 32) {
        hex = text;
    } else {
        return false;
    }
    if (!is_hex(hex, 0, 32)) {
        return false;
    }
    Uuid out {};
    for (size_t i = 0; i < out.size(); i++) {
        out[i] = static_cast<uint8_t>(std::strtoul(hex.substr(i * 2, 2).c_str(), nullptr, 16));
    }
    uuid = out;
    return true;
}

bool uuid_is_valid(const Uuid& uuid)
{
    const bool all_zero = std::all_of(uuid.begin(), uuid.end(), [](uint8_t b) { return b == 0x00; });
    const bool all_ones = std::all_of(uuid.begin(), uuid.end(), [](uint8_t b) { return b == 0xFF; });
    return !all_zero && !all_ones;
}

std::vector<PcieDeviceInfo> scan_pcie_devices(const std::filesystem::path& sysfs_class,
    const std::filesystem::path& dev_dir)
{
    std::vector<PcieDeviceInfo> devices;
    for (const auto& entry : class_entries(sysfs_class)) {
        try {
            if (auto info = describe_entry(entry, dev_dir)) {
                devices.push_back(std::move(*info));
            }
        } catch (const std::exception& e) {
            HSB_LOG_WARN("{}: {}; skipping", entry.string(), e.what());
        }
    }
    return devices;
}

bool read_device_info(uint16_t minor, PcieDeviceInfo& info, std::string& error,
    const std::filesystem::path& sysfs_class, const std::filesystem::path& dev_dir)
{
    const std::filesystem::path entry = sysfs_class / fmt::format("{}{}", DEVICE_NAME, minor);
    std::error_code ec;
    if (std::filesystem::is_directory(entry, ec)) {
        if (auto described = describe_entry(entry, dev_dir)) {
            info = std::move(*described);
            return true;
        }
    }
    // No (usable) class entry: fall back to the node alone, as an old driver would leave it.
    FdGuard fd(open_device(dev_dir, minor, false, error));
    if (!fd) {
        return false;
    }
    PcieDeviceInfo fresh;
    fresh.minor = minor;
    fresh.device_node = device_node_path(dev_dir, minor);
    bool from_ioctl = identity_from_ioctl(fd.get(), fresh, error);
    if (!from_ioctl && !identity_from_bar(fd.get(), fresh, error)) {
        return false;
    }
    finish_info(fresh, from_ioctl);
    info = std::move(fresh);
    return true;
}

bool bar_read32(uint16_t minor, uint32_t offset, uint32_t& value, std::string& error,
    const std::filesystem::path& dev_dir)
{
    FdGuard fd(open_device(dev_dir, minor, false, error));
    if (!fd) {
        return false;
    }
    return ioctl_read32(fd.get(), offset, value, error);
}

bool bar_write32(uint16_t minor, uint32_t offset, uint32_t value, std::string& error,
    const std::filesystem::path& dev_dir)
{
    FdGuard fd(open_device(dev_dir, minor, true, error));
    if (!fd) {
        return false;
    }
    return ioctl_write32(fd.get(), offset, value, error);
}

bool read_uuid(uint16_t minor, Uuid& uuid, std::string& error, const std::filesystem::path& dev_dir)
{
    FdGuard fd(open_device(dev_dir, minor, false, error));
    if (!fd) {
        return false;
    }
    return ioctl_read_uuid(fd.get(), uuid, error);
}

bool program_uuid(uint16_t minor, const Uuid& uuid, std::string& error,
    const std::filesystem::path& dev_dir)
{
    FdGuard fd(open_device(dev_dir, minor, true, error));
    if (!fd) {
        return false;
    }
    // MSW first: HSB_UUID_3 (0xB8) carries uuid[0..3], HSB_UUID_0 (0xAC) uuid[12..15].
    for (size_t i = 0; i < 4; i++) {
        if (!ioctl_write32(fd.get(), UUID_REGS[i], uuid_to_word(uuid, i * 4), error)) {
            return false;
        }
    }
    Uuid readback {};
    if (!ioctl_read_uuid(fd.get(), readback, error)) {
        return false;
    }
    if (readback != uuid) {
        error = fmt::format("UUID readback mismatch: wrote {}, read {}",
            uuid_to_text(uuid), uuid_to_text(readback));
        return false;
    }
    HSB_LOG_INFO("{}: board UUID programmed: {}", device_node_path(dev_dir, minor), uuid_to_text(uuid));
    return true;
}

} // namespace hololink::module::pcie
