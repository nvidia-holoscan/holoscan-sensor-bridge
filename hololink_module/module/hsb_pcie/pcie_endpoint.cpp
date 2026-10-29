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

// PcieEndpoint: BAR0 register access to /dev/hololink_pcieN. Port of the BAR-only half of
// hsb_pcie/driver/src/hsb_pcie/msix_pcie.cpp (MSIXPCIE::open_device, discover_bdf, map_bar0,
// close_all, bar_read32, bar_write32, read_identity, read_uuid, program_uuid) with
// fprintf(stderr) replaced by HSB_LOG_* and identity served by HL_PCIE_IOC_IDENTITY when the
// driver has it.

#include "pcie_endpoint.hpp"

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fmt/format.h>

#include "hololink_pcie_uapi.h"
#include "pcie_regs.hpp"
#include "pcie_sysfs.hpp"
#include <hololink/module/logging.hpp>

// UUID_REGS / uuid_to_word / uuid_from_word live next to the loopback endpoint.
#include "loopback_pcie_endpoint.hpp"

namespace hololink::module::hsb_pcie {

// The ioctl encoding embeds the argument size, so a struct that drifts from the kernel's
// definition surfaces as ENOTTY rather than as corruption. Pin the layouts here too.
static_assert(sizeof(struct hl_pcie_rw) == 24, "hl_pcie_rw layout changed");
static_assert(sizeof(struct hl_pcie_identity_info) == 64, "hl_pcie_identity_info layout changed");

namespace {

    /// PCI requester ID from a BDF string: bus[15:8] | device[7:3] | function[2:0].
    bool requester_id_from_bdf(const std::string& bdf, uint16_t& requester_id)
    {
        unsigned domain = 0, bus = 0, device = 0, function = 0;
        if (std::sscanf(bdf.c_str(), "%x:%x:%x.%x", &domain, &bus, &device, &function) != 4) {
            return false;
        }
        requester_id = static_cast<uint16_t>(((bus & 0xFFu) << 8) | ((device & 0x1Fu) << 3) | (function & 0x07u));
        return true;
    }

    std::string uuid_text(const uint8_t* uuid)
    {
        return fmt::format(
            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            uuid[0], uuid[1], uuid[2], uuid[3], uuid[4], uuid[5], uuid[6], uuid[7],
            uuid[8], uuid[9], uuid[10], uuid[11], uuid[12], uuid[13], uuid[14], uuid[15]);
    }

    bool uuid_valid(const uint8_t* uuid)
    {
        for (size_t i = 0; i < UUID_BYTES; i++) {
            if (uuid[i] != 0x00 && uuid[i] != 0xFF) {
                return true;
            }
        }
        return false;
    }

} // namespace

std::string PcieEndpoint::device_path(uint16_t minor)
{
    return fmt::format("/dev/{}{}", DEVICE_NAME, minor);
}

std::string PcieEndpoint::sysfs_class_path(uint16_t minor)
{
    return fmt::format("/sys/class/{}/{}{}", DEVICE_NAME, DEVICE_NAME, minor);
}

PcieEndpoint::PcieEndpoint(uint16_t minor, const PcieEndpointConfig& config)
    : minor_(minor)
    , config_(config)
{
    identity_.minor = minor;
}

PcieEndpoint::~PcieEndpoint()
{
    close();
}

bool PcieEndpoint::open()
{
    std::scoped_lock<std::mutex> lock(mutex_);
    if (open_) {
        return true;
    }
    if (!open_device()) {
        return false;
    }
    discover_bdf();
    map_bar0();
    if (!refresh_identity_locked()) {
        HSB_LOG_ERROR("{} failed identity validation (HSB_VER={:#x})", device_path(minor_), identity_.hsb_version);
        close_all();
        return false;
    }
    // When the sysfs class link did not resolve, map_bar0() had no BDF to work with and
    // bailed out; the IDENTITY ioctl may have supplied one since, so try the mmap again
    // rather than silently staying on the slower 32-bit ioctl path.
    if (!bar0_ && !bdf_.empty()) {
        map_bar0();
    }
    open_ = true;
    HSB_LOG_INFO("{} opened: bdf={} requester_id={:#06x} hsb_version={:#x} hsb_date={} uuid={} bar0={}",
        device_path(minor_), bdf_, requester_id_, identity_.hsb_version, identity_.hsb_date,
        uuid_text(identity_.uuid), bar0_ ? "mmap" : "ioctl");
    return true;
}

void PcieEndpoint::close()
{
    std::scoped_lock<std::mutex> lock(mutex_);
    close_all();
    open_ = false;
}

bool PcieEndpoint::open_device()
{
    const std::string path = device_path(minor_);
    fd_ = ::open(path.c_str(), O_RDWR);
    if (fd_ < 0) {
        HSB_LOG_ERROR("{}", ::hololink::module::pcie::device_node_open_error(path, errno));
        return false;
    }
    return true;
}

void PcieEndpoint::discover_bdf()
{
    // Resolve the sysfs class link back to the PCI device directory, whose basename is the BDF.
    const std::string link = sysfs_class_path(minor_) + "/device";
    char resolved[PATH_MAX];
    const char* real = ::realpath(link.c_str(), resolved);
    if (!real) {
        HSB_LOG_WARN("could not resolve {}: {}. BAR0 mmap and requester-ID discovery are "
                     "unavailable; falling back to ioctl access.",
            link, std::strerror(errno));
        bdf_.clear();
        requester_id_ = 0;
        return;
    }
    std::string path(real);
    const size_t slash = path.find_last_of('/');
    bdf_ = (slash == std::string::npos) ? path : path.substr(slash + 1);
    if (!requester_id_from_bdf(bdf_, requester_id_)) {
        HSB_LOG_WARN("unparsable PCI BDF '{}'", bdf_);
        requester_id_ = 0;
    }
}

void PcieEndpoint::map_bar0()
{
    if (!config_.prefer_mmap || bdf_.empty()) {
        return;
    }
    // On arm64 the PCI core maps non-prefetchable BARs as Device memory (strongly ordered,
    // no write buffering), so plain volatile accesses to this window are correctly ordered.
    const std::string path = "/sys/bus/pci/devices/" + bdf_ + "/resource0";
    bar0_fd_ = ::open(path.c_str(), O_RDWR | O_SYNC);
    if (bar0_fd_ < 0) {
        HSB_LOG_INFO("open {} failed: {}. Using the driver's 32-bit ioctls.", path, std::strerror(errno));
        return;
    }
    void* mapped = ::mmap(nullptr, BAR0_LEN, PROT_READ | PROT_WRITE, MAP_SHARED, bar0_fd_, 0);
    if (mapped == MAP_FAILED) {
        HSB_LOG_INFO("mmap {} failed: {}. Using the driver's 32-bit ioctls.", path, std::strerror(errno));
        ::close(bar0_fd_);
        bar0_fd_ = -1;
        return;
    }
    bar0_ = static_cast<volatile uint32_t*>(mapped);
    bar0_size_ = BAR0_LEN;
}

void PcieEndpoint::close_all()
{
    if (bar0_) {
        ::munmap(const_cast<uint32_t*>(bar0_), bar0_size_);
        bar0_ = nullptr;
        bar0_size_ = 0;
    }
    if (bar0_fd_ >= 0) {
        ::close(bar0_fd_);
        bar0_fd_ = -1;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

PcieEndpointInterface::Status PcieEndpoint::bar_read32_locked(uint32_t offset, uint32_t& value) const
{
    if (!valid_offset(offset)) {
        return Status::INVALID_ADDRESS;
    }
    if (fd_ < 0) {
        return Status::NOT_ATTACHED;
    }
    if (bar0_) {
        value = bar0_[offset / REG_SIZE];
        return Status::SUCCESS;
    }
    struct hl_pcie_rw rw { };
    rw.bar = 0;
    rw.offset = offset;
    rw.width = REG_SIZE;
    rw.count = 1;
    rw.data = 0;
    if (::ioctl(fd_, HL_PCIE_IOC_READ, &rw) < 0) {
        return errno == EINVAL ? Status::INVALID_ADDRESS : Status::READ_FAILED;
    }
    value = static_cast<uint32_t>(rw.data);
    return Status::SUCCESS;
}

PcieEndpointInterface::Status PcieEndpoint::bar_write32_locked(uint32_t offset, uint32_t value)
{
    if (!valid_offset(offset)) {
        return Status::INVALID_ADDRESS;
    }
    if (fd_ < 0) {
        return Status::NOT_ATTACHED;
    }
    if (bar0_) {
        bar0_[offset / REG_SIZE] = value;
        // Keep the compiler from sinking this past a subsequent read; Device-memory
        // ordering on arm64 handles the rest.
        __atomic_signal_fence(__ATOMIC_SEQ_CST);
        return Status::SUCCESS;
    }
    struct hl_pcie_rw rw { };
    rw.bar = 0;
    rw.offset = offset;
    rw.width = REG_SIZE;
    rw.count = 1;
    rw.data = value;
    if (::ioctl(fd_, HL_PCIE_IOC_WRITE, &rw) < 0) {
        return errno == EINVAL ? Status::INVALID_ADDRESS : Status::WRITE_FAILED;
    }
    return Status::SUCCESS;
}

PcieEndpointInterface::Status PcieEndpoint::bar_read32(uint32_t offset, uint32_t& value)
{
    // Range/alignment failures never touch hardware, whatever the attach state.
    if (!valid_offset(offset)) {
        return Status::INVALID_ADDRESS;
    }
    std::scoped_lock<std::mutex> lock(mutex_);
    if (!open_) {
        return Status::NOT_ATTACHED;
    }
    return bar_read32_locked(offset, value);
}

PcieEndpointInterface::Status PcieEndpoint::bar_write32(uint32_t offset, uint32_t value)
{
    if (!valid_offset(offset)) {
        return Status::INVALID_ADDRESS;
    }
    std::scoped_lock<std::mutex> lock(mutex_);
    if (!open_) {
        return Status::NOT_ATTACHED;
    }
    return bar_write32_locked(offset, value);
}

bool PcieEndpoint::read_identity_ioctl_locked() const
{
    if (identity_ioctl_ == IoctlSupport::No) {
        return false;
    }
    struct hl_pcie_identity_info info { };
    if (::ioctl(fd_, HL_PCIE_IOC_IDENTITY, &info) < 0) {
        if (errno == ENOTTY || errno == EINVAL) {
            // Older driver without the IDENTITY extension.
            if (identity_ioctl_ == IoctlSupport::Unknown) {
                HSB_LOG_INFO("{}: driver has no HL_PCIE_IOC_IDENTITY; reading identity from BAR0", device_path(minor_));
            }
            identity_ioctl_ = IoctlSupport::No;
        } else {
            HSB_LOG_WARN("{}: HL_PCIE_IOC_IDENTITY failed: {}", device_path(minor_), std::strerror(errno));
        }
        return false;
    }
    identity_ioctl_ = IoctlSupport::Yes;

    identity_.hsb_version = info.hsb_version;
    identity_.hsb_date = info.hsb_date;
    identity_.hsb_status = info.hsb_status;
    std::memcpy(identity_.uuid, info.uuid, UUID_BYTES);
    identity_.flags = 0;
    if (info.flags & HL_PCIE_ID_F_VERSION_VALID) {
        identity_.flags |= IDENTITY_VERSION_VALID;
    }
    if (info.flags & HL_PCIE_ID_F_MAGIC_VALID) {
        identity_.flags |= IDENTITY_MAGIC_VALID;
    }
    if (info.flags & HL_PCIE_ID_F_UUID_VALID) {
        identity_.flags |= IDENTITY_UUID_VALID;
    }
    identity_.bdf = fmt::format("{:04x}:{:02x}:{:02x}.{:x}", info.domain, info.bus,
        (info.devfn >> 3) & 0x1Fu, info.devfn & 0x7u);
    identity_.requester_id = info.requester_id;
    identity_.minor = info.minor;
    identity_.num_msix_vectors = info.num_msix_vectors;
    identity_.bar0_len = static_cast<uint32_t>(info.bar0_len);

    // sysfs is the authority for the BDF when the class link resolved (the driver's answer
    // must agree); otherwise adopt the driver's.
    if (bdf_.empty()) {
        bdf_ = identity_.bdf;
        requester_id_ = identity_.requester_id;
    } else if (bdf_ != identity_.bdf) {
        HSB_LOG_WARN("{}: sysfs BDF '{}' disagrees with the driver's '{}'", device_path(minor_), bdf_, identity_.bdf);
    }
    return true;
}

bool PcieEndpoint::read_identity_bar_locked() const
{
    using namespace hololink::pcie;
    uint32_t value = 0;
    if (bar_read32_locked(HSB_VER, value) != Status::SUCCESS) {
        return false;
    }
    identity_.hsb_version = value;
    bar_read32_locked(HSB_DATE, identity_.hsb_date);
    bar_read32_locked(HSB_STAT, identity_.hsb_status);
    for (size_t i = 0; i < 4; i++) {
        uint32_t word = 0;
        if (bar_read32_locked(UUID_REGS[i], word) != Status::SUCCESS) {
            return false;
        }
        uuid_from_word(word, identity_.uuid, i * 4);
    }
    identity_.flags = 0;
    if (identity_.hsb_version != 0 && identity_.hsb_version != 0xFFFFFFFFu) {
        identity_.flags |= IDENTITY_VERSION_VALID;
    }
    if (identity_.hsb_status == HSB_MAGIC_EXPECT) {
        identity_.flags |= IDENTITY_MAGIC_VALID;
    }
    if (uuid_valid(identity_.uuid)) {
        identity_.flags |= IDENTITY_UUID_VALID;
    }
    identity_.bdf = bdf_;
    identity_.requester_id = requester_id_;
    identity_.minor = minor_;
    identity_.bar0_len = BAR0_LEN;
    return true;
}

bool PcieEndpoint::refresh_identity_locked() const
{
    if (!read_identity_ioctl_locked() && !read_identity_bar_locked()) {
        return false;
    }
    // A BAR read of all-ones means the endpoint is not completing reads at all -- an
    // unassigned BAR or a downed link. Never accept that.
    if (!(identity_.flags & IDENTITY_VERSION_VALID)) {
        HSB_LOG_WARN("{}: HSB_VER reads {:#010x}; check the PCIe BAR path", device_path(minor_), identity_.hsb_version);
        return false;
    }
    return true;
}

PcieIdentity PcieEndpoint::identity() const
{
    std::scoped_lock<std::mutex> lock(mutex_);
    if (open_) {
        refresh_identity_locked();
    }
    return identity_; // copy: never hand out a reference to state rewritten under mutex_
}

PcieEndpointInterface::Status PcieEndpoint::read_uuid(uint8_t uuid[UUID_BYTES])
{
    std::scoped_lock<std::mutex> lock(mutex_);
    if (!open_) {
        return Status::NOT_ATTACHED;
    }
    for (size_t i = 0; i < 4; i++) {
        uint32_t word = 0;
        Status status = bar_read32_locked(UUID_REGS[i], word);
        if (status != Status::SUCCESS) {
            return status;
        }
        uuid_from_word(word, uuid, i * 4);
    }
    return Status::SUCCESS;
}

PcieEndpointInterface::Status PcieEndpoint::program_uuid(const uint8_t uuid[UUID_BYTES])
{
    std::scoped_lock<std::mutex> lock(mutex_);
    if (!open_) {
        return Status::NOT_ATTACHED;
    }
    for (size_t i = 0; i < 4; i++) {
        Status status = bar_write32_locked(UUID_REGS[i], uuid_to_word(uuid, i * 4));
        if (status != Status::SUCCESS) {
            return status;
        }
    }
    uint8_t readback[UUID_BYTES];
    for (size_t i = 0; i < 4; i++) {
        uint32_t word = 0;
        Status status = bar_read32_locked(UUID_REGS[i], word);
        if (status != Status::SUCCESS) {
            return status;
        }
        uuid_from_word(word, readback, i * 4);
    }
    if (std::memcmp(readback, uuid, UUID_BYTES) != 0) {
        HSB_LOG_ERROR("{}: UUID readback {} != programmed {} (identity registers read-only?)",
            device_path(minor_), uuid_text(readback), uuid_text(uuid));
        std::memcpy(identity_.uuid, readback, UUID_BYTES);
        return Status::WRITE_FAILED;
    }
    std::memcpy(identity_.uuid, uuid, UUID_BYTES);
    identity_.flags |= IDENTITY_UUID_VALID;
    HSB_LOG_INFO("{}: board UUID programmed: {}", device_path(minor_), uuid_text(uuid));
    return Status::SUCCESS;
}

} // namespace hololink::module::hsb_pcie
