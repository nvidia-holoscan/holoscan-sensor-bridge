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

#ifndef HOLOLINK_MODULE_HSB_PCIE_PCIE_ENDPOINT_HPP
#define HOLOLINK_MODULE_HSB_PCIE_PCIE_ENDPOINT_HPP

/**
 * @file
 * @brief BAR0 access to a Hololink PCIe endpoint through the hololink_pcie kernel driver.
 *
 * The BAR-only subset of hsb_pcie/driver/src/hsb_pcie/msix_pcie.{hpp,cpp} (MSIXPCIE):
 * open_device / discover_bdf / map_bar0 / read_identity / bar_read32 / bar_write32 /
 * read_uuid / program_uuid. Dropped: the relay transaction path, DMA / dma-buf / MSI-X ioctls,
 * PCIEController attachment and every emulator include -- the module-native receiver owns
 * the data path on its own fd. Identity is preferably read with HL_PCIE_IOC_IDENTITY
 * (driver extension); an older driver that answers ENOTTY falls back to BAR reads.
 */

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

#include "pcie_endpoint_interface.hpp"

namespace hololink::module::hsb_pcie {

/// Construction-time knobs for PcieEndpoint.
struct PcieEndpointConfig {
    /// Prefer mmap'ing the sysfs resource0 BAR window; fall back to the driver's 32-bit
    /// ioctls when that is unavailable (no permission, BAR claimed exclusively, ...).
    bool prefer_mmap { true };
};

class PcieEndpoint : public PcieEndpointInterface {
public:
    static constexpr const char* DEVICE_NAME = "hololink_pcie";

    /// "/dev/hololink_pcie<minor>"
    static std::string device_path(uint16_t minor);
    /// "/sys/class/hololink_pcie/hololink_pcie<minor>"
    static std::string sysfs_class_path(uint16_t minor);

    /// Cheap: no I/O until open().
    explicit PcieEndpoint(uint16_t minor, const PcieEndpointConfig& config = PcieEndpointConfig {});
    ~PcieEndpoint() override;

    PcieEndpoint(const PcieEndpoint&) = delete;
    PcieEndpoint& operator=(const PcieEndpoint&) = delete;

    /**
     * Open /dev/hololink_pcie<minor>, resolve the BDF from sysfs, map BAR0 (or arrange the
     * ioctl fallback) and read the identity. Returns false (and logs) when the device cannot
     * be opened or HSB_VER does not read as a live design.
     */
    bool open() override;
    void close() override;
    bool is_open() const override { return open_; }

    uint32_t minor() const override { return minor_; }
    const std::string& bdf() const override { return bdf_; }
    uint16_t requester_id() const override { return requester_id_; }
    /// Re-read from the driver on every call while open, so a UUID programmed after open()
    /// is reflected; the last value otherwise. Note that HL_PCIE_IOC_IDENTITY serves
    /// hsb_version/hsb_date/hsb_status and the VERSION_VALID flag from the driver's
    /// probe-time cache -- only the UUID is re-read -- so this is NOT a liveness probe of
    /// the design. Use bar_read32(HSB_VER) for that (see PcieHololink::reset()).
    PcieIdentity identity() const override;

    Status bar_read32(uint32_t offset, uint32_t& value) override;
    Status bar_write32(uint32_t offset, uint32_t value) override;

    /// Read HSB_UUID_3..0 (MSW first, big-endian words) into `uuid`.
    Status read_uuid(uint8_t uuid[UUID_BYTES]);
    /**
     * Write `uuid` into HSB_UUID_3..0 and verify the readback. Returns WRITE_FAILED when the
     * readback differs (RTL leaves the registers read-only), or the transport error.
     */
    Status program_uuid(const uint8_t uuid[UUID_BYTES]);

    bool bar0_mapped() const { return bar0_ != nullptr; }
    /// True when the driver served HL_PCIE_IOC_IDENTITY (false: BAR-read fallback in use).
    bool identity_ioctl_supported() const { return identity_ioctl_ == IoctlSupport::Yes; }

private:
    enum class IoctlSupport { Unknown,
        Yes,
        No };

    bool open_device();
    void discover_bdf();
    void map_bar0();
    void close_all();
    /// Fill identity_ (under mutex_). Returns false when HSB_VER is not a live design.
    bool refresh_identity_locked() const;
    bool read_identity_ioctl_locked() const;
    bool read_identity_bar_locked() const;
    Status bar_read32_locked(uint32_t offset, uint32_t& value) const;
    Status bar_write32_locked(uint32_t offset, uint32_t value);

    uint16_t minor_;
    PcieEndpointConfig config_;
    int fd_ { -1 };
    int bar0_fd_ { -1 };
    volatile uint32_t* bar0_ { nullptr };
    size_t bar0_size_ { 0 };
    bool open_ { false };
    // mutable: identity() (const) adopts the driver's BDF when sysfs did not resolve one.
    mutable std::string bdf_;
    mutable uint16_t requester_id_ { 0 };
    mutable std::mutex mutex_;
    mutable PcieIdentity identity_;
    mutable IoctlSupport identity_ioctl_ { IoctlSupport::Unknown };
};

} // namespace hololink::module::hsb_pcie

#endif /* HOLOLINK_MODULE_HSB_PCIE_PCIE_ENDPOINT_HPP */
