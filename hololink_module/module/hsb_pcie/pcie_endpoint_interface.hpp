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

#ifndef HOLOLINK_MODULE_HSB_PCIE_PCIE_ENDPOINT_INTERFACE_HPP
#define HOLOLINK_MODULE_HSB_PCIE_PCIE_ENDPOINT_INTERFACE_HPP

/**
 * @file
 * @brief Abstract BAR0 register access to a Hololink PCIe endpoint.
 *
 * This is the seam between the address policy / ECB bridge and whatever actually
 * carries a 32-bit BAR0 access: the kernel driver (PcieEndpoint, next phase) or the
 * in-memory LoopbackPcieEndpoint used by the tests. It is the BAR-only subset of the
 * PCIEPeripheral surface in hsb_pcie/driver/src/hsb_pcie/msix_pcie.hpp -- DMA, dma-buf
 * and MSI-X are deliberately absent; the module-native receiver owns those on its own fd.
 */

#include <cstddef>
#include <cstdint>
#include <string>

namespace hololink::module::hsb_pcie {

/// 128-bit board UUID as programmed into HSB_UUID_0..3.
inline constexpr size_t UUID_BYTES = 16;

/// PcieIdentity::flags bits.
inline constexpr uint32_t IDENTITY_VERSION_VALID = 1u << 0;
inline constexpr uint32_t IDENTITY_MAGIC_VALID = 1u << 1;
inline constexpr uint32_t IDENTITY_UUID_VALID = 1u << 2;

/**
 * @brief What the endpoint reports about itself.
 *
 * Mirrors the driver's hl_pcie_identity_info (IDENTITY ioctl) plus the fields the
 * enumerator derives from sysfs, so a PcieHololink can re-identify a board after SYS_RST
 * and compare against the enumeration metadata.
 */
struct PcieIdentity {
    uint32_t hsb_version { 0 }; ///< HSB_VER (BAR0 0x80)
    uint32_t hsb_date { 0 }; ///< HSB_DATE (BAR0 0x84)
    uint32_t hsb_status { 0 }; ///< HSB_STAT (BAR0 0x88), the "HLPC" magic on bring-up bitfiles
    uint8_t uuid[UUID_BYTES] { 0 }; ///< HSB_UUID_3..0, byte order per msix_pcie.hpp UUID_REGS
    uint32_t flags { 0 }; ///< IDENTITY_*_VALID bits
    std::string bdf; ///< "DDDD:BB:DD.F"
    uint16_t requester_id { 0 }; ///< (bus << 8) | devfn, as programmed into PCIE_DP
    uint32_t minor { 0 }; ///< N in /dev/hololink_pcieN
    uint32_t num_msix_vectors { 0 };
    uint32_t bar0_len { 0 };
};

/**
 * @brief 32-bit register access to BAR0 of one endpoint.
 *
 * BAR0 is BAR0_LEN bytes; offsets must be 4-byte aligned and inside the window.
 * Implementations return Status::INVALID_ADDRESS for anything else and never touch
 * hardware for it -- the address policy relies on that to turn an out-of-window
 * access into an ECB RESPONSE_INVALID_ADDR.
 */
class PcieEndpointInterface {
public:
    /// BAR0 extent: USR_MODULE_END + 1 in the register map.
    static constexpr uint32_t BAR0_LEN = 0x100000u;
    static constexpr uint32_t REG_SIZE = 4u;

    enum class Status {
        SUCCESS = 0,
        INVALID_ADDRESS, ///< misaligned or outside BAR0
        READ_FAILED, ///< the carrier (ioctl/mmap) failed the read
        WRITE_FAILED, ///< the carrier failed the write
        NOT_ATTACHED, ///< open() has not succeeded (or close() was called)
    };

    virtual ~PcieEndpointInterface() = default;

    /// Attach to the endpoint. Returns true when bar_read32/bar_write32 may be used.
    virtual bool open() = 0;
    /// Detach; subsequent register accesses return NOT_ATTACHED.
    virtual void close() = 0;
    virtual bool is_open() const = 0;

    virtual uint32_t minor() const = 0;
    virtual const std::string& bdf() const = 0;
    virtual uint16_t requester_id() const = 0;
    /**
     * Snapshot of what the endpoint currently reports. Returned by value: implementations
     * refresh their identity under a lock on every call, so handing out a reference would
     * let a concurrent caller (e.g. an enumerator) observe a torn std::string bdf.
     */
    virtual PcieIdentity identity() const = 0;

    virtual Status bar_read32(uint32_t offset, uint32_t& value) = 0;
    virtual Status bar_write32(uint32_t offset, uint32_t value) = 0;

    /// True when `offset` names an aligned 32-bit word inside BAR0.
    static constexpr bool valid_offset(uint32_t offset)
    {
        return ((offset & (REG_SIZE - 1u)) == 0u) && (offset < BAR0_LEN);
    }
};

const char* to_string(PcieEndpointInterface::Status status);

} // namespace hololink::module::hsb_pcie

#endif /* HOLOLINK_MODULE_HSB_PCIE_PCIE_ENDPOINT_INTERFACE_HPP */
