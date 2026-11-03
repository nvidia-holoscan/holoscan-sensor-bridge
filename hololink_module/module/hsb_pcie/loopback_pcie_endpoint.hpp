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

#ifndef HOLOLINK_MODULE_HSB_PCIE_LOOPBACK_PCIE_ENDPOINT_HPP
#define HOLOLINK_MODULE_HSB_PCIE_LOOPBACK_PCIE_ENDPOINT_HPP

/**
 * @file
 * @brief In-memory BAR0 model of a Hololink PCIe endpoint, for tests and hardware-free
 * development hosts.
 *
 * Ported from hsb_pcie/driver/src/hsb_pcie/loopback_pcie.{hpp,cpp} (LoopbackPCIE), keeping
 * only the BAR0 register file and the minimal I2C START->DONE behaviour; everything about
 * PCIEController / relay transactions / DMA / MSI-X was dropped because the module path does
 * not go through those.
 */

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pcie_endpoint_interface.hpp"

namespace hololink::module::hsb_pcie {

/// The UUID the hsb_pcie endpoint bitfile reports (read-only on the FPGA):
/// 58c1419b-1e7a-4120-95fd-4efc156187c1, canonical byte order.
inline constexpr uint8_t HSB_PCIE_MODULE_UUID[UUID_BYTES] = {
    0x58, 0xc1, 0x41, 0x9b, 0x1e, 0x7a, 0x41, 0x20,
    0x95, 0xfd, 0x4e, 0xfc, 0x15, 0x61, 0x87, 0xc1
};

/**
 * @brief BAR0 registers holding the board UUID, most-significant word first.
 *
 * Same convention as msix_pcie.hpp UUID_REGS: HSB_UUID_3 (0xB8) carries the leading four
 * bytes of the canonical string, HSB_UUID_0 (0xAC) the trailing four; each register is
 * big-endian with respect to the byte array.
 */
inline constexpr uint32_t UUID_REGS[4] = { 0xB8u, 0xB4u, 0xB0u, 0xACu };

/// Pack uuid[byte_index .. byte_index+3] into one big-endian register word.
inline constexpr uint32_t uuid_to_word(const uint8_t* uuid, size_t byte_index)
{
    return (static_cast<uint32_t>(uuid[byte_index]) << 24)
        | (static_cast<uint32_t>(uuid[byte_index + 1]) << 16)
        | (static_cast<uint32_t>(uuid[byte_index + 2]) << 8)
        | static_cast<uint32_t>(uuid[byte_index + 3]);
}

/// Unpack one big-endian register word into uuid[byte_index .. byte_index+3].
inline void uuid_from_word(uint32_t word, uint8_t* uuid, size_t byte_index)
{
    uuid[byte_index] = static_cast<uint8_t>(word >> 24);
    uuid[byte_index + 1] = static_cast<uint8_t>(word >> 16);
    uuid[byte_index + 2] = static_cast<uint8_t>(word >> 8);
    uuid[byte_index + 3] = static_cast<uint8_t>(word);
}

/**
 * @brief A sparse BAR0 register file seeded with a plausible identity.
 *
 * Writes read back; registers never written read 0. An I2C START completes immediately
 * (STATUS = DONE) and the trailing zero write returns the controller to idle, which is just
 * enough for a host polling for completion to make progress. There is no sensor on the far
 * side, so the I2C data buffer keeps whatever the host put there.
 *
 * Tests inspect and prime registers with peek()/poke(), which bypass the open() gate.
 */
class LoopbackPcieEndpoint : public PcieEndpointInterface {
public:
    explicit LoopbackPcieEndpoint(uint32_t minor = 0, std::string bdf = "0000:00:00.0");
    ~LoopbackPcieEndpoint() override = default;

    bool open() override;
    void close() override;
    bool is_open() const override { return open_; }

    uint32_t minor() const override { return identity_.minor; }
    const std::string& bdf() const override { return identity_.bdf; }
    uint16_t requester_id() const override { return identity_.requester_id; }
    /// Re-read from the register file on every call so a UUID programmed through
    /// bar_write32 shows up here exactly as it would after the driver re-reads it.
    PcieIdentity identity() const override;

    Status bar_read32(uint32_t offset, uint32_t& value) override;
    Status bar_write32(uint32_t offset, uint32_t value) override;

    // ---- test hooks --------------------------------------------------------------------
    /// Current value of a register (0 when never written). No alignment/range check.
    uint32_t peek(uint32_t offset) const;
    /// Set a register directly, without the I2C side effects of a host write.
    void poke(uint32_t offset, uint32_t value);
    /// True when the host (or poke) has ever touched `offset`.
    bool has_register(uint32_t offset) const;
    /// Number of BAR0 registers touched so far. Useful as a liveness check.
    size_t register_count() const;
    /// Forget every register and re-seed the identity block.
    void reset_registers();

private:
    uint32_t read_register(uint32_t offset) const;
    void write_register(uint32_t offset, uint32_t value);
    void seed_identity();

    bool open_ { false };
    mutable std::mutex mutex_;
    mutable PcieIdentity identity_;
    std::unordered_map<uint32_t, uint32_t> registers_;
};

} // namespace hololink::module::hsb_pcie

#endif /* HOLOLINK_MODULE_HSB_PCIE_LOOPBACK_PCIE_ENDPOINT_HPP */
