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

// In-memory BAR0 model; BAR-only port of hsb_pcie/driver/src/hsb_pcie/loopback_pcie.cpp.

#include "loopback_pcie_endpoint.hpp"

#include <cstdio>
#include <cstring>

#include "pcie_regs.hpp"

namespace hololink::module::hsb_pcie {

namespace {
    // Plausible stand-ins so identity validation and the smoke paths behave like hardware.
    constexpr uint32_t LOOPBACK_HSB_DATE = 20260219u;
    constexpr uint32_t LOOPBACK_MSIX_VECTORS = 64u;

    /// (bus << 8) | devfn from "DDDD:BB:DD.F"; 0 when the string does not parse.
    uint16_t requester_id_from_bdf(const std::string& bdf)
    {
        unsigned domain = 0, bus = 0, device = 0, function = 0;
        if (std::sscanf(bdf.c_str(), "%x:%x:%x.%x", &domain, &bus, &device, &function) != 4) {
            return 0;
        }
        return static_cast<uint16_t>(((bus & 0xFFu) << 8) | ((device & 0x1Fu) << 3) | (function & 0x7u));
    }
} // namespace

const char* to_string(PcieEndpointInterface::Status status)
{
    switch (status) {
    case PcieEndpointInterface::Status::SUCCESS:
        return "SUCCESS";
    case PcieEndpointInterface::Status::INVALID_ADDRESS:
        return "INVALID_ADDRESS";
    case PcieEndpointInterface::Status::READ_FAILED:
        return "READ_FAILED";
    case PcieEndpointInterface::Status::WRITE_FAILED:
        return "WRITE_FAILED";
    case PcieEndpointInterface::Status::NOT_ATTACHED:
        return "NOT_ATTACHED";
    }
    return "UNKNOWN";
}

LoopbackPcieEndpoint::LoopbackPcieEndpoint(uint32_t minor, std::string bdf)
{
    identity_.minor = minor;
    identity_.bdf = std::move(bdf);
    identity_.requester_id = requester_id_from_bdf(identity_.bdf);
    identity_.num_msix_vectors = LOOPBACK_MSIX_VECTORS;
    identity_.bar0_len = BAR0_LEN;
    seed_identity();
}

void LoopbackPcieEndpoint::seed_identity()
{
    using namespace hololink::pcie;
    registers_[HSB_VER] = REG_MAP_VERSION;
    registers_[HSB_DATE] = LOOPBACK_HSB_DATE;
    registers_[HSB_STAT] = HSB_MAGIC_EXPECT;
    // The board arrives already provisioned with the module UUID (plan decision 3); the
    // registers stay writable so program_uuid round-trips here as on hardware.
    for (size_t i = 0; i < 4; i++) {
        registers_[UUID_REGS[i]] = uuid_to_word(HSB_PCIE_MODULE_UUID, i * 4);
    }
}

bool LoopbackPcieEndpoint::open()
{
    std::scoped_lock<std::mutex> lock(mutex_);
    open_ = true;
    return true;
}

void LoopbackPcieEndpoint::close()
{
    std::scoped_lock<std::mutex> lock(mutex_);
    open_ = false;
}

PcieIdentity LoopbackPcieEndpoint::identity() const
{
    using namespace hololink::pcie;
    std::scoped_lock<std::mutex> lock(mutex_);
    identity_.hsb_version = read_register(HSB_VER);
    identity_.hsb_date = read_register(HSB_DATE);
    identity_.hsb_status = read_register(HSB_STAT);
    for (size_t i = 0; i < 4; i++) {
        uuid_from_word(read_register(UUID_REGS[i]), identity_.uuid, i * 4);
    }
    identity_.flags = 0;
    if (identity_.hsb_version != 0 && identity_.hsb_version != 0xFFFFFFFFu) {
        identity_.flags |= IDENTITY_VERSION_VALID;
    }
    if (identity_.hsb_status == HSB_MAGIC_EXPECT) {
        identity_.flags |= IDENTITY_MAGIC_VALID;
    }
    bool uuid_nonzero = false;
    for (uint8_t byte : identity_.uuid) {
        uuid_nonzero |= (byte != 0);
    }
    if (uuid_nonzero) {
        identity_.flags |= IDENTITY_UUID_VALID;
    }
    return identity_;
}

uint32_t LoopbackPcieEndpoint::read_register(uint32_t offset) const
{
    auto it = registers_.find(offset);
    return it == registers_.end() ? 0u : it->second;
}

void LoopbackPcieEndpoint::write_register(uint32_t offset, uint32_t value)
{
    using namespace hololink::pcie;
    registers_[offset] = value;

    // Model just enough of the I2C controller that a host driver polling for completion
    // makes progress: a START write completes immediately, and the trailing zero write
    // returns the controller to idle.
    if (offset == I2C_CTRL_BASE + I2C_CONTROL) {
        registers_[I2C_CTRL_BASE + I2C_STATUS]
            = (value & I2C_CONTROL_START_MASK) ? I2C_STATUS_DONE_MASK : 0u;
    }
}

PcieEndpointInterface::Status LoopbackPcieEndpoint::bar_read32(uint32_t offset, uint32_t& value)
{
    std::scoped_lock<std::mutex> lock(mutex_);
    if (!open_) {
        return Status::NOT_ATTACHED;
    }
    if (!valid_offset(offset)) {
        return Status::INVALID_ADDRESS;
    }
    value = read_register(offset);
    return Status::SUCCESS;
}

PcieEndpointInterface::Status LoopbackPcieEndpoint::bar_write32(uint32_t offset, uint32_t value)
{
    std::scoped_lock<std::mutex> lock(mutex_);
    if (!open_) {
        return Status::NOT_ATTACHED;
    }
    if (!valid_offset(offset)) {
        return Status::INVALID_ADDRESS;
    }
    write_register(offset, value);
    return Status::SUCCESS;
}

uint32_t LoopbackPcieEndpoint::peek(uint32_t offset) const
{
    std::scoped_lock<std::mutex> lock(mutex_);
    return read_register(offset);
}

void LoopbackPcieEndpoint::poke(uint32_t offset, uint32_t value)
{
    std::scoped_lock<std::mutex> lock(mutex_);
    registers_[offset] = value;
}

bool LoopbackPcieEndpoint::has_register(uint32_t offset) const
{
    std::scoped_lock<std::mutex> lock(mutex_);
    return registers_.find(offset) != registers_.end();
}

size_t LoopbackPcieEndpoint::register_count() const
{
    std::scoped_lock<std::mutex> lock(mutex_);
    return registers_.size();
}

void LoopbackPcieEndpoint::reset_registers()
{
    std::scoped_lock<std::mutex> lock(mutex_);
    registers_.clear();
    seed_identity();
}

} // namespace hololink::module::hsb_pcie
