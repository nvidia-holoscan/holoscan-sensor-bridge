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

// HSB address -> BAR0 policy table (design plan section B). Replaces the I2C/SPI rebase in
// hsb_pcie/driver/src/hsb_pcie/hsb_pcie_emulator.cpp and adds shadowing of hololink-only words.

#include "address_policy.hpp"

#include <stdexcept>

#include <hololink/core/hololink.hpp>
#include <hololink/module/logging.hpp>

namespace hololink::module::hsb_pcie {

namespace {
    constexpr uint64_t ADDRESS_SPACE_END = 0x1'0000'0000ull;
    constexpr uint32_t BAR0_END = PcieEndpointInterface::BAR0_LEN;
} // namespace

const char* to_string(Action action)
{
    switch (action) {
    case Action::ForwardIdentity:
        return "ForwardIdentity";
    case Action::ForwardRebase:
        return "ForwardRebase";
    case Action::Shadow:
        return "Shadow";
    case Action::ReadOnlyConstant:
        return "ReadOnlyConstant";
    case Action::ApbSequencer:
        return "ApbSequencer";
    case Action::Reject:
        return "Reject";
    }
    return "UNKNOWN";
}

AddressPolicy::AddressPolicy(uint32_t apb_clk_hz)
    : apb_clk_hz_(apb_clk_hz)
{
    if (apb_clk_hz == 0) {
        throw std::invalid_argument("AddressPolicy: apb_clk_hz must be > 0");
    }
    build_ranges();
    // The SDK's I2c::set_i2c_clock programs CLK_CNT with 0x19, which on this block would be
    // 1.2 MHz at the Versa's 60 MHz APB clock; pin the alias write to 400 kHz (CLK_CNT 75)
    // whatever the SDK asked for. Captured by value: the policy may be copied or outlived by
    // the hook's owner.
    const uint32_t clk_cnt = i2c_clk_cnt();
    set_write_hook(I2C_ALIAS_CLK_CNT, [clk_cnt](uint32_t, uint32_t) { return clk_cnt; });
}

void AddressPolicy::build_ranges()
{
    // Ordered: first match wins, so the specific hololink-only words precede the BAR0 window.
    // The CTRL_EVT APB words and APB_RAM are never forwarded: the endpoint bitfile does not
    // execute sequencer programs written over PCIe, so the in-process software sequencer
    // (ApbSequencer) owns them.
    ranges_ = {
        // PTP: no PTP block on the endpoint; 0x188 reads "locked" so ptp_synchronize() returns.
        { 0x0100u, FPGA_PTP_SYNC_STAT, Action::Shadow, 0, 0, "ptp" },
        { FPGA_PTP_SYNC_STAT, FPGA_PTP_SYNC_STAT + 4u, Action::ReadOnlyConstant, 0, PTP_SYNC_STAT_LOCKED, "ptp_sync_stat" },
        { FPGA_PTP_SYNC_STAT + 4u, 0x0200u, Action::Shadow, 0, 0, "ptp" },
        // CTRL_EVT: event enables/clear are shadowed for the software sequencer ...
        { 0x0200u, 0x0210u, Action::Shadow, 0, 0, "ctrl_evt" },
        // ... host MAC/IP/UDP ports and FPGA UDP port are Ethernet-only ...
        { 0x0210u, CTRL_EVT_APB_INTERRUPT_EN, Action::Shadow, 0, 0, "ctrl_evt_host" },
        // ... APB interrupt enable / timeout are shadowed; a write of 1 to SW_EVENT arms the
        // deferred software-sequencer trigger ...
        { CTRL_EVT_APB_INTERRUPT_EN, CTRL_EVT_SW_EVENT, Action::Shadow, 0, 0, "ctrl_evt_apb" },
        { CTRL_EVT_SW_EVENT, CTRL_EVT_SW_EVENT + 4u, Action::ApbSequencer, 0, 0, "ctrl_evt_sw_event" },
        // ... the EVT VLAN word is Ethernet-only ...
        { 0x0230u, 0x0234u, Action::Shadow, 0, 0, "ctrl_evt_vlan" },
        // ... and the status words are written by the software sequencer.
        { CTRL_EVT_STAT, CTRL_EVT_FSM_STAT + 4u, Action::Shadow, 0, 0, "ctrl_evt_stat" },
        // APB RAM (vector table + sequencer programs): shadowed, read by the software sequencer.
        { APB_RAM_BEGIN, APB_RAM_END, Action::Shadow, 0, 0, "apb_ram" },
        // Everything else inside BAR0 -- including 0x04 SYS_RST -- reaches the FPGA.
        { 0x0u, BAR0_END, Action::ForwardIdentity, 0, 0, "bar0" },
        // Ethernet-only blocks: SIF (incl. I2S FIFO at 0x01200000) and HIF.
        { 0x0100'0000u, 0x0200'0000u, Action::Shadow, 0, 0, "sif" },
        { 0x0200'0000u, 0x0300'0000u, Action::Shadow, 0, 0, "hif" },
        // The SDK addresses the SPI/I2C blocks outside the 1 MiB BAR0 window; rebase.
        { SPI_ALIAS_BEGIN, SPI_ALIAS_END, Action::ForwardRebase, SPI_BAR0_BASE, 0, "spi_alias" },
        { I2C_ALIAS_BEGIN, I2C_ALIAS_END, Action::ForwardRebase, I2C_BAR0_BASE, 0, "i2c_alias" },
        // Absent blocks.
        { 0x4000'0000u, 0x7000'0000u, Action::Reject, 0, 0, "jesd" },
        { 0x7000'0000u, 0x8000'0000u, Action::Shadow, 0, 0, "vsync" },
        { 0x8000'0000u, ADDRESS_SPACE_END, Action::Reject, 0, 0, "i2s_apb" },
    };
}

void AddressPolicy::set_default_action(Action action)
{
    if (action != Action::Shadow && action != Action::Reject) {
        throw std::invalid_argument("AddressPolicy default action must be Shadow or Reject");
    }
    default_action_ = action;
}

const Range* AddressPolicy::lookup(uint32_t address) const
{
    for (const Range& range : ranges_) {
        if (range.contains(address)) {
            return &range;
        }
    }
    return nullptr;
}

Action AddressPolicy::action_for(uint32_t address) const
{
    const Range* range = lookup(address);
    return range ? range->action : default_action_;
}

uint32_t AddressPolicy::shadow_read(uint32_t address) const
{
    auto it = shadow_.find(address);
    return it == shadow_.end() ? 0u : it->second;
}

void AddressPolicy::shadow_write(uint32_t address, uint32_t value)
{
    shadow_[address] = value;
}

bool AddressPolicy::shadow_contains(uint32_t address) const
{
    return shadow_.find(address) != shadow_.end();
}

void AddressPolicy::set_write_hook(uint32_t address, WriteHook hook)
{
    write_hooks_[address] = std::move(hook);
}

void AddressPolicy::clear_write_hook(uint32_t address)
{
    write_hooks_.erase(address);
}

uint32_t AddressPolicy::bar0_offset(const Range& range, uint32_t address)
{
    if (range.action == Action::ForwardRebase) {
        return range.bar0_base + (address - range.begin);
    }
    return address;
}

Result AddressPolicy::from_status(PcieEndpointInterface::Status status)
{
    switch (status) {
    case PcieEndpointInterface::Status::SUCCESS:
        return { true, hololink::RESPONSE_SUCCESS };
    case PcieEndpointInterface::Status::INVALID_ADDRESS:
        return { false, hololink::RESPONSE_INVALID_ADDR };
    case PcieEndpointInterface::Status::READ_FAILED:
    case PcieEndpointInterface::Status::WRITE_FAILED:
    case PcieEndpointInterface::Status::NOT_ATTACHED:
        break;
    }
    return { false, hololink::RESPONSE_ERROR_GENERAL };
}

Result AddressPolicy::dispatch_write(uint32_t address, uint32_t value, PcieEndpointInterface& endpoint)
{
    if ((address & 3u) != 0u) {
        return { false, hololink::RESPONSE_INVALID_ADDR };
    }
    auto hook = write_hooks_.find(address);
    if (hook != write_hooks_.end()) {
        uint32_t rewritten = hook->second(address, value);
        if (rewritten != value) {
            HSB_LOG_DEBUG("write hook address={:#x} value={:#x} -> {:#x}", address, value, rewritten);
            value = rewritten;
        }
    }
    const Range* range = lookup(address);
    const Action action = range ? range->action : default_action_;
    switch (action) {
    case Action::ForwardIdentity:
        return from_status(endpoint.bar_write32(address, value));
    case Action::ForwardRebase:
        return from_status(endpoint.bar_write32(bar0_offset(*range, address), value));
    case Action::Shadow:
        shadow_write(address, value);
        return { true, hololink::RESPONSE_SUCCESS };
    case Action::ReadOnlyConstant:
        // Accepted and dropped, as a read-only hardware register would.
        return { true, hololink::RESPONSE_SUCCESS };
    case Action::ApbSequencer:
        shadow_write(address, value);
        if ((value & 1u) != 0u) {
            // Deferred: the batch-level caller (PcieHololink::write_uint32) fires it once every
            // pair of the batch has landed, so CTRL_EVT_APB_TIMEOUT written after SW_EVENT in
            // the same WriteData (I2c::i2c_transaction) bounds this run, as on an Ethernet HSB.
            HSB_LOG_TRACE("SW_EVENT armed at {:#x}; deferring the software sequencer", address);
            sequencer_pending_ = true;
        }
        return { true, hololink::RESPONSE_SUCCESS };
    case Action::Reject:
        break;
    }
    HSB_LOG_DEBUG("rejecting write address={:#x} ({})", address, range ? range->name : "unmapped");
    return { false, hololink::RESPONSE_INVALID_ADDR };
}

bool AddressPolicy::fire_pending_sequencer_trigger(PcieEndpointInterface& endpoint)
{
    if (!sequencer_pending_) {
        return false;
    }
    sequencer_pending_ = false;
    if (sequencer_running_) {
        // A program wrote 1 to CTRL_EVT_SW_EVENT while it was itself running; the flag was
        // cleared above and the outer fire completes once the program returns.
        HSB_LOG_WARN("SW_EVENT re-armed from inside a running sequencer program; ignored");
        return false;
    }
    if (!sequencer_trigger_) {
        HSB_LOG_WARN("SW_EVENT triggered but no software sequencer is installed");
        return false;
    }
    sequencer_running_ = true;
    try {
        sequencer_trigger_(endpoint);
    } catch (...) {
        sequencer_running_ = false;
        sequencer_pending_ = false;
        throw;
    }
    sequencer_running_ = false;
    if (sequencer_pending_) {
        HSB_LOG_WARN("SW_EVENT re-armed by the sequencer program itself; not re-entered");
        sequencer_pending_ = false;
    }
    return true;
}

Result AddressPolicy::dispatch_read(uint32_t address, PcieEndpointInterface& endpoint, uint32_t& value)
{
    if ((address & 3u) != 0u) {
        return { false, hololink::RESPONSE_INVALID_ADDR };
    }
    const Range* range = lookup(address);
    const Action action = range ? range->action : default_action_;
    switch (action) {
    case Action::ForwardIdentity:
        return from_status(endpoint.bar_read32(address, value));
    case Action::ForwardRebase:
        return from_status(endpoint.bar_read32(bar0_offset(*range, address), value));
    case Action::Shadow:
    case Action::ApbSequencer:
        value = shadow_read(address);
        return { true, hololink::RESPONSE_SUCCESS };
    case Action::ReadOnlyConstant:
        value = range->constant;
        return { true, hololink::RESPONSE_SUCCESS };
    case Action::Reject:
        break;
    }
    HSB_LOG_DEBUG("rejecting read address={:#x} ({})", address, range ? range->name : "unmapped");
    return { false, hololink::RESPONSE_INVALID_ADDR };
}

} // namespace hololink::module::hsb_pcie
