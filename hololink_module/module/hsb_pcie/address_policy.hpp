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

#ifndef HOLOLINK_MODULE_HSB_PCIE_ADDRESS_POLICY_HPP
#define HOLOLINK_MODULE_HSB_PCIE_ADDRESS_POLICY_HPP

/**
 * @file
 * @brief Maps the HSB register addresses the SDK emits onto the PCIe endpoint's BAR0.
 *
 * The module's address space is BAR0 offsets (0x00000-0xFFFFF) plus the HSB aliases the
 * SDK still emits (I2C at 0x03000200, SPI at 0x03000000, Ethernet-only blocks). This is the
 * in-process replacement for the translation the hsb_pcie emulator daemon performs in
 * hsb_pcie/driver/src/hsb_pcie/hsb_pcie_emulator.cpp (the I2C/SPI rebase) -- and, unlike the
 * daemon, it also decides which hololink-only registers are shadowed in host memory instead
 * of being forwarded to an FPGA that does not implement them.
 *
 * The range table is the one in the design plan, section B.
 */

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "pcie_endpoint_interface.hpp"

namespace hololink::module::hsb_pcie {

enum class Action {
    ForwardIdentity, ///< BAR0 offset == HSB address; forwarded untouched
    ForwardRebase, ///< forwarded to bar0_base + (address - range.begin)
    Shadow, ///< host-side RAM; reads of never-written words return 0
    ReadOnlyConstant, ///< reads return `constant`; writes are accepted and dropped
    ApbSequencer, ///< Shadow, plus a write of 1 arms the software sequencer (fired by the bridge)
    Reject, ///< RESPONSE_INVALID_ADDR
};

const char* to_string(Action action);

struct Range {
    uint32_t begin; ///< first address covered
    uint64_t end; ///< one past the last address covered (may be 0x1'0000'0000)
    Action action;
    uint32_t bar0_base; ///< ForwardRebase: BAR0 offset that `begin` maps to
    uint32_t constant; ///< ReadOnlyConstant: the value every read returns
    const char* name;

    bool contains(uint32_t address) const { return address >= begin && address < end; }
};

/// Outcome of one dispatched access, in ECB terms.
struct Result {
    bool ok;
    uint32_t response_code; ///< hololink::RESPONSE_* (0 success, 3 invalid address, ...)
};

/**
 * @brief Ordered range table + shadow RAM + hooks.
 *
 * Not internally synchronized: every access arrives under Hololink::execute_mutex_ (the
 * bridge) or from the sequencer running inside such an access.
 *
 * APB sequencer programs are always executed by the in-process software sequencer: the
 * endpoint bitfile does not run programs written to CTRL_BUS_EVT/EVT_RAM over PCIe, so the
 * CTRL_EVT APB words and APB_RAM are shadowed here and CTRL_EVT_SW_EVENT is the trigger.
 *
 * Software-sequencer trigger timing: a write of 1 to CTRL_EVT_SW_EVENT does NOT run the
 * sequencer inline. The SDK's I2c::i2c_transaction (hololink.cpp) queues SW_EVENT=1,
 * SW_EVENT=0 and only then CTRL_EVT_APB_TIMEOUT in the same WR_BLOCK; on an Ethernet HSB the
 * sequencer runs asynchronously so the later APB_TIMEOUT write lands before a long POLL
 * expires. In-process that cannot happen, so dispatch_write() only records a pending
 * trigger and the request-level caller (EcbBridge::process) fires it once every pair of the
 * request has been applied -- see fire_pending_sequencer_trigger().
 */
class AddressPolicy {
public:
    // HSB aliases (hololink.hpp I2C_CTRL / SPI_CTRL) and their BAR0 homes (pcie_regs.hpp).
    static constexpr uint32_t SPI_ALIAS_BEGIN = 0x0300'0000u;
    static constexpr uint32_t SPI_ALIAS_END = 0x0300'0200u;
    static constexpr uint32_t SPI_BAR0_BASE = 0x0004'0000u;
    static constexpr uint32_t I2C_ALIAS_BEGIN = 0x0300'0200u;
    static constexpr uint32_t I2C_ALIAS_END = 0x0300'0400u;
    static constexpr uint32_t I2C_BAR0_BASE = 0x0004'0200u;
    static constexpr uint32_t I2C_REG_CLK_CNT = 0x0Cu;
    static constexpr uint32_t I2C_ALIAS_CLK_CNT = I2C_ALIAS_BEGIN + I2C_REG_CLK_CNT;

    // CTRL_EVT / APB_RAM words owned by the software sequencer (shadowed, never forwarded).
    static constexpr uint32_t CTRL_EVT_APB_INTERRUPT_EN = 0x0224u;
    static constexpr uint32_t CTRL_EVT_APB_TIMEOUT = 0x0228u;
    static constexpr uint32_t CTRL_EVT_SW_EVENT = 0x022Cu;
    static constexpr uint32_t CTRL_EVT_STAT = 0x0280u;
    static constexpr uint32_t CTRL_EVT_FSM_STAT = 0x0284u;
    static constexpr uint32_t APB_RAM_BEGIN = 0x2000u;
    static constexpr uint32_t APB_RAM_END = 0x3000u;

    /// PTP sync status the SDK polls for lock; the constant keeps ptp_synchronize() short.
    static constexpr uint32_t FPGA_PTP_SYNC_STAT = 0x0188u;
    static constexpr uint32_t PTP_SYNC_STAT_LOCKED = 0xFu;

    /// SCL rate the I2C block is pinned to; the SDK's set_i2c_clock asks for 2 MHz (0x19).
    static constexpr uint32_t I2C_SCL_HZ = 400000u;
    /// APB clock the I2C divider is computed against: the CertusPro-NX Versa's 60 MHz. The
    /// constructor takes it as a parameter so tests can exercise the arithmetic, but
    /// PcieHololinkV1::create_backing() always builds the policy with this default.
    static constexpr uint32_t DEFAULT_APB_CLK_HZ = 60000000u;

    /// clk_cnt divider producing approximately `scl_hz` (i2c.sv: SCL ~= apb_clk / (2 * clk_cnt));
    /// same arithmetic as pcie_regs.hpp clk_cnt_for. 400 kHz -> 75 at 60 MHz.
    static constexpr uint32_t clk_cnt_for(uint32_t scl_hz, uint32_t apb_clk_hz = DEFAULT_APB_CLK_HZ)
    {
        return scl_hz == 0 ? 1u : (apb_clk_hz + scl_hz) / (2u * scl_hz);
    }

    /// Rewrites the value of a write before it is forwarded.
    using WriteHook = std::function<uint32_t(uint32_t address, uint32_t value)>;
    /// Runs the software sequencer; armed by a write of 1 to CTRL_EVT_SW_EVENT and invoked
    /// from fire_pending_sequencer_trigger().
    using SequencerTrigger = std::function<void(PcieEndpointInterface& endpoint)>;

    /// @param apb_clk_hz the board's APB clock, which sets the I2C CLK_CNT this policy pins
    ///        (i2c_clk_cnt()) and that PcieHololink programs; 0 is rejected.
    explicit AddressPolicy(uint32_t apb_clk_hz = DEFAULT_APB_CLK_HZ);

    /// The APB clock this policy was built for.
    uint32_t apb_clk_hz() const { return apb_clk_hz_; }
    /// CLK_CNT for I2C_SCL_HZ at apb_clk_hz(): what the alias write hook pins and what
    /// PcieHololink::program_i2c_timing() writes.
    uint32_t i2c_clk_cnt() const { return clk_cnt_for(I2C_SCL_HZ, apb_clk_hz_); }

    // ---- table -------------------------------------------------------------------------
    /// First range containing `address`, or nullptr when it falls through to default_action().
    const Range* lookup(uint32_t address) const;
    /// Effective action for `address` (default_action() when unmatched).
    Action action_for(uint32_t address) const;
    const std::vector<Range>& ranges() const { return ranges_; }

    /// Shadow (default) or Reject.
    Action default_action() const { return default_action_; }
    void set_default_action(Action action);

    // ---- shadow RAM ----------------------------------------------------------------------
    uint32_t shadow_read(uint32_t address) const;
    void shadow_write(uint32_t address, uint32_t value);
    bool shadow_contains(uint32_t address) const;
    void shadow_clear() { shadow_.clear(); }

    // ---- hooks ---------------------------------------------------------------------------
    void set_write_hook(uint32_t address, WriteHook hook);
    void clear_write_hook(uint32_t address);
    void set_sequencer_trigger(SequencerTrigger trigger) { sequencer_trigger_ = std::move(trigger); }
    bool has_sequencer_trigger() const { return static_cast<bool>(sequencer_trigger_); }

    /// True when a dispatch_write() armed the sequencer and nothing has fired it yet.
    bool sequencer_trigger_pending() const { return sequencer_pending_; }
    /**
     * Fire the pending trigger (if any) and clear it. Every caller that dispatches writes
     * outside EcbBridge::process must call this once its batch of writes is complete.
     * A trigger re-armed while the sequencer itself is running is dropped (with a warning)
     * rather than re-entered.
     * @returns true when the trigger ran.
     */
    bool fire_pending_sequencer_trigger(PcieEndpointInterface& endpoint);
    /// Forget a pending trigger without running it.
    void clear_pending_sequencer_trigger() { sequencer_pending_ = false; }

    // ---- dispatch ------------------------------------------------------------------------
    Result dispatch_write(uint32_t address, uint32_t value, PcieEndpointInterface& endpoint);
    Result dispatch_read(uint32_t address, PcieEndpointInterface& endpoint, uint32_t& value);

    /// BAR0 offset a ForwardIdentity/ForwardRebase address lands on (for diagnostics/tests).
    static uint32_t bar0_offset(const Range& range, uint32_t address);

private:
    void build_ranges();
    static Result from_status(PcieEndpointInterface::Status status);

    uint32_t apb_clk_hz_ { DEFAULT_APB_CLK_HZ };
    Action default_action_ { Action::Shadow };
    std::vector<Range> ranges_;
    std::unordered_map<uint32_t, uint32_t> shadow_;
    std::unordered_map<uint32_t, WriteHook> write_hooks_;
    SequencerTrigger sequencer_trigger_;
    bool sequencer_pending_ { false };
    bool sequencer_running_ { false };
};

} // namespace hololink::module::hsb_pcie

#endif /* HOLOLINK_MODULE_HSB_PCIE_ADDRESS_POLICY_HPP */
