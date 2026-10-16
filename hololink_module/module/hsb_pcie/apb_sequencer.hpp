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

#ifndef HOLOLINK_MODULE_HSB_PCIE_APB_SEQUENCER_HPP
#define HOLOLINK_MODULE_HSB_PCIE_APB_SEQUENCER_HPP

/**
 * @file
 * @brief Software APB sequencer -- the only engine that executes SW_EVENT programs.
 *
 * The endpoint bitfile does not execute programs written to CTRL_BUS_EVT/EVT_RAM over PCIe,
 * so the policy shadows those words and this class runs the programs in-process.
 *
 * Port of the HSB emulator's SW_EVENT handler,
 * src/hololink/emulation/src/common/apb_events.cpp:126-245 (next_value / next_cmd /
 * sequence_write / sequence_read / sequence_poll / handle_apb_sw_event), operating on the
 * AddressPolicy's shadow copy of APB RAM (0x2000-0x2FFF). Program words are decoded exactly
 * as Hololink::Sequencer encodes them (hololink.cpp add_command/append). Every WR/RD/POLL
 * address inside a program is dispatched back through the policy, so an I2C alias such as
 * 0x03000200 reaches BAR0 0x40200 just as a direct SDK write would.
 *
 * The trigger installed by install() is not invoked from the SW_EVENT write itself: the
 * policy records it and EcbBridge::process fires it once the whole request has been applied
 * (AddressPolicy::fire_pending_sequencer_trigger), so the CTRL_EVT_APB_TIMEOUT the SDK
 * writes after SW_EVENT in the same WR_BLOCK is the bound do_poll() uses.
 */

#include <cstdint>
#include <string>

#include "address_policy.hpp"
#include "pcie_endpoint_interface.hpp"

namespace hololink::module::hsb_pcie {

class ApbSequencer {
public:
    /// Hololink::Event::SW_EVENT; the vector table slot is APB_RAM + 4 * SW_EVENT.
    static constexpr uint32_t SW_EVENT = 2;
    static constexpr uint32_t SW_EVENT_VECTOR = AddressPolicy::APB_RAM_BEGIN + 4 * SW_EVENT;
    /// Program-word terminator (the address slot of the final RD).
    static constexpr uint32_t DONE_COMMAND_ADDRESS = 0xFFFF'FFFFu;
    /// 2-bit op codes, Hololink::Sequencer::Op.
    enum Op : uint8_t {
        POLL = 0,
        WR = 1,
        RD = 2,
        ERROR = 3,
    };
    /// Set in shadow CTRL_EVT_FSM_STAT (0x284) when a program fails.
    static constexpr uint32_t FSM_ERROR_MASK = 1u << 28;
    /// Module-defined: set in shadow CTRL_EVT_STAT (0x280) when a program fails (bit 31) and
    /// the SW_EVENT bit (1 << SW_EVENT) once a program has run.
    static constexpr uint32_t STAT_APB_ERROR_MASK = 1u << 31;
    static constexpr uint32_t STAT_SW_EVENT_MASK = 1u << SW_EVENT;
    /// CTRL_EVT_APB_TIMEOUT units (hololink.hpp APB_TIMEOUT_SCALE).
    static constexpr double APB_TIMEOUT_UNIT_S = 51.2e-9;
    /// Poll bound when the SDK has not written CTRL_EVT_APB_TIMEOUT.
    static constexpr double DEFAULT_TIMEOUT_S = 1.0;
    /// Guard against a program that never reaches DONE.
    static constexpr unsigned MAX_COMMANDS = 4096;

    struct RunStats {
        bool ran { false };
        bool error { false };
        unsigned commands { 0 };
        uint32_t start_address { 0 };
        /// POLL bound (poll_timeout_s()) in effect when the run started.
        double poll_timeout_s { 0.0 };
        /// Number of POLL commands executed in this run.
        unsigned polls { 0 };
        std::string message;
    };

    ApbSequencer(AddressPolicy& policy, PcieEndpointInterface& endpoint);
    /// Uninstalls the trigger if install() was called, so a policy that outlives this
    /// sequencer never calls back into a destroyed object.
    ~ApbSequencer();

    ApbSequencer(const ApbSequencer&) = delete;
    ApbSequencer& operator=(const ApbSequencer&) = delete;

    /// Register this sequencer as the policy's SW_EVENT trigger (captures `this`). The
    /// policy arms it on a SW_EVENT write; AddressPolicy::fire_pending_sequencer_trigger()
    /// (called by EcbBridge::process after the request's pair loop) runs run_sw_event().
    void install();
    /// Remove the trigger installed by install(); no-op otherwise.
    void uninstall();
    bool installed() const { return installed_; }

    /// Run the program the SW_EVENT vector points at. Returns false on any error (also
    /// recorded in shadow 0x284/0x280 and in last_run()).
    bool run_sw_event();

    const RunStats& last_run() const { return last_run_; }

    /// POLL bound derived from shadow CTRL_EVT_APB_TIMEOUT (or DEFAULT_TIMEOUT_S when unset).
    double poll_timeout_s() const;

private:
    struct Cursor {
        uint32_t address; ///< next APB RAM word to consume
        uint32_t cmd_word;
        uint8_t cmd_bit;
    };

    bool next_value(Cursor& cursor, uint32_t& value);
    bool next_cmd(Cursor& cursor, uint8_t& cmd);
    bool do_write(Cursor& cursor, uint32_t register_address);
    bool do_read(Cursor& cursor, uint32_t register_address);
    bool do_poll(Cursor& cursor, uint32_t register_address);
    bool fail(const std::string& message);

    AddressPolicy& policy_;
    PcieEndpointInterface& endpoint_;
    RunStats last_run_;
    bool installed_ { false };
};

} // namespace hololink::module::hsb_pcie

#endif /* HOLOLINK_MODULE_HSB_PCIE_APB_SEQUENCER_HPP */
