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

// Software SW_EVENT sequencer; port of src/hololink/emulation/src/common/apb_events.cpp:126-245.

#include "apb_sequencer.hpp"

#include <chrono>
#include <thread>

#include <fmt/format.h>

#include <hololink/module/logging.hpp>

namespace hololink::module::hsb_pcie {

namespace {
    constexpr uint32_t REG_SIZE = 4u;
    constexpr uint8_t CMD_BITS = 2u;
    constexpr uint8_t CMD_MASK = 0x3u;
    constexpr uint8_t CMD_WORD_BITS = REG_SIZE * 8u;
} // namespace

ApbSequencer::ApbSequencer(AddressPolicy& policy, PcieEndpointInterface& endpoint)
    : policy_(policy)
    , endpoint_(endpoint)
{
}

ApbSequencer::~ApbSequencer()
{
    uninstall();
}

void ApbSequencer::install()
{
    policy_.set_sequencer_trigger([this](PcieEndpointInterface&) { run_sw_event(); });
    installed_ = true;
}

void ApbSequencer::uninstall()
{
    if (installed_) {
        policy_.set_sequencer_trigger(nullptr);
        installed_ = false;
    }
}

double ApbSequencer::poll_timeout_s() const
{
    uint32_t units = policy_.shadow_read(AddressPolicy::CTRL_EVT_APB_TIMEOUT);
    if (units == 0) {
        return DEFAULT_TIMEOUT_S;
    }
    return units * APB_TIMEOUT_UNIT_S;
}

bool ApbSequencer::fail(const std::string& message)
{
    last_run_.error = true;
    last_run_.message = message;
    policy_.shadow_write(AddressPolicy::CTRL_EVT_FSM_STAT,
        policy_.shadow_read(AddressPolicy::CTRL_EVT_FSM_STAT) | FSM_ERROR_MASK);
    policy_.shadow_write(AddressPolicy::CTRL_EVT_STAT,
        policy_.shadow_read(AddressPolicy::CTRL_EVT_STAT) | STAT_APB_ERROR_MASK);
    HSB_LOG_ERROR("APB software sequencer: {}", message);
    return false;
}

// Returns the APB RAM word at cursor.address and advances by one word (emulator next_value).
bool ApbSequencer::next_value(Cursor& cursor, uint32_t& value)
{
    if (cursor.address < AddressPolicy::APB_RAM_BEGIN || cursor.address >= AddressPolicy::APB_RAM_END
        || (cursor.address & 3u) != 0u) {
        return fail(fmt::format("program ran outside APB RAM at {:#x}", cursor.address));
    }
    value = policy_.shadow_read(cursor.address);
    cursor.address += REG_SIZE;
    return true;
}

// Next 2-bit command; pulls a fresh command word when the current one is exhausted
// (emulator next_cmd, mirroring Hololink::Sequencer::add_command).
bool ApbSequencer::next_cmd(Cursor& cursor, uint8_t& cmd)
{
    if (cursor.cmd_bit >= CMD_WORD_BITS) {
        cursor.cmd_bit = 0;
        if (!next_value(cursor, cursor.cmd_word)) {
            return false;
        }
    }
    cmd = static_cast<uint8_t>((cursor.cmd_word >> cursor.cmd_bit) & CMD_MASK);
    cursor.cmd_bit = static_cast<uint8_t>(cursor.cmd_bit + CMD_BITS);
    return true;
}

// WR: consumes DATA; writes it to register_address through the policy.
bool ApbSequencer::do_write(Cursor& cursor, uint32_t register_address)
{
    uint32_t value = 0;
    if (!next_value(cursor, value)) {
        return false;
    }
    Result result = policy_.dispatch_write(register_address, value, endpoint_);
    if (!result.ok) {
        return fail(fmt::format("WR {:#x} <- {:#x} failed (response_code={})", register_address, value, result.response_code));
    }
    HSB_LOG_TRACE("seq WR {:#x} <- {:#x}", register_address, value);
    return true;
}

// RD: reads register_address through the policy and stores the result in the DATA slot.
bool ApbSequencer::do_read(Cursor& cursor, uint32_t register_address)
{
    uint32_t value = 0;
    Result result = policy_.dispatch_read(register_address, endpoint_, value);
    if (!result.ok) {
        return fail(fmt::format("RD {:#x} failed (response_code={})", register_address, result.response_code));
    }
    const uint32_t slot = cursor.address;
    uint32_t discard = 0;
    if (!next_value(cursor, discard)) { // validates + advances past the slot
        return false;
    }
    policy_.shadow_write(slot, value);
    HSB_LOG_TRACE("seq RD {:#x} -> {:#x} into {:#x}", register_address, value, slot);
    return true;
}

// POLL: consumes MATCH then MASK (note the order, per Hololink::Sequencer::poll) and spins
// until (value & mask) == match or the APB timeout elapses.
bool ApbSequencer::do_poll(Cursor& cursor, uint32_t register_address)
{
    uint32_t match = 0;
    uint32_t mask = 0;
    if (!next_value(cursor, match) || !next_value(cursor, mask)) {
        return false;
    }
    const double timeout_s = poll_timeout_s();
    last_run_.poll_timeout_s = timeout_s;
    last_run_.polls++;
    const auto deadline = std::chrono::steady_clock::now()
        + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(timeout_s));
    while (true) {
        uint32_t value = 0;
        Result result = policy_.dispatch_read(register_address, endpoint_, value);
        if (!result.ok) {
            return fail(fmt::format("POLL {:#x} read failed (response_code={})", register_address, result.response_code));
        }
        if ((value & mask) == match) {
            HSB_LOG_TRACE("seq POLL {:#x} matched {:#x}/{:#x}", register_address, match, mask);
            return true;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return fail(fmt::format("POLL {:#x} mask={:#x} match={:#x} timed out after {} s (last value {:#x})",
                register_address, mask, match, timeout_s, value));
        }
        std::this_thread::yield();
    }
}

bool ApbSequencer::run_sw_event()
{
    last_run_ = RunStats {};
    last_run_.ran = true;
    // Bound every POLL of this run will use; PcieHololink::write_uint32 has already applied the whole
    // request that armed us, so an APB_TIMEOUT written after SW_EVENT is in effect here.
    last_run_.poll_timeout_s = poll_timeout_s();
    // Clear the error indications from a previous run; leave the rest of the words alone.
    policy_.shadow_write(AddressPolicy::CTRL_EVT_FSM_STAT,
        policy_.shadow_read(AddressPolicy::CTRL_EVT_FSM_STAT) & ~FSM_ERROR_MASK);
    policy_.shadow_write(AddressPolicy::CTRL_EVT_STAT,
        (policy_.shadow_read(AddressPolicy::CTRL_EVT_STAT) & ~STAT_APB_ERROR_MASK) | STAT_SW_EVENT_MASK);

    // The emulator hard-codes APB_RAM + 0x800; here the vector table decides, which is what
    // Hololink::configure_apb_event(SW_EVENT, location) programs (and NullSequencer's 0x2080).
    Cursor cursor { policy_.shadow_read(SW_EVENT_VECTOR), 0, 0 };
    last_run_.start_address = cursor.address;
    if (cursor.address == 0) {
        return fail("SW_EVENT vector is not programmed");
    }
    HSB_LOG_DEBUG("APB software sequencer: running program at {:#x}", cursor.address);

    // Layout (apb_events.cpp): a command word, then per command ADDR [DATA [DATA]]; every
    // 16 commands a fresh command word is interleaved. DONE is an address of 0xFFFFFFFF.
    if (!next_value(cursor, cursor.cmd_word)) {
        return false;
    }
    uint8_t cmd = 0;
    uint32_t address = 0;
    if (!next_cmd(cursor, cmd) || !next_value(cursor, address)) {
        return false;
    }
    while (address != DONE_COMMAND_ADDRESS) {
        if (last_run_.commands >= MAX_COMMANDS) {
            return fail(fmt::format("program at {:#x} exceeded {} commands without DONE", last_run_.start_address, MAX_COMMANDS));
        }
        bool ok = false;
        switch (cmd) {
        case WR:
            ok = do_write(cursor, address);
            break;
        case RD:
            ok = do_read(cursor, address);
            break;
        case POLL:
            ok = do_poll(cursor, address);
            break;
        default:
            ok = fail(fmt::format("invalid opcode {} at program word {:#x}", cmd, cursor.address - REG_SIZE));
            break;
        }
        if (!ok) {
            return false;
        }
        last_run_.commands++;
        if (!next_cmd(cursor, cmd) || !next_value(cursor, address)) {
            return false;
        }
    }
    HSB_LOG_DEBUG("APB software sequencer: {} commands completed", last_run_.commands);
    return true;
}

} // namespace hololink::module::hsb_pcie
