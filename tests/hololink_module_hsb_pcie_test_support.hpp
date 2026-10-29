/*
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
 *
 * Shared helpers for the hsb_pcie hardware-free tests: a hand encoder for software APB
 * sequencer programs in the Hololink::Sequencer word layout, and the (address, value)
 * pairs that load one into shadow APB RAM the way Hololink::Sequencer::enable() does.
 */

#ifndef TESTS_HOLOLINK_MODULE_HSB_PCIE_TEST_SUPPORT_HPP
#define TESTS_HOLOLINK_MODULE_HSB_PCIE_TEST_SUPPORT_HPP

#include <cstdint>
#include <utility>
#include <vector>

#include "apb_sequencer.hpp"

namespace hsb_pcie_test {

using Pairs = std::vector<std::pair<uint32_t, uint32_t>>;

/// Default program location: the SW_EVENT sequencer's slot in APB RAM (hololink.cpp).
constexpr uint32_t PROGRAM_LOCATION = 0x2800;

// Hand-encode a program in the Hololink::Sequencer layout: command word first, then per
// command ADDR [DATA [DATA]]; 2 bits per op, LSB first.
struct Program {
    using ApbSequencer = hololink::module::hsb_pcie::ApbSequencer;

    std::vector<uint32_t> words { 0 };
    unsigned command_index { 0 };
    unsigned command_bit { 0 };

    void op(uint32_t code)
    {
        if (command_bit == 32) {
            command_index = words.size();
            words.push_back(0);
            command_bit = 0;
        }
        words[command_index] |= (code << command_bit);
        command_bit += 2;
    }
    void wr(uint32_t address, uint32_t value)
    {
        op(ApbSequencer::WR);
        words.push_back(address);
        words.push_back(value);
    }
    unsigned rd(uint32_t address, uint32_t initial = 0)
    {
        op(ApbSequencer::RD);
        words.push_back(address);
        words.push_back(initial);
        return words.size() - 1;
    }
    void poll(uint32_t address, uint32_t mask, uint32_t match)
    {
        op(ApbSequencer::POLL);
        words.push_back(address);
        words.push_back(match);
        words.push_back(mask);
    }
    void done()
    {
        op(ApbSequencer::RD);
        words.push_back(0xFFFFFFFF);
    }

    /// The writes that install this program at `location`: the words, the SW_EVENT vector
    /// table slot (configure_apb_event) and the APB interrupt enable.
    Pairs load_pairs(uint32_t location = PROGRAM_LOCATION) const
    {
        Pairs pairs;
        uint32_t address = location;
        for (uint32_t word : words) {
            pairs.emplace_back(address, word);
            address += 4;
        }
        pairs.emplace_back(ApbSequencer::SW_EVENT_VECTOR, location);
        pairs.emplace_back(0x224, 1u << ApbSequencer::SW_EVENT);
        return pairs;
    }
};

} // namespace hsb_pcie_test

#endif /* TESTS_HOLOLINK_MODULE_HSB_PCIE_TEST_SUPPORT_HPP */
