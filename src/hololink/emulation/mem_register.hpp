/**
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * See README.md for detailed information.
 */

#ifndef MEM_REGISTER_HPP
#define MEM_REGISTER_HPP

#include <initializer_list>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "hsb_config.hpp"

namespace hololink::emulation {

/**
 * @brief Simple wrapper for a map of 32-bit registers to 32-bit values
 * but also presets and initializes based on HSBConfiguration
 */
class MemRegister {
public:
    MemRegister() = default;
    MemRegister(HSBConfiguration* configuration);
    ~MemRegister();
    /**
     * @brief Write a value to a register
     *
     * @param address The address of the register
     * @param value The value to write to the register
     */
    void write(uint32_t address, uint32_t value);

    /**
     * @brief Convenience functions to write multiple values to registers under a single lock operations
     *
     * @param address_values A vector of pairs of register addresses and values to write. Accepts either a vector or a static initializer list.
     */
    void write_many(const std::vector<std::pair<uint32_t, uint32_t>>& address_values);
    // this really should take in a map and read from the values to the register
    void write_many(const std::initializer_list<std::pair<uint32_t, uint32_t>>& address_values);

    /**
     * @brief Read a value from a register
     *
     * @param address The address of the register
     * @return The value of the register
     */
    uint32_t read(uint32_t address);
    /**
     * @brief Convenience functions to read multiple values from registers under a single lock operations
     *
     * @param addresses A vector of register addresses to read. Accepts either a vector or a static initializer list.
     * @return A vector of values read from the registers
     */
    std::vector<uint32_t> read_many(const std::vector<uint32_t>& addresses);
    std::vector<uint32_t> read_many(const std::initializer_list<uint32_t>& addresses);

private:
    std::unordered_map<uint32_t, uint32_t> registers_ {};
    std::shared_mutex rwlock_ {};
};

} // namespace hololink::emulation

#endif
