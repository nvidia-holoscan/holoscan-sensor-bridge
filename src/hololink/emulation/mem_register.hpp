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
 * @class MemRegister mem_register.hpp
 * @brief Representation of the internal memory space of the HSBEmulator and its registers.
 * @note This currently implemented as a simple map, but the underlying implementation should be expected to change, though the public methods should remain available.
 */
class MemRegister {
public:
    /**
     * @brief Construct a new MemRegister object
     * @param configuration The configuration of the HSBEmulator.
     */
    MemRegister(const HSBConfiguration& configuration);
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

    /**
     * @brief Write a range of values to registers
     *
     * @param start_address The address of the first register to write
     * @param values A vector of values to write to the registers
     */
    void write_range(uint32_t start_address, const std::vector<uint32_t>& values);

    /**
     * @brief Read a range of values from registers
     *
     * @param start_address The address of the first register to read
     * @param num_addresses The number of registers to read
     * @return A vector of values read from the registers
     */
    std::vector<uint32_t> read_range(uint32_t start_address, uint32_t num_addresses);

private:
    std::unordered_map<uint32_t, uint32_t> registers_ {};
    std::shared_mutex rwlock_ {};
};

} // namespace hololink::emulation

#endif
