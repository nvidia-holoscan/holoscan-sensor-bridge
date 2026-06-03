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

#ifndef ADDRESS_MEMORY_HPP
#define ADDRESS_MEMORY_HPP

#include <cstdint>

#include <utility>

#include "hsb_config.hpp"

namespace hololink::emulation {

/**
 * @class AddressMemory address_memory.hpp
 * @brief Representation of the internal memory space of the HSBEmulator and its registers.
 * @note This currently implemented as a simple map, but the underlying implementation should be expected to change, though the public methods should remain available.
 */
class AddressMemory {
public:
    virtual ~AddressMemory() = default;

    /**
     * @brief Write a value to a register
     *
     * @param address_value Pair of address and value to write
     * @return 0 on success, 1 on failure
     */
    virtual int write(AddressValuePair& address_value) = 0;

    /**
     * @brief Read a value from a register
     *
     * @param address_value Pair of address and value to read
     * @return 0 on success, 1 on failure
     */
    virtual int read(AddressValuePair& address_value) = 0;

    /**
     * @brief Convenience function to write multiple values to registers under a single lock
     *
     * @param address_values Pointer to array of pairs of register addresses and values to write
     * @param num_addresses Number of address-value pairs
     * @return 0 on success, 1 on failure
     */
    virtual int write_many(AddressValuePair* address_values, int num_addresses) = 0;

    /**
     * @brief Convenience function to read multiple values from registers under a single lock
     *
     * @param address_values Pointer to array of pairs of register addresses and values to read (caller must allocate at least num_addresses elements)
     * @param num_addresses Number of addresses to read
     * @return 0 on success, 1 on failure
     */
    virtual int read_many(AddressValuePair* address_values, int num_addresses) = 0;

    /**
     * @brief Write a range of values to registers. Allow optimizations if it is known that the addresses are contiguous.
     *
     * @param address_values Pointer to array of pairs of register addresses and values to write (caller must allocate at least num_addresses elements)
     * @param num_addresses The number of addresses to write
     * @return 0 on success, 1 on failure
     */
    virtual int write_range(uint32_t start_address, uint32_t* values, int num_addresses, int stride = 2) = 0;

    /**
     * @brief Read a range of values from registers. Allow optimizations if it is known that the addresses are contiguous.
     *
     * @param address_values Pointer to array of pairs of register addresses and values to read (caller must allocate at least num_addresses elements)
     * @param num_addresses The number of addresses to read
     * @return 0 on success, 1 on failure
     */
    virtual int read_range(uint32_t start_address, uint32_t* values, int num_addresses, int stride = 2) = 0;
};

} // namespace hololink::emulation

#endif
