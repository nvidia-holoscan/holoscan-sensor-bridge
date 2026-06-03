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

#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

#include "register_memory.hpp"

namespace hololink::emulation {

RegisterMemory::RegisterMemory(const HSBConfiguration& configuration)
    : AddressMemory()
{
    registers_[HSB_IP_VERSION] = configuration.hsb_ip_version;
    registers_[FPGA_DATE] = HSB_EMULATOR_DATE;
}

RegisterMemory::~RegisterMemory() = default;

int RegisterMemory::write(AddressValuePair& address_value)
{
    std::unique_lock<std::shared_mutex> lock(rwlock_);
    registers_[address_value.address] = address_value.value;
    return 0;
}

int RegisterMemory::read(AddressValuePair& address_value)
{
    std::shared_lock<std::shared_mutex> lock(rwlock_);
    auto it = registers_.find(address_value.address);
    address_value.value = (it != registers_.end()) ? it->second : 0;
    return 0;
}

int RegisterMemory::write_many(AddressValuePair* address_values, int num_addresses)
{
    if (!address_values || num_addresses < 0) {
        return 1;
    }
    std::unique_lock<std::shared_mutex> lock(rwlock_);
    for (int i = 0; i < num_addresses; i++) {
        registers_[address_values[i].address] = address_values[i].value;
    }
    return 0;
}

int RegisterMemory::read_many(AddressValuePair* address_values, int num_addresses)
{
    if (!address_values || num_addresses < 0) {
        return 1;
    }
    std::shared_lock<std::shared_mutex> lock(rwlock_);
    for (int i = 0; i < num_addresses; i++) {
        auto it = registers_.find(address_values[i].address);
        address_values[i].value = (it != registers_.end()) ? it->second : 0;
    }
    return 0;
}

int RegisterMemory::write_range(uint32_t start_address, uint32_t* values, int num_addresses, int stride)
{
    if (!values || num_addresses < 0) {
        return 1;
    }
    std::unique_lock<std::shared_mutex> lock(rwlock_);
    for (int i = 0; i < num_addresses; i++) {
        registers_[start_address] = values[i * stride];
        start_address += sizeof(uint32_t);
    }
    return 0;
}

int RegisterMemory::read_range(uint32_t start_address, uint32_t* values, int num_addresses, int stride)
{
    if (!values || num_addresses < 0) {
        return 1;
    }
    std::shared_lock<std::shared_mutex> lock(rwlock_);
    for (uint32_t i = 0; i < num_addresses; i++) {
        auto it = registers_.find(start_address);
        values[i * stride] = (it != registers_.end()) ? it->second : 0;
        start_address += sizeof(uint32_t);
    }
    return 0;
}

} // namespace hololink::emulation
