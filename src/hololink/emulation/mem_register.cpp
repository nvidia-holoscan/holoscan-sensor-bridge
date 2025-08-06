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

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "hololink/core/hololink.hpp"

#include "mem_register.hpp"

namespace hololink::emulation {

#define HSB_EMULATOR_DATE 20250608

MemRegister::MemRegister(HSBConfiguration* configuration)
{
    registers_[hololink::HSB_IP_VERSION] = configuration->hsb_ip_version;
    registers_[hololink::FPGA_DATE] = HSB_EMULATOR_DATE;
}

// register cleanup is handled by child classes
MemRegister::~MemRegister() { }
void MemRegister::write(uint32_t address, uint32_t value)
{
    std::unique_lock<std::shared_mutex> lock(rwlock_);
    registers_[address] = value;
}
void MemRegister::write_many(const std::vector<std::pair<uint32_t, uint32_t>>& address_values)
{
    std::unique_lock<std::shared_mutex> lock(rwlock_);
    for (auto& address_value : address_values) {
        registers_[address_value.first] = address_value.second;
    }
}
void MemRegister::write_many(const std::initializer_list<std::pair<uint32_t, uint32_t>>& address_values)
{
    std::unique_lock<std::shared_mutex> lock(rwlock_);
    for (auto& address_value : address_values) {
        registers_[address_value.first] = address_value.second;
    }
}
uint32_t MemRegister::read(uint32_t address)
{
    std::shared_lock<std::shared_mutex> lock(rwlock_);
    // get value from registers or 0 if not found/created
    auto it = registers_.find(address);
    if (it != registers_.end()) {
        return it->second;
    }
    return 0;
}

std::vector<uint32_t> MemRegister::read_many(const std::vector<uint32_t>& addresses)
{
    std::shared_lock<std::shared_mutex> lock(rwlock_);
    std::vector<uint32_t> values(addresses.size());
    for (size_t i = 0; i < addresses.size(); i++) {
        auto it = registers_.find(addresses[i]);
        if (it != registers_.end()) {
            values[i] = it->second;
        } else {
            values[i] = 0;
        }
    }
    return values;
}
std::vector<uint32_t> MemRegister::read_many(const std::initializer_list<uint32_t>& addresses)
{
    std::shared_lock<std::shared_mutex> lock(rwlock_);
    std::vector<uint32_t> values(addresses.size());
    size_t idx = 0;
    for (const uint32_t& addr : addresses) {
        auto it = registers_.find(addr);
        if (it != registers_.end()) {
            values[idx] = it->second;
        } else {
            values[idx] = 0;
        }
        ++idx;
    }

    return values;
}

} // namespace hololink::emulation