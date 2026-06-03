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
 *
 * See README.md for detailed information.
 */

#ifndef REGISTER_MEMORY_HPP
#define REGISTER_MEMORY_HPP

#include <cstdint>
#include <shared_mutex>
#include <unordered_map>
#include <utility>

#include "address_memory.hpp"
#include "hsb_config.hpp"

namespace hololink::emulation {

class RegisterMemory : public AddressMemory {
public:
    RegisterMemory(const HSBConfiguration& configuration);
    ~RegisterMemory() override;

    int write(AddressValuePair& address_value) override;
    int read(AddressValuePair& address_value) override;
    int write_many(AddressValuePair* address_values, int num_addresses) override;
    int read_many(AddressValuePair* address_values, int num_addresses) override;
    int write_range(uint32_t start_address, uint32_t* values, int num_addresses, int stride = 2) override;
    int read_range(uint32_t start_address, uint32_t* values, int num_addresses, int stride = 2) override;

private:
    std::unordered_map<uint32_t, uint32_t> registers_ {};
    std::shared_mutex rwlock_ {};
};

} // namespace hololink::emulation

#endif
