/*
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
 */

#ifndef CLNX_FLASH_HPP
#define CLNX_FLASH_HPP

#include <memory>
#include <vector>

#include "winbond_w25q128jw.hpp"

namespace hololink {

class ClnxFlash : public WinbondW25q128jw {
public:
    // CLNX registers
    static constexpr uint8_t CLNX_WRITE = 0x1;
    static constexpr uint8_t FLASH_FORWARD_EN_ADDRESS = 6;

    ClnxFlash(const std::string& context, std::shared_ptr<Hololink> hololink, uint32_t spi_controller_address);

protected:
    std::vector<uint8_t> spi_command(const std::vector<uint8_t>& command_bytes,
        const std::vector<uint8_t>& write_bytes = {},
        uint32_t read_byte_count = 0) override;

private:
    std::shared_ptr<Hololink::Spi> slow_spi_;
    std::shared_ptr<Hololink::Spi> fast_spi_;
};

} // namespace hololink

#endif // CLNX_FLASH_HPP
