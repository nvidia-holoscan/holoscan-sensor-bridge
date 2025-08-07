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

#include "clnx_flash.hpp"

namespace hololink {

ClnxFlash::ClnxFlash(const std::string& context, std::shared_ptr<Hololink> hololink, uint32_t spi_controller_address)
    : WinbondW25q128jw(context, hololink, spi_controller_address)
{
    { // variable scoping; solely for self-documentation
        uint32_t chip_select = 0;
        uint32_t cpol = 0;
        uint32_t cpha = 1;
        uint32_t width = 1;
        uint32_t clock_divisor = 0xF;
        slow_spi_ = get_spi(chip_select, cpol, cpha, width, clock_divisor);
    }
    { // variable scoping; solely for self-documentation
        uint32_t chip_select = 0;
        uint32_t cpol = 1;
        uint32_t cpha = 1;
        uint32_t width = 1;
        uint32_t clock_divisor = 0x4;
        fast_spi_ = get_spi(chip_select, cpol, cpha, width, clock_divisor);
    }
}

std::vector<uint8_t> ClnxFlash::spi_command(const std::vector<uint8_t>& command_bytes,
    const std::vector<uint8_t>& write_bytes,
    uint32_t read_byte_count)
{
    // Enable CLNX bridge for 1 transaction
    uint8_t transactions = 1;
    uint8_t spi_flash_forward_value = 0x1 | (transactions << 4);
    std::vector<uint8_t> request = {
        CLNX_WRITE,
        FLASH_FORWARD_EN_ADDRESS,
        spi_flash_forward_value
    };

    slow_spi_->spi_transaction({}, request, 0);

    // Execute the SPI flash command
    auto result = fast_spi_->spi_transaction(command_bytes, write_bytes, read_byte_count);
    if (result.size() < command_bytes.size()) {
        throw std::runtime_error("SPI response shorter than command length");
    }
    return { result.begin() + static_cast<std::ptrdiff_t>(command_bytes.size()), result.end() };
}

} // namespace hololink
