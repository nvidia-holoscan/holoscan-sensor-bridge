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

#include "cpnx_flash.hpp"

namespace hololink {

CpnxFlash::CpnxFlash(const std::string& context, std::shared_ptr<Hololink> hololink, uint32_t spi_controller_address)
    : WinbondW25q128jw(context, hololink, spi_controller_address)
{
    uint32_t chip_select = 0;
    uint32_t cpol = 1;
    uint32_t cpha = 1;
    uint32_t width = 1;
    uint32_t clock_divisor = 0;
    spi_ = get_spi(chip_select, cpol, cpha, width, clock_divisor);
}

std::vector<uint8_t> CpnxFlash::spi_command(const std::vector<uint8_t>& command_bytes,
    const std::vector<uint8_t>& write_bytes,
    uint32_t read_byte_count)
{
    auto result = spi_->spi_transaction(command_bytes, write_bytes, read_byte_count);
    if (result.size() < command_bytes.size()) {
        throw std::runtime_error("SPI transaction returned fewer bytes than command length");
    }
    return { result.begin() + static_cast<std::ptrdiff_t>(command_bytes.size()), result.end() };
}

} // namespace hololink
