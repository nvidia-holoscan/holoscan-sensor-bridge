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

#ifndef WINBOND_W25Q128JW_HPP
#define WINBOND_W25Q128JW_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <hololink/core/hololink.hpp>

namespace hololink {

class WinbondW25q128jw {
public:
    // SPI commands
    static constexpr uint8_t PAGE_PROGRAM = 0x2;
    static constexpr uint8_t READ = 0x3;
    static constexpr uint8_t STATUS = 0x5;
    static constexpr uint8_t WRITE_ENABLE = 0x6;
    static constexpr uint8_t WRITE_STATUS_2 = 0x31;
    static constexpr uint8_t ENABLE_RESET = 0x66;
    static constexpr uint8_t RESET = 0x99;
    static constexpr uint8_t JEDEC_ID = 0x9F;
    static constexpr uint8_t BLOCK_ERASE = 0xD8;

    // Device identifiers
    static constexpr uint8_t WINBOND = 0xEF;
    static constexpr uint16_t W25Q128JW_IQ = 0x6018;

    // Configuration
    static constexpr uint8_t QE = 0x2;
    static constexpr size_t BLOCK_SIZE = 128; // bytes
    static constexpr size_t ERASE_SIZE = 64 * 1024; // bytes

    WinbondW25q128jw(const std::string& context, std::shared_ptr<Hololink> hololink, uint32_t spi_address);
    virtual ~WinbondW25q128jw() = default;

    void program(const std::vector<uint8_t>& content);
    bool verify(const std::vector<uint8_t>& content);

protected:
    virtual std::vector<uint8_t> spi_command(const std::vector<uint8_t>& command_bytes,
        const std::vector<uint8_t>& write_bytes = {},
        uint32_t read_byte_count = 0)
        = 0;

    std::shared_ptr<Hololink::Spi> get_spi(uint32_t chip_select, uint32_t cpol, uint32_t cpha,
        uint32_t width, uint32_t clock_divisor);
    void wait_for_spi_ready();
    void check_id();
    void status(const std::string& message);

    std::string context_;
    std::shared_ptr<Hololink> hololink_;
    uint32_t hsb_ip_version_;
    uint32_t datecode_;
    uint32_t spi_address_;
};

} // namespace hololink

#endif // WINBOND_W25Q128JW_HPP
