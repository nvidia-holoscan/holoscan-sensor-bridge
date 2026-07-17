/*
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
 */

#include "micron_mt25ql256aba.hpp"

#include <algorithm>

#include <fmt/format.h>
#include <hololink/core/logging_internal.hpp>

namespace hololink {

std::shared_ptr<Hololink::Spi> get_traditional_spi(Hololink& hololink, uint32_t spi_address, uint32_t chip_select,
    uint32_t clock_divisor, uint32_t cpol, uint32_t cpha, uint32_t width);

MicronMT25QL256ABA::MicronMT25QL256ABA(const std::string& context, std::shared_ptr<Hololink> hololink, uint32_t spi_address)
    : context_(context)
    , hololink_(hololink)
    , hsb_ip_version_(hololink->get_hsb_ip_version())
    , datecode_(hololink->get_fpga_date())
    , spi_address_(spi_address)
{
}

std::shared_ptr<Hololink::Spi> MicronMT25QL256ABA::get_spi(uint32_t chip_select, uint32_t cpol, uint32_t cpha,
    uint32_t width, uint32_t clock_divisor)
{
    bool traditional_spi = (hsb_ip_version_ < 0x2503) || ((hsb_ip_version_ == 0x2503) && (datecode_ < 0xE57E47B7));

    if (traditional_spi) {
        constexpr uint32_t CPNX_SPI_CTRL = 0x03000200;

        uint32_t spi_ctrl_address;
        if (spi_address_ == CPNX_SPI_BUS) {
            spi_ctrl_address = CPNX_SPI_CTRL;
        } else {
            throw std::runtime_error(fmt::format("Unexpected spi_address=0x{:x}", spi_address_));
        }

        return get_traditional_spi(*hololink_, spi_ctrl_address, chip_select, clock_divisor, cpol, cpha, width);
    }
    return hololink_->get_spi(spi_address_, chip_select, clock_divisor, cpol, cpha, width);
}

void MicronMT25QL256ABA::wait_for_spi_ready()
{
    hololink::Timeout timeout(.5);

    while (!timeout.expired()) {
        auto result = spi_command({ STATUS }, {}, 1);
        if ((result[0] & 1) == 0) {
            return;
        }
        HSB_LOG_DEBUG("{}: BUSY, got r={:02x}", context_, fmt::join(result, " "));
    }
    throw std::runtime_error(fmt::format("{}: SPI busy timeout", context_));
}

void MicronMT25QL256ABA::check_id()
{
    spi_command({ ENTER_4BYTE_ADDR });
    auto manufacturer_device_id = spi_command({ JEDEC_ID }, {}, 3);
    HSB_LOG_INFO("{}: manufacturer_device_id={:02x} {:02x} {:02x}",
        context_, manufacturer_device_id[0], manufacturer_device_id[1], manufacturer_device_id[2]);

    std::vector<uint8_t> expected = {
        MICRON,
        static_cast<uint8_t>(MT25QL256ABA_IQ >> 8),
        static_cast<uint8_t>(MT25QL256ABA_IQ & 0xFF)
    };

    if (manufacturer_device_id != expected) {
        throw std::runtime_error(fmt::format("{}: Device ID mismatch", context_));
    }
}

void MicronMT25QL256ABA::status(const std::string& message)
{
    HSB_LOG_INFO("{}: {}", context_, message);
}

void MicronMT25QL256ABA::program(const std::vector<uint8_t>& content)
{
    check_id();
    size_t content_size = content.size();
    spi_command({ ENTER_4BYTE_ADDR });

    for (size_t erase_address = 0; erase_address < content_size; erase_address += ERASE_SIZE) {
        HSB_LOG_DEBUG("{}: erase address=0x{:X}", context_, erase_address);

        wait_for_spi_ready();
        spi_command({ WRITE_ENABLE });

        std::vector<uint8_t> page_erase = {
            BLOCK_ERASE,
            static_cast<uint8_t>((erase_address >> 24) & 0xFF),
            static_cast<uint8_t>((erase_address >> 16) & 0xFF),
            static_cast<uint8_t>((erase_address >> 8) & 0xFF),
            static_cast<uint8_t>(erase_address & 0xFF)
        };
        spi_command(page_erase);

        for (size_t address = erase_address; address < std::min(content_size, erase_address + ERASE_SIZE); address += BLOCK_SIZE) {
            if ((address & 0xFFFF) == 0) {
                status(fmt::format("address=0x{:X}", address));
            }

            wait_for_spi_ready();
            spi_command({ WRITE_ENABLE });

            std::vector<uint8_t> command_bytes = {
                PAGE_PROGRAM,
                static_cast<uint8_t>((address >> 24) & 0xFF),
                static_cast<uint8_t>((address >> 16) & 0xFF),
                static_cast<uint8_t>((address >> 8) & 0xFF),
                static_cast<uint8_t>(address & 0xFF)
            };

            size_t block_end = std::min(address + BLOCK_SIZE, content_size);
            std::vector<uint8_t> write_data(content.begin() + address, content.begin() + block_end);
            spi_command(command_bytes, write_data);
        }
    }

    wait_for_spi_ready();
    spi_command({ ENABLE_RESET });
    spi_command({ RESET });
    wait_for_spi_ready();
}

bool MicronMT25QL256ABA::verify(const std::vector<uint8_t>& content)
{
    check_id();
    bool ok = true;
    HSB_LOG_INFO("Content size={}", content.size());
    spi_command({ ENTER_4BYTE_ADDR });

    for (size_t address = 0; address < content.size(); address += BLOCK_SIZE) {
        if ((address & 0xFFFF) == 0) {
            status(fmt::format("verify address=0x{:X}", address));
        }

        size_t block_end = std::min(address + BLOCK_SIZE, content.size());
        std::vector<uint8_t> original_content(content.begin() + address, content.begin() + block_end);

        wait_for_spi_ready();
        std::vector<uint8_t> command_bytes = {
            READ,
            static_cast<uint8_t>((address >> 24) & 0xFF),
            static_cast<uint8_t>((address >> 16) & 0xFF),
            static_cast<uint8_t>((address >> 8) & 0xFF),
            static_cast<uint8_t>(address & 0xFF)
        };

        auto flash_content = spi_command(command_bytes, {}, original_content.size());
        if (flash_content != original_content) {
            HSB_LOG_INFO("{}: verify failed, address=0x{:X}", context_, address);
            HSB_LOG_DEBUG("  expected: {}", fmt::join(original_content, " "));
            HSB_LOG_DEBUG("       got: {}", fmt::join(flash_content, " "));
            ok = false;
        }
    }

    return ok;
}

} // namespace hololink
