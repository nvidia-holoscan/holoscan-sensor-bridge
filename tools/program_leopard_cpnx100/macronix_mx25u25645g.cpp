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

#include "macronix_mx25u25645g.hpp"

#include <fmt/format.h>
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/timeout.hpp>

namespace hololink {

// This routine is not expected to be used anywhere
// except for programming tools, so there isn't a
// header file you can include for it.
std::shared_ptr<Hololink::Spi> get_traditional_spi(Hololink& hololink, uint32_t spi_address, uint32_t chip_select,
    uint32_t clock_divisor, uint32_t cpol, uint32_t cpha, uint32_t width);

MacronixMx25u25645g::MacronixMx25u25645g(const std::string& context, std::shared_ptr<Hololink> hololink, uint32_t spi_address)
    : context_(context)
    , hololink_(hololink)
    , hsb_ip_version_(hololink->get_hsb_ip_version())
    , datecode_(hololink->get_fpga_date())
    , spi_address_(spi_address)
{
}

std::shared_ptr<Hololink::Spi> MacronixMx25u25645g::get_spi(uint32_t chip_select, uint32_t cpol, uint32_t cpha,
    uint32_t width, uint32_t clock_divisor)
{
    HSB_LOG_INFO("{}: hsb_ip_version={:#X}", context_, hsb_ip_version_);

    bool traditional_spi = (hsb_ip_version_ < 0x2503) || ((hsb_ip_version_ == 0x2503) && (datecode_ < 0xE57E47B7));

    if (traditional_spi) {
        HSB_LOG_INFO("Using traditional SPI controller.");
        uint32_t spi_address = 0x03000000;
        return get_traditional_spi(*hololink_, spi_address, chip_select, clock_divisor, cpol, cpha, width);
    } else {
        return hololink_->get_spi(spi_address_, chip_select, clock_divisor, cpol, cpha, width);
    }
}

void MacronixMx25u25645g::wait_for_spi_ready()
{
    Timeout timeout(60.0f);
    while (true) {
        auto r = spi_command({ STATUS }, {}, 1);
        if ((r[0] & 1) == 0) {
            return;
        }
        HSB_LOG_DEBUG("{}: BUSY, got r={:02X}", context_, fmt::join(r, " "));
        if (timeout.expired()) {
            throw std::runtime_error(fmt::format("{} timeout waiting for SPI ready.", context_));
        }
    }
}

void MacronixMx25u25645g::check_id()
{
    auto manufacturer_device_id = spi_command({ JEDEC_ID }, {}, 3);
    HSB_LOG_INFO("{}: manufacturer_device_id={:02X}", context_, fmt::join(manufacturer_device_id, " "));

    std::vector<uint8_t> expected = {
        MACRONIX,
        static_cast<uint8_t>(MX25U25645G >> 8),
        static_cast<uint8_t>(MX25U25645G & 0xFF)
    };

    if (manufacturer_device_id != expected) {
        throw std::runtime_error(fmt::format("Unexpected flash JEDEC-ID: {:02X}", fmt::join(manufacturer_device_id, " ")));
    }
}

void MacronixMx25u25645g::status(const std::string& message)
{
    HSB_LOG_INFO("{}: {}", context_, message);
}

void MacronixMx25u25645g::program(const std::vector<uint8_t>& content)
{
    check_id();
    size_t content_size = content.size();

    for (size_t erase_address = 0; erase_address < content_size; erase_address += ERASE_SIZE) {
        HSB_LOG_DEBUG("{}: erase address=0x{:X}", context_, erase_address);
        wait_for_spi_ready();
        spi_command({ WRITE_ENABLE });

        std::vector<uint8_t> page_erase = {
            BLOCK_ERASE,
            static_cast<uint8_t>((erase_address >> 16) & 0xFF),
            static_cast<uint8_t>((erase_address >> 8) & 0xFF),
            static_cast<uint8_t>((erase_address >> 0) & 0xFF)
        };
        spi_command(page_erase);

        for (size_t address = erase_address;
             address < std::min(content_size, erase_address + ERASE_SIZE);
             address += BLOCK_SIZE) {

            // Provide some status
            if ((address & 0xFFFF) == 0) {
                status(fmt::format("address=0x{:X}", address));
            }

            // Write this page
            wait_for_spi_ready();
            spi_command({ WRITE_ENABLE });

            std::vector<uint8_t> command_bytes = {
                PAGE_PROGRAM,
                static_cast<uint8_t>((address >> 16) & 0xFF),
                static_cast<uint8_t>((address >> 8) & 0xFF),
                static_cast<uint8_t>((address >> 0) & 0xFF)
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

bool MacronixMx25u25645g::verify(const std::vector<uint8_t>& content)
{
    check_id();
    bool ok = true;

    for (size_t address = 0; address < content.size(); address += BLOCK_SIZE) {
        // Provide some status
        if ((address & 0xFFFF) == 0) {
            status(fmt::format("verify address=0x{:X}", address));
        }

        // original_content, on the last page, will be shorter than BLOCK_SIZE
        size_t block_end = std::min(address + BLOCK_SIZE, content.size());
        std::vector<uint8_t> original_content(content.begin() + address, content.begin() + block_end);

        // Fetch this page from flash
        wait_for_spi_ready();
        std::vector<uint8_t> command_bytes = {
            READ,
            static_cast<uint8_t>((address >> 16) & 0xFF),
            static_cast<uint8_t>((address >> 8) & 0xFF),
            static_cast<uint8_t>((address >> 0) & 0xFF)
        };

        auto flash_content = spi_command(command_bytes, {}, original_content.size());

        // Check it
        if (flash_content != original_content) {
            HSB_LOG_INFO("{}: verify failed, address=0x{:X}", context_, address);
            ok = false;
        }
    }

    return ok;
}

} // namespace hololink