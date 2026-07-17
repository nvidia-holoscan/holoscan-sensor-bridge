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

#include "hsb_da326_flash_2511.hpp"
#include "../firmware_utils.hpp"
#include "cpnx_flash.hpp"

#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>

#include <iostream>

using hololink::load_firmware_file;

namespace hololink {

bool hsb_da326_flash_2511(const std::string& ip_address,
    const std::string& cpnx_path)
{
    std::cout << "[hsb_da326_flash_2511] Modern connection flash" << std::endl;
    std::cout << "  IP: " << ip_address << std::endl;
    std::cout << "  CPNX: " << cpnx_path << std::endl;

    try {
        std::cout << "[hsb_da326_flash_2511] Loading firmware file..." << std::endl;
        auto cpnx_content = load_firmware_file(cpnx_path);
        std::cout << "  CPNX size: " << cpnx_content.size() << " bytes" << std::endl;

        std::cout << "[hsb_da326_flash_2511] Connecting to device..." << std::endl;
        auto channel_metadata = Enumerator::find_channel(ip_address);
        auto hololink = Hololink::from_enumeration_metadata(channel_metadata);
        hololink->start();

        auto current_hsb_ip_version = hololink->get_hsb_ip_version();
        uint32_t cpnx_spi_bus = CPNX_SPI_BUS;
        if (current_hsb_ip_version > 0x2603) {
            cpnx_spi_bus = 0;
        }

        std::cout << "[hsb_da326_flash_2511] Current HSB IP version 0x" << std::hex << current_hsb_ip_version
                  << std::dec << " uses CPNX SPI bus " << cpnx_spi_bus << "." << std::endl;
        std::cout << "[hsb_da326_flash_2511] Programming CPNX on SPI bus " << cpnx_spi_bus << "..." << std::endl;
        CpnxFlash cpnx_flash("CPNX", hololink, cpnx_spi_bus);
        cpnx_flash.program(cpnx_content);

        std::cout << "[hsb_da326_flash_2511] Verifying CPNX..." << std::endl;
        if (!cpnx_flash.verify(cpnx_content)) {
            std::cerr << "[hsb_da326_flash_2511] CPNX verification failed!" << std::endl;
            hololink->stop();
            return false;
        }

        hololink->stop();
        std::cout << "[hsb_da326_flash_2511] Flash completed successfully!" << std::endl;
        std::cout << "[hsb_da326_flash_2511] Power cycle the device to apply changes." << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[hsb_da326_flash_2511] Error: " << e.what() << std::endl;
        return false;
    }
}

} // namespace hololink
