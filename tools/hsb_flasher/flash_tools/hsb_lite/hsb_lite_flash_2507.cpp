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

#include "hsb_lite_flash_2507.hpp"
#include "../firmware_utils.hpp"
#include "clnx_flash.hpp"
#include "cpnx_flash.hpp"

#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>

#include <iostream>

namespace hololink {

bool hsb_lite_flash_2507(const std::string& ip_address,
    const std::string& clnx_path,
    const std::string& cpnx_path)
{
    std::cout << "[hsb_lite_flash_2507] Modern connection flash" << std::endl;
    std::cout << "  IP: " << ip_address << std::endl;
    std::cout << "  CLNX: " << clnx_path << std::endl;
    std::cout << "  CPNX: " << cpnx_path << std::endl;

    try {
        std::cout << "[hsb_lite_flash_2507] Loading firmware files..." << std::endl;
        auto clnx_content = load_firmware_file(clnx_path);
        auto cpnx_content = load_firmware_file(cpnx_path);
        std::cout << "  CLNX size: " << clnx_content.size() << " bytes" << std::endl;
        std::cout << "  CPNX size: " << cpnx_content.size() << " bytes" << std::endl;

        std::cout << "[hsb_lite_flash_2507] Connecting to device..." << std::endl;
        auto channel_metadata = Enumerator::find_channel(ip_address);
        auto hololink = Hololink::from_enumeration_metadata(channel_metadata);
        hololink->start();

        std::cout << "[hsb_lite_flash_2507] Programming CLNX..." << std::endl;
        ClnxFlash clnx_flash("CLNX", hololink, CLNX_SPI_BUS);
        clnx_flash.program(clnx_content);

        std::cout << "[hsb_lite_flash_2507] Verifying CLNX..." << std::endl;
        if (!clnx_flash.verify(clnx_content)) {
            std::cerr << "[hsb_lite_flash_2507] CLNX verification failed!" << std::endl;
            return false;
        }

        std::cout << "[hsb_lite_flash_2507] Programming CPNX..." << std::endl;
        CpnxFlash cpnx_flash("CPNX", hololink, CPNX_SPI_BUS);
        cpnx_flash.program(cpnx_content);

        std::cout << "[hsb_lite_flash_2507] Verifying CPNX..." << std::endl;
        if (!cpnx_flash.verify(cpnx_content)) {
            std::cerr << "[hsb_lite_flash_2507] CPNX verification failed!" << std::endl;
            return false;
        }

        std::cout << "[hsb_lite_flash_2507] Flash completed successfully!" << std::endl;
        std::cout << "[hsb_lite_flash_2507] Power cycle the device to apply changes." << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[hsb_lite_flash_2507] Error: " << e.what() << std::endl;
        return false;
    }
}

} // namespace hololink
