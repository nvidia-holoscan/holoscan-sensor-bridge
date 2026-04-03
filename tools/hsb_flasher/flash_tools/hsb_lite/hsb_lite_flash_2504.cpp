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

#include "hsb_lite_flash_2504.hpp"
#include "../firmware_utils.hpp"
#include "clnx_flash.hpp"
#include "cpnx_flash.hpp"

#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>

#include <iostream>

namespace hololink {

bool hsb_lite_flash_2504(const std::string& ip_address,
    const std::string& clnx_path,
    const std::string& cpnx_path,
    int64_t current_version,
    const std::string& fpga_uuid,
    const std::string& serial_number)
{
    std::cout << "[hsb_lite_flash_2504] Legacy connection flash" << std::endl;
    std::cout << "  IP: " << ip_address << std::endl;
    std::cout << "  CLNX: " << clnx_path << std::endl;
    std::cout << "  CPNX: " << cpnx_path << std::endl;
    std::cout << "  Version: 0x" << std::hex << current_version << std::dec << std::endl;
    std::cout << "  Using legacy connection mode with workarounds" << std::endl;

    try {
        std::cout << "[hsb_lite_flash_2504] Loading firmware files..." << std::endl;
        auto clnx_content = load_firmware_file(clnx_path);
        auto cpnx_content = load_firmware_file(cpnx_path);
        std::cout << "  CLNX size: " << clnx_content.size() << " bytes" << std::endl;
        std::cout << "  CPNX size: " << cpnx_content.size() << " bytes" << std::endl;

        std::cout << "[hsb_lite_flash_2504] Creating legacy connection..." << std::endl;

        Metadata manual_metadata;
        manual_metadata["control_port"] = static_cast<int64_t>(8192);
        manual_metadata["hsb_ip_version"] = current_version;
        manual_metadata["peer_ip"] = ip_address;
        manual_metadata["sequence_number_checking"] = static_cast<int64_t>(0);
        manual_metadata["serial_number"] = serial_number + "_legacy";
        manual_metadata["fpga_uuid"] = fpga_uuid;

        // Legacy workarounds
        manual_metadata["ptp_enable"] = static_cast<int64_t>(0); // Skip PTP init
        manual_metadata["block_enable"] = static_cast<int64_t>(0); // Use slower individual writes
        manual_metadata["legacy_no_ack"] = static_cast<int64_t>(0); // Still request ACKs

        // Board ID workaround for very old devices
        manual_metadata["board_id"] = static_cast<int64_t>(HOLOLINK_100G_BOARD_ID);

        // Configure data plane and sensor
        DataChannel::use_data_plane_configuration(manual_metadata, 0);
        DataChannel::use_sensor(manual_metadata, 0);

        auto hololink = Hololink::from_enumeration_metadata(manual_metadata);
        hololink->start();
        std::cout << "  Connected to device at " << ip_address << std::endl;

        std::cout << "[hsb_lite_flash_2504] Programming CLNX..." << std::endl;
        ClnxFlash clnx_flash("CLNX", hololink, CLNX_SPI_BUS);
        clnx_flash.program(clnx_content);

        std::cout << "[hsb_lite_flash_2504] Verifying CLNX..." << std::endl;
        if (!clnx_flash.verify(clnx_content)) {
            std::cerr << "[hsb_lite_flash_2504] CLNX verification failed!" << std::endl;
            return false;
        }

        std::cout << "[hsb_lite_flash_2504] Programming CPNX..." << std::endl;
        CpnxFlash cpnx_flash("CPNX", hololink, CPNX_SPI_BUS);
        cpnx_flash.program(cpnx_content);

        std::cout << "[hsb_lite_flash_2504] Verifying CPNX..." << std::endl;
        if (!cpnx_flash.verify(cpnx_content)) {
            std::cerr << "[hsb_lite_flash_2504] CPNX verification failed!" << std::endl;
            return false;
        }

        std::cout << "[hsb_lite_flash_2504] Flash completed successfully!" << std::endl;
        std::cout << "[hsb_lite_flash_2504] Power cycle the device to apply changes." << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[hsb_lite_flash_2504] Error: " << e.what() << std::endl;
        return false;
    }
}

} // namespace hololink
