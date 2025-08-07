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

#include "enumerate.hpp"

#include <hololink/core/enumerator.hpp>
#include <iostream>

namespace hololink::tools {

void enumerate(const std::string& interface, int timeout)
{
    auto enumerator = std::make_shared<hololink::Enumerator>(interface);
    auto callback = [](Enumerator& enumerator,
                        const std::vector<uint8_t>& packet,
                        Metadata& metadata) -> bool {
        std::string mac_id = metadata.get<std::string>("mac_id").value_or("N/A");
        int64_t hsb_ip_version = metadata.get<int64_t>("hsb_ip_version").value_or(0);
        int64_t fpga_crc = metadata.get<int64_t>("fpga_crc").value_or(0);
        std::string ip_address = metadata.get<std::string>("peer_ip").value_or("N/A");
        std::string fpga_uuid = metadata.get<std::string>("fpga_uuid").value_or("N/A");
        std::string serial_number = metadata.get<std::string>("serial_number").value_or("N/A");
        std::string interface = metadata.get<std::string>("interface").value_or("N/A");
        std::string board_description = metadata.get<std::string>("board_description").value_or("N/A");

        std::cout << "mac_id=" << mac_id
                  << " hsb_ip_version=0x" << std::hex << hsb_ip_version
                  << " fpga_crc=0x" << std::hex << fpga_crc << std::dec
                  << " ip_address=" << ip_address
                  << " fpga_uuid=" << fpga_uuid
                  << " serial_number=" << serial_number
                  << " interface=" << interface << " board=" << board_description
                  << std::endl;
        return true;
    };

    if (timeout > 0) {
        enumerator->enumeration_packets(callback, std::make_shared<hololink::Timeout>(timeout));
    } else {
        enumerator->enumeration_packets(callback);
    }
}

} // namespace hololink::tools
