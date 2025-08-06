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

#include "set_ip.hpp"

#include <hololink/core/arp_wrapper.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/logging.hpp>
#include <hololink/core/networking.hpp>
#include <hololink/core/serializer.hpp>

#include <arpa/inet.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unistd.h>

namespace hololink::tools {

// Helper function to make BOOTP reply
static std::vector<uint8_t> make_bootp_reply(const Metadata& metadata, const std::string& new_device_ip, const std::string& local_ip)
{
    std::vector<uint8_t> reply(1000);
    hololink::core::Serializer serializer(reply.data(), reply.size());

    const uint8_t BOOTREPLY = 2;
    serializer.append_uint8(BOOTREPLY); // opcode
    serializer.append_uint8(std::stoul(metadata.get<std::string>("hardware_type").value_or("0")));
    serializer.append_uint8(std::stoul(metadata.get<std::string>("hardware_address_length").value_or("0")));
    uint8_t hops = 0;
    serializer.append_uint8(hops);
    serializer.append_uint32_be(std::stoul(metadata.get<std::string>("transaction_id").value_or("0")));
    serializer.append_uint16_be(std::stoul(metadata.get<std::string>("seconds").value_or("0")));
    uint16_t flags = 0;
    serializer.append_uint16_be(flags);
    uint32_t ciaddr = 0; // per bootp spec
    serializer.append_uint32_be(ciaddr);

    // Convert IP addresses to network byte order
    struct in_addr addr;
    inet_pton(AF_INET, new_device_ip.c_str(), &addr);
    serializer.append_buffer(reinterpret_cast<uint8_t*>(&addr), sizeof(addr));
    inet_pton(AF_INET, local_ip.c_str(), &addr);
    serializer.append_buffer(reinterpret_cast<uint8_t*>(&addr), sizeof(addr));

    uint32_t gateway_ip = 0;
    serializer.append_uint32_be(gateway_ip);

    // Hardware address
    std::string hw_addr_str = metadata.get<std::string>("hardware_address").value_or("");
    std::vector<uint8_t> hw_addr(16, 0);
    if (!hw_addr_str.empty()) {
        std::stringstream ss(hw_addr_str);
        std::string byte;
        size_t i = 0;
        while (std::getline(ss, byte, ',') && i < 16) {
            hw_addr[i++] = std::stoul(byte);
        }
    }
    serializer.append_buffer(hw_addr.data(), hw_addr.size());

    // Host name and file name (empty)
    std::vector<uint8_t> host_name(64, 0);
    std::vector<uint8_t> file_name(128, 0);
    std::vector<uint8_t> vendor_specific(64, 0);

    serializer.append_buffer(host_name.data(), host_name.size());
    serializer.append_buffer(file_name.data(), file_name.size());
    serializer.append_buffer(vendor_specific.data(), vendor_specific.size());

    // serializer.length will never exceed the size
    // passed into the constructor.
    reply.resize(serializer.length());
    return reply;
}

void set_ip(const std::unordered_map<std::string, std::string>& mac_ip_map, const std::string& interface, bool one_time)
{
    int timeout_s = one_time ? 45 : 0;
    if (!one_time) {
        std::cout << "Running in daemon mode; run with '--one-time' to exit after configuration." << std::endl;
    }

    auto enumerator = std::make_shared<Enumerator>(interface);
    std::unordered_map<std::string, bool> reported;

    auto start_time = std::chrono::steady_clock::now();

    auto callback = [&](Enumerator& enumerator, const std::vector<uint8_t>& packet, Metadata& metadata) -> bool {
        if (timeout_s > 0) {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= timeout_s) {
                return false;
            }
        }

        std::string mac_id = metadata.get<std::string>("mac_id").value_or("");
        std::string peer_ip = metadata.get<std::string>("peer_ip").value_or("");
        auto it = mac_ip_map.find(mac_id);

        if (it == mac_ip_map.end()) {
            return true;
        }

        const std::string& new_peer_ip = it->second;
        if (new_peer_ip == peer_ip) {
            if (!reported[mac_id]) {
                std::cout << "Found mac_id=" << mac_id << " using peer_ip=" << peer_ip << std::endl;
                reported[mac_id] = true;
            }
            if (one_time && reported.size() == mac_ip_map.size()) {
                return false;
            }
            return true;
        }

        // Update the IP
        std::string local_device = metadata.get<std::string>("interface").value_or("");
        std::string local_ip = metadata.get<std::string>("interface_address").value_or("");
        auto local_mac = hololink::core::local_mac(local_device);
        std::stringstream mac_str;
        for (size_t i = 0; i < local_mac.size(); ++i) {
            if (i > 0) {
                mac_str << ":";
            }
            mac_str << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(local_mac[i]) << std::dec;
        }

        if (new_peer_ip == local_ip) {
            throw std::runtime_error("Can't assign " + new_peer_ip + " to " + mac_id + " because that's the host IP address.");
        }

        std::cout << "Updating mac=" << mac_id << " from ip=" << peer_ip << " to new_ip=" << new_peer_ip << std::endl;

        // Set ARP entry
        int socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
        if (socket_fd < 0) {
            throw std::runtime_error("Failed to create socket");
        }

        try {
            int r = hololink::core::ArpWrapper::arp_set(socket_fd, local_device.c_str(), new_peer_ip.c_str(), mac_id.c_str());
            if (r != 0) {
                throw std::runtime_error("Unable to set IP address. This is likely due to not having "
                                         "CAP_NET_ADMIN permissions. Try running again as root.");
            }
        } catch (const std::exception& e) {
            close(socket_fd);
            std::cerr << e.what() << std::endl;
            return false;
        }
        close(socket_fd);

        // Send BOOTP reply
        auto reply = make_bootp_reply(metadata, new_peer_ip, local_ip);
        std::string reply_str(reply.begin(), reply.end());
        enumerator.send_bootp_reply(new_peer_ip, reply_str, const_cast<Metadata&>(metadata));

        if (reported.find(mac_id) != reported.end()) {
            reported.erase(mac_id);
        }

        return true;
    };

    enumerator->enumeration_packets(callback);
}

} // namespace hololink::tools
