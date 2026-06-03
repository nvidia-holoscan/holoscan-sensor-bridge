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

#include "hsb_discovery.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <hololink/core/enumerator.hpp>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>

namespace hsb_flasher {

/**
 * Probe device version using raw UDP.
 */
static int64_t probe_version_udp(const std::string& ip_address)
{
    constexpr uint8_t RD_DWORD = 0x14;
    constexpr uint8_t REQUEST_FLAGS_ACK_REQUEST = 0x01;
    constexpr uint32_t HSB_IP_VERSION_ADDR = 0x80;
    constexpr int CONTROL_PORT = 8192;

    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        return 0;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(CONTROL_PORT);
    if (inet_pton(AF_INET, ip_address.c_str(), &addr.sin_addr) <= 0) {
        close(sock);
        return 0;
    }

    struct timeval tv = { 2, 0 };
    if (setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
        int err = errno;
        std::cerr << "[probe_version_udp] setsockopt(SO_RCVTIMEO) failed for sock: "
                  << strerror(err) << " (errno " << err << ")" << std::endl;
        close(sock);
        return 0;
    }

    uint8_t req[10];
    uint16_t seq = 0x0100;
    req[0] = RD_DWORD;
    req[1] = REQUEST_FLAGS_ACK_REQUEST;
    req[2] = (seq >> 8) & 0xFF;
    req[3] = seq & 0xFF;
    req[4] = 0;
    req[5] = 0;
    req[6] = (HSB_IP_VERSION_ADDR >> 24) & 0xFF;
    req[7] = (HSB_IP_VERSION_ADDR >> 16) & 0xFF;
    req[8] = (HSB_IP_VERSION_ADDR >> 8) & 0xFF;
    req[9] = HSB_IP_VERSION_ADDR & 0xFF;

    ssize_t ret = sendto(sock, req, sizeof(req), 0, (struct sockaddr*)&addr, sizeof(addr));
    if (ret < 0) {
        std::cerr << "[probe_version_udp] sendto failed: " << strerror(errno) << std::endl;
        close(sock);
        return 0;
    }
    if (ret != static_cast<ssize_t>(sizeof(req))) {
        std::cerr << "[probe_version_udp] sendto sent " << ret
                  << " bytes, expected " << sizeof(req) << std::endl;
        close(sock);
        return 0;
    }

    uint8_t resp[256];
    ssize_t len = recvfrom(sock, resp, sizeof(resp), 0, nullptr, nullptr);
    close(sock);

    if (len < 14) {
        return 0;
    }

    uint32_t value = (resp[10] << 24) | (resp[11] << 16) | (resp[12] << 8) | resp[13];
    return value & 0xFFFF;
}

/**
 * Extract MAC address from raw BOOTP packet.
 * BOOTP header: op(1), htype(1), hlen(1), hops(1), xid(4), secs(2), flags(2),
 *               ciaddr(4), yiaddr(4), siaddr(4), giaddr(4), chaddr(16)
 * MAC is at offset 28, length is at offset 2.
 */
static std::string extract_mac_from_packet(const std::vector<uint8_t>& packet)
{
    if (packet.size() < 44) {
        return "N/A";
    }
    uint8_t hlen = packet[2];
    if (hlen > 16 || hlen == 0) {
        return "N/A";
    }
    std::stringstream ss;
    for (int i = 0; i < hlen; ++i) {
        if (i > 0)
            ss << ":";
        ss << std::hex << std::setw(2) << std::setfill('0') << std::uppercase
           << static_cast<int>(packet[28 + i]);
    }
    return ss.str();
}

void BootpEnumerationProvider::enumerate(Callback callback, float timeout)
{
    auto enumerator = std::make_shared<hololink::Enumerator>("");
    enumerator->enumeration_packets(
        [&callback](hololink::Enumerator&, const std::vector<uint8_t>& packet,
            hololink::Metadata& metadata) -> bool {
            return callback(packet, metadata);
        },
        std::make_shared<hololink::Timeout>(timeout));
}

bool discover_device(hsb_flasher_context& context, IEnumerationProvider& provider)
{
    context.enumeration_metadata.clear();
    std::set<std::string> seen_macs;
    std::vector<uint8_t> matched_packet;

    auto callback = [&context, &seen_macs, &matched_packet](
                        const std::vector<uint8_t>& packet, hololink::Metadata& metadata) -> bool {
        std::string ip_address = metadata.get<std::string>("peer_ip").value_or("N/A");
        log_debug(context.log_level, "Discovered device at " + ip_address);

        if (ip_address == context.hololink_ip) {
            std::string mac = metadata.get<std::string>("mac_id").value_or(
                extract_mac_from_packet(packet));

            if (seen_macs.empty()) {
                matched_packet = packet;
                context.enumeration_metadata = metadata;
            }
            seen_macs.insert(mac);
        }

        return true;
    };

    provider.enumerate(callback, context.timeout);

    if (seen_macs.empty()) {
        return false;
    }

    if (seen_macs.size() > 1) {
        log_info("Error: Multiple devices (" + std::to_string(seen_macs.size()) + ") found at " + context.hololink_ip + "; cannot determine which to use");
        return false;
    }

    // Enrich metadata with fallback values for legacy devices that lack vendor data
    bool vendor_parse_ok = context.enumeration_metadata.get<int64_t>("data_plane").has_value();
    if (!vendor_parse_ok) {
        log_info("Vendor data unavailable, using fallback extraction...");

        if (!context.enumeration_metadata.get<std::string>("mac_id").has_value()) {
            context.enumeration_metadata["mac_id"] = extract_mac_from_packet(matched_packet);
        }

        if (!context.enumeration_metadata.get<std::string>("fpga_uuid").has_value()) {
            context.enumeration_metadata["fpga_uuid"] = std::string("N/A");
        }

        if (!context.enumeration_metadata.get<int64_t>("hsb_ip_version").has_value()) {
            std::string ip = context.enumeration_metadata.get<std::string>("peer_ip").value_or("N/A");
            if (ip != "N/A") {
                context.enumeration_metadata["hsb_ip_version"] = probe_version_udp(ip);
            }
        }
    }

    int64_t version = context.enumeration_metadata.get<int64_t>("hsb_ip_version").value_or(0);
    std::stringstream hex_stream;
    hex_stream << std::hex << std::setw(4) << std::setfill('0') << version;
    log_info("Found device: " + context.hololink_ip + " with version 0x" + hex_stream.str());

    return true;
}

bool discover_device(hsb_flasher_context& context)
{
    BootpEnumerationProvider provider;
    return discover_device(context, provider);
}

} // namespace hsb_flasher
