/**
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
 *
 * See README.md for detailed information.
 */

#include <cassert>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "net.hpp"

namespace hololink::emulation {

// taken from hololink/core/networking.cpp with logging removed
std::tuple<std::string, std::string, hololink::core::MacAddress> local_ip_and_mac(
    const std::string& destination_ip, uint32_t port)
{
    // We need a port number for the connect call to work, but because it's
    // SOCK_DGRAM, there's no actual traffic sent.
    int s = socket(AF_INET, SOCK_DGRAM, 0);
    if (s < 0) {
        fprintf(stderr, "Failed to create socket: %d - %s\n", errno, strerror(errno));
        throw std::runtime_error("Failed to create socket");
    }

    // Start with a map of IP address to interfaces.
    std::map<in_addr_t, std::string> interface_by_ip;
    // First, find out how many interfaces there are.
    ifconf ifconf_request {};
    if (ioctl(s, SIOCGIFCONF, &ifconf_request) < 0) {
        fprintf(stderr, "ioctl failed with errno=%d: \"%s\"\n", errno, strerror(errno));
        close(s);
        throw std::runtime_error("ioctl failed");
    }
    assert(ifconf_request.ifc_len > 0);
    //
    std::vector<ifreq> ifreq_buffer(ifconf_request.ifc_len / sizeof(ifreq));
    ifconf_request.ifc_ifcu.ifcu_req = ifreq_buffer.data();
    if (ioctl(s, SIOCGIFCONF, &ifconf_request) < 0) {
        fprintf(stderr, "ioctl failed with errno=%d: \"%s\"\n", errno, strerror(errno));
        close(s);
        throw std::runtime_error("ioctl failed");
    }
    assert(static_cast<size_t>(ifconf_request.ifc_len) == ifreq_buffer.size() * sizeof(ifreq));
    assert(ifconf_request.ifc_ifcu.ifcu_req == ifreq_buffer.data());
    for (auto&& req : ifreq_buffer) {
        const std::string name(req.ifr_ifrn.ifrn_name);
        const in_addr ip = ((struct sockaddr_in*)&req.ifr_ifru.ifru_addr)->sin_addr;
        interface_by_ip[ip.s_addr] = name;
    }

    // datagram sockets, when connected, will only
    // do I/O with the address they're connected to.
    // Once it's connected, getsockname() will tell
    // us the IP we're using on our side.
    timeval timeout {};
    if (setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) < 0) {
        fprintf(stderr, "setsockopt failed with errno=%d: \"%s\"\n", errno, strerror(errno));
        close(s);
        throw std::runtime_error("setsockopt failed");
    }

    sockaddr_in address {};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    if (inet_pton(AF_INET, destination_ip.c_str(), &address.sin_addr) == 0) {
        fprintf(stderr, "Failed to convert address %s\n", destination_ip.c_str());
        close(s);
        throw std::runtime_error("Failed to convert address");
    }
    if (connect(s, (sockaddr*)&address, sizeof(address)) < 0) {
        fprintf(stderr, "connection to %s failed with errno=%d: \"%s\"\n", destination_ip.c_str(), errno, strerror(errno));
        close(s);
        throw std::runtime_error("connect failed");
    }
    sockaddr_in ip {};
    ip.sin_family = AF_UNSPEC;
    socklen_t ip_len = sizeof(ip);
    if (getsockname(s, (sockaddr*)&ip, &ip_len) < 0) {
        fprintf(stderr, "getsockname failed with errno=%d: \"%s\"\n", errno, strerror(errno));
        close(s);
        throw std::runtime_error("getsockname failed");
    }
    const std::string binterface = interface_by_ip[ip.sin_addr.s_addr];

    ifreq ifhwaddr_request {};
    std::strncpy(ifhwaddr_request.ifr_ifrn.ifrn_name, binterface.c_str(),
        sizeof(ifhwaddr_request.ifr_ifrn.ifrn_name));
    if (ioctl(s, SIOCGIFHWADDR, &ifhwaddr_request) < 0) {
        fprintf(stderr, "ioctl failed with errno=%d: \"%s\"\n", errno, strerror(errno));
        close(s);
        throw std::runtime_error("ioctl failed");
    }
    hololink::core::MacAddress mac;
    static_assert(mac.max_size() <= sizeof(ifhwaddr_request.ifr_ifru.ifru_addr.sa_data));
    std::copy(ifhwaddr_request.ifr_ifru.ifru_addr.sa_data,
        ifhwaddr_request.ifr_ifru.ifru_addr.sa_data + mac.max_size(),
        mac.begin());
    auto result = std::tuple<std::string, std::string, hololink::core::MacAddress> { inet_ntoa(ip.sin_addr), binterface, mac };
    close(s);
    return result;
}

}
