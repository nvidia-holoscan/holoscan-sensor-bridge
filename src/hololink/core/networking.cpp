/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "networking.hpp"

#include <arpa/inet.h>
#include <assert.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>

#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "logging_internal.hpp"

namespace hololink::core {

size_t round_up(size_t value, size_t alignment)
{
    if (alignment == 0) {
        throw std::runtime_error("Alignment must be greater than zero");
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

std::tuple<std::string, std::string, MacAddress> local_ip_and_mac(
    const std::string& destination_ip, uint32_t port)
{
    // We need a port number for the connect call to work, but because it's
    // SOCK_DGRAM, there's no actual traffic sent.
    UniqueFileDescriptor s(socket(AF_INET, SOCK_DGRAM, 0));
    if (!s) {
        throw std::runtime_error("Failed to create socket");
    }

    // Start with a map of IP address to interfaces.
    std::map<in_addr_t, std::string> interface_by_ip;
    // First, find out how many interfaces there are.
    ifconf ifconf_request {};
    if (ioctl(s.get(), SIOCGIFCONF, &ifconf_request) < 0) {
        throw std::runtime_error(
            fmt::format("ioctl failed with errno={}: \"{}\"", errno, strerror(errno)));
    }
    assert(ifconf_request.ifc_len > 0);
    //
    std::vector<ifreq> ifreq_buffer(ifconf_request.ifc_len / sizeof(ifreq));
    ifconf_request.ifc_ifcu.ifcu_req = ifreq_buffer.data();
    if (ioctl(s.get(), SIOCGIFCONF, &ifconf_request) < 0) {
        throw std::runtime_error(
            fmt::format("ioctl failed with errno={}: \"{}\"", errno, strerror(errno)));
    }
    assert(static_cast<size_t>(ifconf_request.ifc_len) == ifreq_buffer.size() * sizeof(ifreq));
    assert(ifconf_request.ifc_ifcu.ifcu_req == ifreq_buffer.data());
    for (auto&& req : ifreq_buffer) {
        const std::string name(req.ifr_ifrn.ifrn_name);
        const in_addr ip = ((struct sockaddr_in*)&req.ifr_ifru.ifru_addr)->sin_addr;
        HSB_LOG_TRACE("name={} ip={}", name, inet_ntoa(ip));
        interface_by_ip[ip.s_addr] = name;
    }

    // datagram sockets, when connected, will only
    // do I/O with the address they're connected to.
    // Once it's connected, getsockname() will tell
    // us the IP we're using on our side.
    timeval timeout {};
    if (setsockopt(s.get(), SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) < 0) {
        throw std::runtime_error(
            fmt::format("setsockopt failed with errno={}: \"{}\"", errno, strerror(errno)));
    }

    sockaddr_in address {};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    if (inet_pton(AF_INET, destination_ip.c_str(), &address.sin_addr) == 0) {
        throw std::runtime_error(fmt::format("Failed to convert address {}", destination_ip));
    }
    if (connect(s.get(), (sockaddr*)&address, sizeof(address)) < 0) {
        throw std::runtime_error(
            fmt::format("connect failed with errno={}: \"{}\"", errno, strerror(errno)));
    }
    sockaddr_in ip {};
    ip.sin_family = AF_UNSPEC;
    socklen_t ip_len = sizeof(ip);
    if (getsockname(s.get(), (sockaddr*)&ip, &ip_len) < 0) {
        throw std::runtime_error(
            fmt::format("getsockname failed with errno={}: \"{}\"", errno, strerror(errno)));
    }
    const std::string binterface = interface_by_ip[ip.sin_addr.s_addr];

    ifreq ifhwaddr_request {};
    std::strncpy(ifhwaddr_request.ifr_ifrn.ifrn_name, binterface.c_str(),
        sizeof(ifhwaddr_request.ifr_ifrn.ifrn_name));
    if (ioctl(s.get(), SIOCGIFHWADDR, &ifhwaddr_request) < 0) {
        throw std::runtime_error(
            fmt::format("ioctl failed with errno={}: \"{}\"", errno, strerror(errno)));
    }
    MacAddress mac;
    static_assert(mac.max_size() <= sizeof(ifhwaddr_request.ifr_ifru.ifru_addr.sa_data));
    std::copy(ifhwaddr_request.ifr_ifru.ifru_addr.sa_data,
        ifhwaddr_request.ifr_ifru.ifru_addr.sa_data + mac.max_size(),
        mac.begin());
    HSB_LOG_DEBUG("destination_ip={} local_ip={} mac_id={:x}:{:x}:{:x}:{:x}:{:x}:{:x}",
        destination_ip, inet_ntoa(ip.sin_addr), mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    return { inet_ntoa(ip.sin_addr), binterface, mac };
}

std::tuple<std::string, std::string, MacAddress> local_ip_and_mac_from_socket(int socket_fd)
{
    // Start with a map of IP address to interfaces.
    std::map<in_addr_t, std::string> interface_by_ip;
    // First, find out how many interfaces there are.
    ifconf ifconf_request {};
    if (ioctl(socket_fd, SIOCGIFCONF, &ifconf_request) < 0) {
        throw std::runtime_error(
            fmt::format("ioctl failed with errno={}: \"{}\"", errno, strerror(errno)));
    }
    assert(ifconf_request.ifc_len > 0);
    //
    std::vector<ifreq> ifreq_buffer(ifconf_request.ifc_len / sizeof(ifreq));
    ifconf_request.ifc_ifcu.ifcu_req = ifreq_buffer.data();
    if (ioctl(socket_fd, SIOCGIFCONF, &ifconf_request) < 0) {
        throw std::runtime_error(
            fmt::format("ioctl failed with errno={}: \"{}\"", errno, strerror(errno)));
    }
    assert(static_cast<size_t>(ifconf_request.ifc_len) == ifreq_buffer.size() * sizeof(ifreq));
    assert(ifconf_request.ifc_ifcu.ifcu_req == ifreq_buffer.data());
    for (auto&& req : ifreq_buffer) {
        const std::string name(req.ifr_ifrn.ifrn_name);
        const in_addr ip = ((struct sockaddr_in*)&req.ifr_ifru.ifru_addr)->sin_addr;
        HSB_LOG_TRACE("name={} ip={}", name, inet_ntoa(ip));
        interface_by_ip[ip.s_addr] = name;
    }

    sockaddr_in ip {};
    ip.sin_family = AF_UNSPEC;
    socklen_t ip_len = sizeof(ip);
    if (getsockname(socket_fd, (sockaddr*)&ip, &ip_len) < 0) {
        throw std::runtime_error(
            fmt::format("getsockname failed with errno={}: \"{}\"", errno, strerror(errno)));
    }
    const std::string binterface = interface_by_ip[ip.sin_addr.s_addr];

    ifreq ifhwaddr_request {};
    std::strncpy(ifhwaddr_request.ifr_ifrn.ifrn_name, binterface.c_str(),
        sizeof(ifhwaddr_request.ifr_ifrn.ifrn_name));
    if (ioctl(socket_fd, SIOCGIFHWADDR, &ifhwaddr_request) < 0) {
        throw std::runtime_error(
            fmt::format("ioctl failed with errno={}: \"{}\"", errno, strerror(errno)));
    }
    MacAddress mac;
    static_assert(mac.max_size() <= sizeof(ifhwaddr_request.ifr_ifru.ifru_addr.sa_data));
    std::copy(ifhwaddr_request.ifr_ifru.ifru_addr.sa_data,
        ifhwaddr_request.ifr_ifru.ifru_addr.sa_data + mac.max_size(),
        mac.begin());
    HSB_LOG_DEBUG("local_ip={} mac_id={:x}:{:x}:{:x}:{:x}:{:x}:{:x}",
        inet_ntoa(ip.sin_addr), mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    return { inet_ntoa(ip.sin_addr), binterface, mac };
}

std::tuple<std::string, std::string, MacAddress> local_ip_and_mac_for_interface(
    const std::string& interface, const std::string& destination_ip)
{
    if (interface.empty()) {
        return local_ip_and_mac(destination_ip);
    }

    in_addr peer {};
    const bool have_peer = (inet_pton(AF_INET, destination_ip.c_str(), &peer) == 1);

    ifaddrs* ifa_list = nullptr;
    if (getifaddrs(&ifa_list) != 0) {
        HSB_LOG_DEBUG("getifaddrs failed with errno={}: \"{}\"; falling back to a "
                      "routing-table lookup for {}.",
            errno, strerror(errno), destination_ip);
        return local_ip_and_mac(destination_ip);
    }
    std::unique_ptr<ifaddrs, decltype(&freeifaddrs)> ifa_guard(ifa_list, &freeifaddrs);

    // An interface may hold several IPv4 addresses. Prefer the one whose subnet
    // contains the board -- that is the address the board can actually reach,
    // and the one the kernel would have put in ipi_spec_dst. Otherwise take the
    // first, which is the single-address case.
    std::string first_ipv4;
    std::string subnet_ipv4;
    for (ifaddrs* ifa = ifa_list; ifa != nullptr; ifa = ifa->ifa_next) {
        if ((ifa->ifa_addr == nullptr) || (ifa->ifa_addr->sa_family != AF_INET)) {
            continue;
        }
        if ((ifa->ifa_name == nullptr) || (interface != ifa->ifa_name)) {
            continue;
        }
        const in_addr addr = reinterpret_cast<const sockaddr_in*>(ifa->ifa_addr)->sin_addr;
        char addr_buffer[INET_ADDRSTRLEN] = {};
        inet_ntop(AF_INET, &addr, addr_buffer, sizeof(addr_buffer));
        if (first_ipv4.empty()) {
            first_ipv4 = addr_buffer;
        }
        if (have_peer && (ifa->ifa_netmask != nullptr)) {
            const in_addr mask
                = reinterpret_cast<const sockaddr_in*>(ifa->ifa_netmask)->sin_addr;
            if ((addr.s_addr & mask.s_addr) == (peer.s_addr & mask.s_addr)) {
                subnet_ipv4 = addr_buffer;
                break;
            }
        }
    }

    const std::string local_ip = subnet_ipv4.empty() ? first_ipv4 : subnet_ipv4;
    if (local_ip.empty()) {
        HSB_LOG_DEBUG("Interface \"{}\" has no IPv4 address; falling back to a "
                      "routing-table lookup for {}.",
            interface, destination_ip);
        return local_ip_and_mac(destination_ip);
    }

    MacAddress mac {};
    try {
        mac = local_mac(interface);
    } catch (const std::exception& e) {
        HSB_LOG_DEBUG("No MAC for interface \"{}\" ({}); falling back to a "
                      "routing-table lookup for {}.",
            interface, e.what(), destination_ip);
        return local_ip_and_mac(destination_ip);
    }

    HSB_LOG_TRACE("local_ip_and_mac_for_interface interface={} destination_ip={} -> local_ip={}",
        interface, destination_ip, local_ip);
    return { local_ip, interface, mac };
}

/**
 * @brief
 *
 * @param interface
 * @return MacAddress.
 */
MacAddress local_mac(const std::string& interface)
{
    UniqueFileDescriptor s(socket(AF_INET, SOCK_DGRAM, 0));
    if (!s) {
        throw std::runtime_error("Failed to create socket");
    }

    ifreq ifhwaddr_request = {}; // Initialize to 0s
    interface.copy(
        ifhwaddr_request.ifr_ifrn.ifrn_name, sizeof(ifhwaddr_request.ifr_ifrn.ifrn_name));
    if (ioctl(s.get(), SIOCGIFHWADDR, &ifhwaddr_request) < 0) {
        throw std::runtime_error(
            fmt::format("ioctl failed with errno={}: \"{}\"", errno, strerror(errno)));
    }
    MacAddress mac;
    static_assert(mac.max_size() <= sizeof(ifhwaddr_request.ifr_ifru.ifru_addr.sa_data));
    std::copy(ifhwaddr_request.ifr_ifru.ifru_addr.sa_data,
        ifhwaddr_request.ifr_ifru.ifru_addr.sa_data + mac.max_size(),
        mac.begin());

    return mac;
}

} // namespace hololink::core
