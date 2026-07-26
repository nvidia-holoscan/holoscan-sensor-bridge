/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "networking.hpp"

#include <arpa/inet.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

#include <fmt/core.h>

#include "hololink/module/logging.hpp" // HSB_LOG_*

namespace hololink::module {

// RAII guard that closes a socket fd on scope exit.
class SocketGuard {
public:
    explicit SocketGuard(int fd)
        : fd_(fd)
    {
    }
    ~SocketGuard()
    {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }
    SocketGuard(const SocketGuard&) = delete;
    SocketGuard& operator=(const SocketGuard&) = delete;

    int get() const { return fd_; }
    explicit operator bool() const { return fd_ >= 0; }

private:
    int fd_;
};

std::tuple<std::string, std::string, MacAddress> local_ip_and_mac(
    const std::string& destination_ip, uint32_t port)
{
    // A port is required for the connect() below to resolve a route, but
    // because this is SOCK_DGRAM no actual traffic is sent.
    SocketGuard s(::socket(AF_INET, SOCK_DGRAM, 0));
    if (!s) {
        throw std::runtime_error("Failed to create socket");
    }

    // Build a map of local IP address -> interface name.
    std::map<in_addr_t, std::string> interface_by_ip;
    ifconf ifconf_request {};
    if (::ioctl(s.get(), SIOCGIFCONF, &ifconf_request) < 0) {
        throw std::runtime_error(
            fmt::format("ioctl failed with errno={}: \"{}\"", errno, strerror(errno)));
    }
    std::vector<ifreq> ifreq_buffer(ifconf_request.ifc_len / sizeof(ifreq));
    ifconf_request.ifc_ifcu.ifcu_req = ifreq_buffer.data();
    if (::ioctl(s.get(), SIOCGIFCONF, &ifconf_request) < 0) {
        throw std::runtime_error(
            fmt::format("ioctl failed with errno={}: \"{}\"", errno, strerror(errno)));
    }
    for (auto&& req : ifreq_buffer) {
        const std::string name(req.ifr_ifrn.ifrn_name);
        const in_addr ip = ((struct sockaddr_in*)&req.ifr_ifru.ifru_addr)->sin_addr;
        HSB_LOG_TRACE("name={} ip={}", name, inet_ntoa(ip));
        interface_by_ip[ip.s_addr] = name;
    }

    // A connected datagram socket only does I/O with the connected
    // address; getsockname() then reports the local IP the route uses.
    timeval timeout {};
    if (::setsockopt(s.get(), SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) < 0) {
        throw std::runtime_error(
            fmt::format("setsockopt failed with errno={}: \"{}\"", errno, strerror(errno)));
    }

    sockaddr_in address {};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    if (inet_pton(AF_INET, destination_ip.c_str(), &address.sin_addr) == 0) {
        throw std::runtime_error(fmt::format("Failed to convert address {}", destination_ip));
    }
    if (::connect(s.get(), (sockaddr*)&address, sizeof(address)) < 0) {
        throw std::runtime_error(
            fmt::format("connect failed with errno={}: \"{}\"", errno, strerror(errno)));
    }
    sockaddr_in ip {};
    ip.sin_family = AF_UNSPEC;
    socklen_t ip_len = sizeof(ip);
    if (::getsockname(s.get(), (sockaddr*)&ip, &ip_len) < 0) {
        throw std::runtime_error(
            fmt::format("getsockname failed with errno={}: \"{}\"", errno, strerror(errno)));
    }
    auto local_interface_it = interface_by_ip.find(ip.sin_addr.s_addr);
    if (local_interface_it == interface_by_ip.end()) {
        throw std::runtime_error(
            fmt::format("No local interface found for local IP {}", inet_ntoa(ip.sin_addr)));
    }
    const std::string& local_interface = local_interface_it->second;

    ifreq ifhwaddr_request {};
    // ifhwaddr_request is zero-initialized, so copying one fewer byte than the
    // field size leaves the name null-terminated.
    std::strncpy(ifhwaddr_request.ifr_ifrn.ifrn_name, local_interface.c_str(),
        sizeof(ifhwaddr_request.ifr_ifrn.ifrn_name) - 1);
    if (::ioctl(s.get(), SIOCGIFHWADDR, &ifhwaddr_request) < 0) {
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
    return { inet_ntoa(ip.sin_addr), local_interface, mac };
}

} // namespace hololink::module
