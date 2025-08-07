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

#ifndef EMULATION_NET_HPP
#define EMULATION_NET_HPP

#include <cstdint>
#include <string>
#include <tuple>

#include "hololink/core/networking.hpp"

namespace hololink::emulation {

#define IP_ADDRESS_MAX_BITS 32u
#define IP_ADDRESS_DEFAULT_BITS 24u

#ifndef DEFAULT_TTL
#define DEFAULT_TTL 0x40
#endif

#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

std::tuple<std::string, std::string, hololink::core::MacAddress> local_ip_and_mac(
    const std::string& destination_ip, uint32_t port);

// store in host byte order
struct IPAddress {
    uint32_t ip_address;
    uint32_t subnet_mask;
};

static inline struct IPAddress IPAddress_from_string(const std::string& ip_address, uint8_t subnet_bits = IP_ADDRESS_DEFAULT_BITS)
{
    if (!subnet_bits) {
        uint32_t ip_address_int = inet_addr(ip_address.c_str());
        if (!ip_address_int) {
            return { 0, 0 };
        }
        throw std::invalid_argument("invalid IP address for subnet_bits=0. can only specify 0 subnet bits for 0.0.0.0");
    } else if (subnet_bits > IP_ADDRESS_MAX_BITS) {
        throw std::invalid_argument("subnet_bits must be in the range 0-32");
    }
    uint32_t ip_address_int = inet_addr(ip_address.c_str());
    if (ip_address_int == INADDR_NONE) {
        throw std::invalid_argument("invalid IP address");
    }
    return { ntohl(ip_address_int), ((uint32_t)0xFFFFFFFF) << (IP_ADDRESS_MAX_BITS - subnet_bits) };
}

static inline std::string IPAddress_to_string(const IPAddress& ip_address)
{
    struct in_addr addr = {
        .s_addr = htonl(ip_address.ip_address),
    };
    return std::string(inet_ntoa(addr));
}

static inline uint32_t get_broadcast_address(const IPAddress& ip_address)
{
    return (ip_address.ip_address & ip_address.subnet_mask) | (0xFFFFFFFF & ~ip_address.subnet_mask);
}

} // namespace hololink::emulation

#endif /* EMULATION_NET_HPP */
