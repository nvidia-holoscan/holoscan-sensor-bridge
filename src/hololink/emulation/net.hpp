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

#include "hsb_config.hpp"

#ifdef __linux__
#include <net/if.h>
#else
#define IFNAMSIZ 16
#endif

namespace hololink::emulation {

#define DATA_SOURCE_UDP_PORT 12288u
#define UDP_PACKET_SIZE 10240u

#define IP_ADDRESS_MAX_BITS 32u
#define IP_ADDRESS_DEFAULT_BITS 24u

#define IPADDRESS_HAS_ADDR 0x1
#define IPADDRESS_HAS_NETMASK 0x2
#define IPADDRESS_HAS_BROADCAST 0x4
#define IPADDRESS_IS_PTP 0x40
#define IPADDRESS_HAS_MAC 0x80

#define HSB_PAGE_SIZE 128u

#define IB_PSN_MASK 0xFFFFFu
#define IB_PSN_SHIFT 12u

/**
 * python (limited API):
 *
 * `def __init__(self: hemu.IPAddress, ip_address: str, /)`
 *
 * @brief Encapsulates all the relevant information for transmission from an Emulator device IP address
 *
 * This structure contains the network interface configuration required for the emulator to send
 * packets, including interface name, IP address, MAC address, optional port and broadcast address
 * (defaulting to 255.255.255.255 if on interface that does not have an explicit broadcast, e.g. loopback).
 */
typedef struct IPAddress {
    char if_name[IFNAMSIZ]; ///< Network interface name (e.g., "eth0", "lo")
    uint32_t ip_address { 0 }; ///< IP address in network byte order
    uint32_t subnet_mask { 0 }; ///< Subnet mask in network byte order
    uint32_t broadcast_address { 0 }; ///< Broadcast address in network byte order
    uint8_t mac[6] { 0 }; ///< MAC address of the interface
    uint16_t port { DATA_SOURCE_UDP_PORT }; ///< UDP port number (defaults to DATA_SOURCE_UDP_PORT)
    uint8_t flags { 0 }; ///< Configuration flags indicating which fields are valid
    /**
     * @brief Configuration flags bitfield:
     * - 0x01 (IPADDRESS_HAS_ADDR): IP address is valid (and whole struct is valid)
     * - 0x02 (IPADDRESS_HAS_NETMASK): netmask is valid
     * - 0x04 (IPADDRESS_HAS_BROADCAST): broadcast address is valid
     * - 0x40 (IPADDRESS_IS_PTP): indicates point-to-point (1) or broadcast (0) interface
     * - 0x80 (IPADDRESS_HAS_MAC): MAC address is valid
     */
} IPAddress;

/**
 * python (use IPAddress object constructor)
 *
 * @brief Construct an IPAddress object from a string representation of the IP address.
 * @param ip_address The string representation of the IP address. Currently must be in format accepted by inet_addr().
 * @return An IPAddress instance
 */
IPAddress IPAddress_from_string(const std::string& ip_address);

/**
 * python (use IPAddress object __str__() method)
 *
 * `def __str__(self: hemu.IPAddress) -> str`
 *
 * @brief Convert an IPAddress object to its string representation.
 * @param ip_address The IPAddress object to convert
 * @return String representation of the IP address
 */
std::string IPAddress_to_string(const IPAddress& ip_address);

/**
 * @brief Get the broadcast address for a given IPAddress.
 * @param ip_address The IPAddress object containing the network configuration
 * @return The broadcast address as a 32-bit unsigned integer
 */
uint32_t get_broadcast_address(const IPAddress& ip_address);

/**
 * @brief Disable broadcast configuration warning messages.
 */
void DISABLE_BROADCAST_CONFIG_WARNING();

} // namespace hololink::emulation

#endif /* EMULATION_NET_HPP */
