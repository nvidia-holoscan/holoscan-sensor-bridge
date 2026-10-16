/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SRC_HOLOLINK_CORE_NETWORKING
#define SRC_HOLOLINK_CORE_NETWORKING

#include <unistd.h> // for close()

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

#include "nullable_pointer.hpp"

namespace hololink::core {

// When we recv from a UDP socket, use
// a buffer large enough to accommodate
// a 9k jumbo packet.
constexpr uint32_t UDP_PACKET_SIZE = 10240;

// All our I/O are aligned to this page size.
constexpr uint32_t PAGE_SIZE = 128;

// The default MTU comes from the TCP/IP specification.
constexpr uint32_t DEFAULT_MTU = 1500;

// Round up
size_t round_up(size_t value, size_t alignment);

/// MAC (medium access control) address
using MacAddress = std::array<uint8_t, 6>;

/// A RAII object which automatically closes a file handle when going out of scope (e.g. for sockets).
/// The null sentinel is -1 (not 0): 0 is a valid file descriptor, so it must not read as "empty".
using UniqueFileDescriptor = std::unique_ptr<Nullable<int, -1>, Nullable<int, -1>::Deleter<int, &close>>;

/**
 * @brief Works only on Linux.
 *
 * @param destination_ip
 * @param port
 * @return our IP address that can connect to `destination_ip` and the MAC ID for that interface.
 */
std::tuple<std::string, std::string, MacAddress> local_ip_and_mac(
    const std::string& destination_ip, uint32_t port = 1);

/**
 * @brief Works only on Linux.
 *
 * @return our IP address, interface name, and the MAC ID for the interface that
 * socket_fd uses to transmit.
 */
std::tuple<std::string, std::string, MacAddress> local_ip_and_mac_from_socket(int socket_fd);

/**
 * @brief Resolve the host-side endpoint from a NAMED interface rather than from
 * the routing table.
 *
 * local_ip_and_mac() connect()s a UDP socket to the board and reads back the
 * source address the kernel picked -- an answer that comes from the routing
 * table. Every HSB board ships with the same default 192.168.0.x data-plane
 * addressing, so with two boards on two NICs the subnets collide and the lookup
 * returns whichever interface won the route tie, for BOTH boards. The FPGA is
 * then told to send frames to a MAC belonging to a different NIC than the one
 * it is cabled to and they are dropped on arrival: link up, control plane fine,
 * zero frames delivered. Per-interface host routes used to paper over this.
 *
 * Enumeration already records the NIC the bootp reply arrived on (IP_PKTINFO),
 * and an interface has an unambiguous MAC and, for the board's subnet, an
 * unambiguous address. So prefer that over asking the routing table.
 *
 * Falls back to local_ip_and_mac(destination_ip) when `interface` is empty or
 * carries no usable IPv4 address -- metadata that predates the "interface" key,
 * or a hand-built blob, still works.
 *
 * @param interface NIC name the device enumerated on; "" to force the fallback
 * @param destination_ip the board, used to pick among an interface's addresses
 * @returns our IP address, interface name, and the MAC ID for that interface
 */
std::tuple<std::string, std::string, MacAddress> local_ip_and_mac_for_interface(
    const std::string& interface, const std::string& destination_ip);

/**
 * @brief Get the Mac ID for the given interface by name.
 *
 * @param interface
 * @return MacAddress.
 */
MacAddress local_mac(const std::string& interface);

} // namespace hololink::core

#endif /* SRC_HOLOLINK_CORE_NULLABLE_POINTER */
