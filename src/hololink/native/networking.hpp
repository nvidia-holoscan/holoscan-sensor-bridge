/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SRC_HOLOLINK_NATIVE_NETWORKING
#define SRC_HOLOLINK_NATIVE_NETWORKING

#include <unistd.h> // for close()

#include <array>
#include <fmt/format.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

#include "nullable_pointer.hpp"

namespace hololink::native {

// When we recv from a UDP socket, use
// a buffer large enough to accomodate
// a 9k jumbo packet.
constexpr uint32_t UDP_PACKET_SIZE = 10240;

// All our I/O are aligned to this page size.
constexpr uint32_t PAGE_SIZE = 128;

// Round up
constexpr static uint32_t round_up(uint32_t value, uint32_t alignment)
{
    // This only works when alignment is a power of two.
    if (alignment & (alignment - 1)) {
        throw std::runtime_error(fmt::format("round_up called with an invalid alignment={:#x}; it must be a power of two.", alignment));
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

/// MAC (medium access control) address
using MacAddress = std::array<uint8_t, 6>;

/// A RAII object which automatically closes a file handle when going out of scope (e.g. for sockets)
using UniqueFileDescriptor = std::unique_ptr<Nullable<int>, Nullable<int>::Deleter<int, &close>>;

/**
 * @brief Works only on Linux.
 *
 * @param destination_ip
 * @param port
 * @returns our IP address that can connect to `destination_ip` and the MAC ID for that interface.
 */
std::tuple<std::string, std::string, MacAddress> local_ip_and_mac(
    const std::string& destination_ip, uint32_t port = 1);

/**
 * @brief Works only on Linux.
 *
 * @returns our IP address, interface name, and the MAC ID for the interface that
 * socket_fd uses to transmit.
 */
std::tuple<std::string, std::string, MacAddress> local_ip_and_mac_from_socket(int socket_fd);

/**
 * @brief Get the Mac ID for the given interface by name
 *
 * @param interface
 * @return MacAddress
 */
MacAddress local_mac(const std::string& interface);

} // namespace hololink::native

#endif /* SRC_HOLOLINK_NATIVE_NULLABLE_POINTER */
