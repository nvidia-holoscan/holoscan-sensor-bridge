/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_SRC_NETWORKING_HPP
#define HOLOLINK_MODULE_SRC_NETWORKING_HPP

#include <array>
#include <cstdint>
#include <string>
#include <tuple>

/* Host-private networking helpers. Adapter-owned so the host framework
 * needs no dependency on src/hololink/core/networking. Not a public
 * header — lives under src/ and is never installed. */

namespace hololink::module {

/// MAC (medium access control) address.
using MacAddress = std::array<uint8_t, 6>;

/* Returns the local IP, kernel interface name, and MAC address of the
 * interface the host would use to reach `destination_ip`. Linux-only.
 * The port is needed only to make the (datagram) connect() resolve a
 * route; no traffic is sent. */
std::tuple<std::string, std::string, MacAddress> local_ip_and_mac(
    const std::string& destination_ip, uint32_t port = 1);

} // namespace hololink::module

#endif // HOLOLINK_MODULE_SRC_NETWORKING_HPP
