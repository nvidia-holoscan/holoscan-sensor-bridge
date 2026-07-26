/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_NETWORKING_HPP
#define HOLOLINK_MODULE_NETWORKING_HPP

#include <array>
#include <cstdint>

/* Adapter-owned networking types and constants. Public counterpart of the
 * internal hololink_module/host/src/networking.hpp, so applications (e.g. the
 * FUSA example player) name no legacy hololink::core type or value. */

namespace hololink::module {

/// MAC (medium access control) address.
using MacAddress = std::array<uint8_t, 6>;

/// Default Ethernet MTU, in bytes. Adapter-owned mirror of the legacy
/// hololink::core value so callers name no legacy hololink::core constant.
constexpr uint32_t DEFAULT_MTU = 1500;

} // namespace hololink::module

#endif // HOLOLINK_MODULE_NETWORKING_HPP
