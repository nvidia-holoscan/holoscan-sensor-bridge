/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_TOOLS_HPP
#define HOLOLINK_MODULE_TOOLS_HPP

#include <cstddef>
#include <cstdlib>
#include <string>

/* Adapter-owned host-side helper utilities. Self-contained / header-only so
 * example programs and tools can use them without linking the legacy
 * src/hololink/common tree. Ported from src/hololink/common/tools.cpp. */

namespace hololink::module {

/* Returns the i-th entry of the comma-separated HOLOLINK_IPS environment
 * variable, or `fallback` when the variable is unset, empty, or has too few
 * entries. Lets ctest pass per-test board IPs through (matching the
 * HOLOLINK_IPS=... shape the pytest --channel-ips option also accepts via
 * PYTEST_ADDOPTS) without each example threading its own --hololink flag. */
inline std::string env_hololink_ip(std::size_t index, const std::string& fallback)
{
    const char* env = std::getenv("HOLOLINK_IPS");
    if (env == nullptr || env[0] == '\0') {
        return fallback;
    }
    const std::string ips(env);
    std::size_t start = 0;
    for (std::size_t i = 0; i < index; ++i) {
        const std::size_t comma = ips.find(',', start);
        if (comma == std::string::npos) {
            return fallback;
        }
        start = comma + 1;
    }
    std::size_t end = ips.find(',', start);
    if (end == std::string::npos) {
        end = ips.size();
    }
    if (end == start) {
        return fallback;
    }
    return ips.substr(start, end - start);
}

} // namespace hololink::module

#endif // HOLOLINK_MODULE_TOOLS_HPP
