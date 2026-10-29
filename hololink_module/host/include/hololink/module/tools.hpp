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

/**
 * @brief Get the IP address of a device from the `HOLOLINK_IPS`
 * environment variable.
 *
 * `HOLOLINK_IPS` holds a comma-separated list of device addresses, for
 * example `192.168.0.2,192.168.0.3`. This returns the entry at position
 * `index`, counting from zero. It lets a test harness (for example
 * `ctest`, or pytest through its `--channel-ips` option) choose the
 * devices a program uses, without each program needing its own
 * command-line option; programs normally pass the result as the default
 * for such an option.
 *
 * The entry is returned exactly as written: whitespace is not removed
 * and the text is not checked to be a valid address.
 *
 * @param index Zero-based position in the list.
 * @param fallback Value to return when no address is available for
 *      `index`: the variable is unset or empty, the list has `index` or
 *      fewer entries, or the entry at `index` is empty (as in `a,,b`).
 * @returns The selected address, or `fallback`.
 */
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
