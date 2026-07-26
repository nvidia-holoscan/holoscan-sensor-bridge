/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_NAME_VALUE_PAIRS_HPP
#define HOLOLINK_MODULE_NAME_VALUE_PAIRS_HPP

#include <cstddef>
#include <map>
#include <string>
#include <string_view>

namespace hololink::module {

/* Decode a "name=value;name=value;…" string into its pairs.
 *
 * Used most often to unpack instance_id strings (the
 * "name=value pairs joined by `;`" convention used throughout the
 * module's service locator), but it has no service-locator coupling
 * and is fine for any blob in this shape.
 *
 * Empty input returns an empty map. A non-empty segment that lacks
 * an `=` is recorded as a name with an empty value (so "flag;k=v"
 * yields {{"flag", ""}, {"k", "v"}}). Empty segments (e.g. between
 * consecutive `;`s, or a trailing `;`) are skipped. Duplicate names:
 * the rightmost occurrence wins, matching the intuition of "later
 * assignments override earlier ones". */
inline std::map<std::string, std::string> parse_name_value_pairs(
    std::string_view input)
{
    std::map<std::string, std::string> out;
    std::size_t cursor = 0;
    while (cursor < input.size()) {
        const std::size_t end = input.find(';', cursor);
        const std::size_t pair_end
            = (end == std::string_view::npos) ? input.size() : end;
        if (cursor < pair_end) {
            const std::size_t equals = input.find('=', cursor);
            if (equals != std::string_view::npos && equals < pair_end) {
                const std::string_view name = input.substr(cursor, equals - cursor);
                const std::string_view value
                    = input.substr(equals + 1, pair_end - equals - 1);
                out[std::string(name)] = std::string(value);
            } else {
                const std::string_view name = input.substr(cursor, pair_end - cursor);
                out[std::string(name)] = std::string();
            }
        }
        if (end == std::string_view::npos) {
            break;
        }
        cursor = end + 1;
    }
    return out;
}

} // namespace hololink::module

#endif // HOLOLINK_MODULE_NAME_VALUE_PAIRS_HPP
