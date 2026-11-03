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

/**
 * @brief Split a string of the form `name=value;name=value;...` into its
 * names and values.
 *
 * Service instance ids are encoded this way (for example
 * `serial=100;data_channel=1`), but this function can be used on any
 * string using this format.
 *
 * Details of how the text is read:
 *  - Pairs are separated by `;`. Within a pair, the name is the text
 *    before the first `=` and the value is everything after it, so
 *    `key=a=b` gives the value `a=b`.
 *  - A pair with no `=` has that text as its name and an empty value:
 *    `flag;k=v` gives `flag` -> `` and `k` -> `v`.
 *  - A `;` with nothing between it and the next `;` (or the end of the
 *    string) adds no pair, so `a=b;;c=d` is read as `a=b;c=d`, and
 *    `a=b;` as `a=b`. A pair that is only `=` gives
 *    an empty name with an empty value.
 *  - If a name occurs more than once, the last occurrence is used.
 *  - Text is taken exactly as written; whitespace is not trimmed and
 *    there is no escaping, so names and values cannot contain `;`
 *    (and names cannot contain `=`).
 *
 * @param input The text to split.
 * @returns The values by name. Empty if `input` is empty.
 */
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
