/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_ENUMERATION_METADATA_HPP
#define HOLOLINK_MODULE_ENUMERATION_METADATA_HPP

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace hololink::module {

/* Discovery / per-channel metadata that flows through enumeration.
 * Keys are well-known field names ("peer_ip", "control_port",
 * "serial_number", "fpga_uuid", "compat_id", "data_plane", …); values
 * are int64_t / string / byte-blob.
 *
 * EnumerationMetadata crosses the .so boundary by reference. The same
 * toolchain constraint plus the ABI check at module load make
 * sizeof / alignment of this type identical on both sides. */
class EnumerationMetadata
    : public std::map<std::string,
          std::variant<int64_t, std::string, std::vector<uint8_t>>> {
public:
    using Value = mapped_type;

    /* Return true if the key is present (regardless of which variant
     * alternative it holds). Provided here because std::map::contains
     * is C++20-only and the project builds against C++17. */
    bool contains(const std::string& key) const
    {
        return find(key) != cend();
    }

    /* Look up a value of a specific alternative type.
     *
     * Throws std::runtime_error when the key is missing or its variant
     * does not hold T. Use the (key, default_value) overload when the
     * field is genuinely optional; use contains(key) when the call
     * site needs to branch on presence without fetching the value. */
    template <typename T>
    T get(const std::string& key) const
    {
        const auto it = find(key);
        if (it != cend() && std::holds_alternative<T>(it->second)) {
            return std::get<T>(it->second);
        }
        throw std::runtime_error(
            "While fetching '" + key + "' from enumeration metadata: "
                                       "key is missing or value is of wrong type");
    }

    /* Look up a value of a specific alternative type, falling back to a
     * caller-supplied default when the key is missing or holds a
     * different alternative. */
    template <typename T>
    T get(const std::string& key, const T& default_value) const
    {
        const auto it = find(key);
        if (it != cend() && std::holds_alternative<T>(it->second)) {
            return std::get<T>(it->second);
        }
        return default_value;
    }
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_ENUMERATION_METADATA_HPP
