/**
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SRC_HOLOLINK_METADATA
#define SRC_HOLOLINK_METADATA

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace hololink {

/**
 * A container that contains key-value pairs with unique keys, values can be either `int64_t`,
 * `std::string` or std::vector<uint8_t>.
 *
 * This is a specialization of std::map which is adding a function to get the contained value
 * without creating an entry if the key is not present, and a function to update one Metadata object
 * with the key-value pairs of another Metadata object.
 */
class Metadata : public std::map<std::string, std::variant<int64_t, std::string, std::vector<uint8_t>>> {
public:
    using Element = Metadata::mapped_type;

    /**
     * @brief Get a value of a given type with the given key. If the key is not present it won't be
     * inserted and an empty value is returned.
     *
     * @tparam T value type
     * @param name key name
     * @return std::optional<const T> An optional value, if the key is not present this has no
     * value.
     */
    template <typename T>
    std::optional<const T> get(const std::string& name) const
    {
        auto data = find(name);
        if (data == cend()) {
            return {};
        }
        if constexpr (std::is_same_v<T, Element>) {
            // special case to get the variant itself
            return data->second;
        } else if (std::holds_alternative<T>(data->second)) {
            return std::get<T>(data->second);
        } else {
            return {};
        }
    }

    /**
     * @brief Update the container with values from another container.
     *
     * @param other container to update this container with
     */
    void update(const Metadata& other)
    {
        for (const auto& element : other) {
            // `emplace` adds if the element with the same name is not existing
            auto r = emplace(element);
            if (!r.second) {
                r.first->second = element.second;
            }
        }
    }
};

} // namespace hololink

#endif /* SRC_HOLOLINK_METADATA */
