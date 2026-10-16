/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#ifndef HOLOLINK_TRANSPORT_ROCE_STRING_UTILS_HPP
#define HOLOLINK_TRANSPORT_ROCE_STRING_UTILS_HPP

#include <ios>
#include <locale>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

namespace hololink::transport::roce::detail {

template <typename... Parts>
std::string concatenate(Parts&&... parts)
{
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    (stream << ... << std::forward<Parts>(parts));
    return stream.str();
}

template <typename T>
std::string hex(T value)
{
    using Value = std::remove_cv_t<T>;
    static_assert(
        std::is_integral_v<Value> && sizeof(T) > 1,
        "hex() requires a non-narrow integral type");
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << "0x" << std::hex << value;
    return stream.str();
}

} // namespace hololink::transport::roce::detail

#endif
