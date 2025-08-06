/*
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
 */

#include "metadata.hpp"
#include "logging_internal.hpp"

#include <sstream>

auto fmt::formatter<hololink::Metadata::Element>::format(const hololink::Metadata::Element& element,
    fmt::format_context& ctx) const -> decltype(ctx.out())
{
    std::string buffer;
    std::visit(
        [&buffer](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::vector<uint8_t>>) {
                std::stringstream content_stream;
                content_stream << "(";
                for (size_t i = 0; i < arg.size(); ++i) {
                    if (i) {
                        content_stream << " ";
                    }
                    if ((i >= 3) && (i < (arg.size() - 3))) {
                        content_stream << "... ";
                        i = arg.size() - 3;
                    }
                    content_stream << fmt::format("{:02X}", arg[i]);
                }
                content_stream << ")";
                buffer = content_stream.str();
            } else if constexpr (std::is_same_v<T, std::string>) {
                buffer = fmt::format("\"{}\"", arg);
            } else {
                buffer = fmt::format("{}", arg);
            }
        },
        element);
    return fmt::format_to(ctx.out(), "{}", buffer);
}

auto fmt::formatter<hololink::Metadata>::format(
    const hololink::Metadata& metadata, fmt::format_context& ctx) const -> decltype(ctx.out())
{
    auto appender = ctx.out();
    std::string buffer;
    for (auto it = metadata.cbegin(); it != metadata.cend(); ++it) {
        fmt::format_to(appender, "\"{}\": {}, ", it->first, it->second);
    }
    return appender;
}
