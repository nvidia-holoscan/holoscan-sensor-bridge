/**
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SRC_HOLOLINK_LOGGING_INTERNAL
#define SRC_HOLOLINK_LOGGING_INTERNAL

// This file defines the internal logging for HSB and should not be released with the public package.

#include "logging.hpp"
#include "metadata.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

/**
 * @brief Formatter for Metadata
 *
 * @tparam
 */
template <>
struct fmt::formatter<hololink::Metadata> : fmt::formatter<fmt::string_view> {
    /**
     * @brief Format function for Metadata
     *
     * @param metadata
     * @param ctx
     * @return auto
     */
    auto format(const hololink::Metadata& metadata, fmt::format_context& ctx) const
        -> decltype(ctx.out());
};

/**
 * @brief Formatter for a Metadata element
 *
 * @tparam
 */
template <>
struct fmt::formatter<hololink::Metadata::Element> : fmt::formatter<fmt::string_view> {
    /**
     * @brief Format function for Metadata element
     *
     * @param element
     * @param ctx
     * @return auto
     */
    auto format(const hololink::Metadata::Element& element, fmt::format_context& ctx) const
        -> decltype(ctx.out());
};

namespace hololink::logging {

template <typename FormatT, typename... ArgsT>
static inline void hsb_log_fmt(char const* file, unsigned line, const char* function, HsbLogLevel level, const FormatT& format, ArgsT&&... args)
{
    if (level >= hsb_log_level) {
        auto fmt_args = fmt::make_format_args<fmt::buffer_context<fmt::char_t<FormatT>>>(args...);
        hsb_logger(file, line, function, level, fmt::vformat(format, fmt_args).c_str());
    }
}

} // namespace hololink::logging

// The internal logging uses libfmt and it's expected that __VA_ARGS__ includes the format string.
#define HSB_LOG_FMT(level, ...) hololink::logging::hsb_log_fmt(__FILE__, __LINE__, static_cast<const char*>(__FUNCTION__), level, __VA_ARGS__)

#define HSB_LOG_TRACE(...) HSB_LOG_FMT(hololink::logging::HsbLogLevel::HSB_LOG_LEVEL_TRACE, __VA_ARGS__)
#define HSB_LOG_DEBUG(...) HSB_LOG_FMT(hololink::logging::HsbLogLevel::HSB_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define HSB_LOG_INFO(...) HSB_LOG_FMT(hololink::logging::HsbLogLevel::HSB_LOG_LEVEL_INFO, __VA_ARGS__)
#define HSB_LOG_WARN(...) HSB_LOG_FMT(hololink::logging::HsbLogLevel::HSB_LOG_LEVEL_WARN, __VA_ARGS__)
#define HSB_LOG_ERROR(...) HSB_LOG_FMT(hololink::logging::HsbLogLevel::HSB_LOG_LEVEL_ERROR, __VA_ARGS__)

#endif /* SRC_HOLOLINK_LOGGING_INTERNAL */
