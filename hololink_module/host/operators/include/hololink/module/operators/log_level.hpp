/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_OPERATORS_LOG_LEVEL_HPP
#define HOLOLINK_MODULE_OPERATORS_LOG_LEVEL_HPP

#include <cstdlib>
#include <stdexcept>
#include <string>

#include <strings.h>

#include <holoscan/logger/logger.hpp>

#include "hololink/module/adapter.hpp"
#include "hololink/module/logging.hpp"

namespace hololink::module::operators {

/**
 * @brief The log level an application should start with before looking at its command line.
 *
 * Returns the level named by the `HOLOLINK_LOG_LEVEL` environment variable, else by
 * `HOLOSCAN_LOG_LEVEL` (the same precedence the module logger uses), else INFO. Use it to
 * initialize the variable that `--log-level` overwrites, so the environment applies exactly
 * when the option is not given.
 *
 * @throws std::runtime_error if the variable holds an unrecognized level.
 */
inline holoscan::LogLevel default_log_level()
{
    const char* name = "HOLOLINK_LOG_LEVEL";
    const char* value = std::getenv(name);
    if (!value) {
        name = "HOLOSCAN_LOG_LEVEL";
        value = std::getenv(name);
    }
    if (!value) {
        return holoscan::LogLevel::INFO;
    }
    static const struct {
        const char* text;
        holoscan::LogLevel level;
    } levels[] = {
        { "TRACE", holoscan::LogLevel::TRACE },
        { "DEBUG", holoscan::LogLevel::DEBUG },
        { "INFO", holoscan::LogLevel::INFO },
        { "WARN", holoscan::LogLevel::WARN },
        { "ERROR", holoscan::LogLevel::ERROR },
        { "CRITICAL", holoscan::LogLevel::CRITICAL },
        { "OFF", holoscan::LogLevel::OFF },
    };
    for (const auto& entry : levels) {
        if (::strcasecmp(value, entry.text) == 0) {
            return entry.level;
        }
    }
    throw std::runtime_error(std::string("Invalid log level \"") + value + "\" in " + name);
}

/**
 * @brief Set the log level of Holoscan and of HSB code together.
 *
 * The module framework and the modules it loads log through the Adapter's
 * logger (see Adapter::set_log_level), not Holoscan's, so
 * holoscan::set_log_level() alone leaves them at the level from the
 * `HOLOLINK_LOG_LEVEL` / `HOLOSCAN_LOG_LEVEL` environment variables
 * (default Info). The module logger has no CRITICAL or OFF level: both
 * leave only its errors.
 *
 * @param level The level for both loggers.
 */
inline void set_log_level(holoscan::LogLevel level)
{
    holoscan::set_log_level(level);
    LogLevel hsb_level = LogLevel::Error;
    switch (level) {
    case holoscan::LogLevel::TRACE:
        hsb_level = LogLevel::Trace;
        break;
    case holoscan::LogLevel::DEBUG:
        hsb_level = LogLevel::Debug;
        break;
    case holoscan::LogLevel::INFO:
        hsb_level = LogLevel::Info;
        break;
    case holoscan::LogLevel::WARN:
        hsb_level = LogLevel::Warning;
        break;
    default:
        break;
    }
    Adapter::get_adapter().set_log_level(hsb_level);
}

} // namespace hololink::module::operators

#endif // HOLOLINK_MODULE_OPERATORS_LOG_LEVEL_HPP
