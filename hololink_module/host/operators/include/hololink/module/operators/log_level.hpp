/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_OPERATORS_LOG_LEVEL_HPP
#define HOLOLINK_MODULE_OPERATORS_LOG_LEVEL_HPP

#include <holoscan/logger/logger.hpp>

#include "hololink/module/adapter.hpp"
#include "hololink/module/logging.hpp"

namespace hololink::module::operators {

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
