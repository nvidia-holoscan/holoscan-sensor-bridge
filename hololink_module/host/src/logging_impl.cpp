/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hololink/module/logging.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <string>

#include <unistd.h>

#include <fmt/core.h>

#if __GLIBC__ == 2 && __GLIBC_MINOR__ < 30
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)
#endif

namespace hololink::module {

// Env vars that set the level. HOLOLINK_LOG_LEVEL takes precedence so a
// Hololink application can raise verbosity without disturbing Holoscan;
// HOLOSCAN_LOG_LEVEL is honored as the shared fallback.
static const char* const HOLOSCAN_LOG_LEVEL_ENV = "HOLOSCAN_LOG_LEVEL";
static const char* const HOLOLINK_LOG_LEVEL_ENV = "HOLOLINK_LOG_LEVEL";

static const char* level_name(LogLevel level)
{
    switch (level) {
    case LogLevel::Trace:
        return "TRACE";
    case LogLevel::Debug:
        return "DEBUG";
    case LogLevel::Info:
        return "INFO";
    case LogLevel::Warning:
        return "WARN";
    case LogLevel::Error:
        return "ERROR";
    }
    return "INVALID";
}

// Resolve the configured level from the environment, defaulting to Info.
// Throws on an unrecognized setting, matching the prior behavior.
static LogLevel resolve_level_from_env()
{
    const char* hololink_env = ::getenv(HOLOLINK_LOG_LEVEL_ENV);
    const char* holoscan_env = ::getenv(HOLOSCAN_LOG_LEVEL_ENV);
    const char* env_name = hololink_env ? HOLOLINK_LOG_LEVEL_ENV : HOLOSCAN_LOG_LEVEL_ENV;
    const char* value = hololink_env ? hololink_env : holoscan_env;
    if (!value) {
        return LogLevel::Info;
    }
    if (::strcasecmp(value, "TRACE") == 0) {
        return LogLevel::Trace;
    }
    if (::strcasecmp(value, "DEBUG") == 0) {
        return LogLevel::Debug;
    }
    if (::strcasecmp(value, "INFO") == 0) {
        return LogLevel::Info;
    }
    if (::strcasecmp(value, "WARN") == 0) {
        return LogLevel::Warning;
    }
    if (::strcasecmp(value, "ERROR") == 0) {
        return LogLevel::Error;
    }
    throw std::runtime_error(
        fmt::format("Invalid environment setting in \"{}\".", env_name));
}

// Monotonic seconds since the first log call — a relative timestamp for
// reading sequences of lines, not a wall clock.
static float log_timestamp_s()
{
    struct timespec now = { 0, 0 };
    float ts = 0;
    if (::clock_gettime(CLOCK_MONOTONIC, &now) == 0) {
        ts = now.tv_sec + now.tv_nsec / 1000000000.0f;
    }
    static float start_time = 0;
    if (start_time < 1) {
        start_time = ts;
    }
    return ts - start_time;
}

/* Host-side Logging implementation. Implements the latest
 * `LoggingInterfaceV_n` and (once V2 lands) is published under every
 * prior version's type_id by inheritance. Owns the process-wide log
 * level and writes decorated lines to stderr — the host framework's
 * own sink, with no dependency on legacy src/hololink/core. */
class Logging : public LoggingInterfaceV1 {
public:
    Logging()
        : level_(resolve_level_from_env())
    {
    }

    LogLevel level() const override
    {
        return level_;
    }

    void log(LogLevel level,
        const char* file, unsigned line, const char* function,
        const char* message) override
    {
        // Drop the directory portion of the source path.
        const char* basename = std::strrchr(file, '/');
        basename = basename ? basename + 1 : file;

        const float ts = log_timestamp_s();
        const long thread_id = static_cast<long>(gettid());
        const std::string msg = fmt::format("{} {:.4f} {}:{} {} tid={:#x} -- {}",
            level_name(level), ts, basename, line, function, thread_id, message);
        std::fprintf(stderr, "%s\n", msg.c_str());
    }

private:
    LogLevel level_;
};

std::shared_ptr<LoggingInterfaceV1> make_logging_impl()
{
    return std::make_shared<Logging>();
}

} // namespace hololink::module
