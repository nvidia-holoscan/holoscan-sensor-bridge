/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_LOGGING_HPP
#define HOLOLINK_MODULE_LOGGING_HPP

#include <memory>

#include <fmt/format.h>

#include "module.hpp"
#include "service.hpp"

namespace hololink::module {

enum class LogLevel : int {
    Trace = 10,
    Debug = 20,
    Info = 30,
    Warning = 40,
    Error = 50,
};

/* Process-wide log sink reached through the locator. The host
 * registers exactly one implementation; modules cache the looked-up
 * shared_ptr for the life of the .so and the HSB_LOG_* macros below
 * dispatch through whichever cached pointer this binary holds. */
class LoggingInterfaceV1 : public Service<LoggingInterfaceV1> {
public:
    static constexpr const char* type_id = "logging.v1";

    // Singleton: hides the inherited three-arg form, passes "" instance_id.
    static std::shared_ptr<LoggingInterfaceV1> get_service(
        std::shared_ptr<Module> module, bool allow_null = false)
    {
        return Service<LoggingInterfaceV1>::get_service(
            std::move(module), "", allow_null);
    }

    virtual ~LoggingInterfaceV1() = default;

    /* The current minimum level that should reach `log`. Callers (and
     * the HSB_LOG_* macros) compare against this before doing any
     * format work, so cheap-to-skip log sites stay cheap. */
    virtual LogLevel level() const = 0;

    /* Emit a fully-formatted log line. Implementations decorate it
     * with their own metadata (timestamp, thread id, etc.) and
     * deliver it to the configured sink. */
    virtual void log(LogLevel level,
        const char* file, unsigned line, const char* function,
        const char* message)
        = 0;
};

/* Per-binary cached pointer the HSB_LOG_* macros dispatch through.
 * Each binary that links hololink::module_runtime gets its own copy
 * via RTLD_LOCAL: the host sets it from Adapter; each loaded module
 * sets it inside hololink_module_init after fetching the logger via
 * LoggingInterfaceV1::get_service. */
extern LoggingInterfaceV1* hsb_logger_cache;

void set_hsb_logger_cache(LoggingInterfaceV1* logger);

} // namespace hololink::module

/* Source-path token for diagnostics (log lines, CUDA error messages, …).
 * Use HOLOLINK_MODULE_FILE rather than __FILE__ directly so the path shown
 * in diagnostics has a single seam. It defaults to __FILE__, which the host
 * CMake always renders project-root-relative via -fmacro-prefix-map (an
 * INTERFACE option on hololink::module_headers) — applied uniformly to
 * every translation unit and each target's precompiled header, so paths are
 * build-location-independent without per-source command-line defines (which
 * would diverge from the PCH and be dropped with -Winvalid-pch). The guard
 * leaves any pre-existing definition (e.g. supplied by the build system)
 * untouched. */
#ifndef HOLOLINK_MODULE_FILE
#define HOLOLINK_MODULE_FILE __FILE__
#endif

/* Caller-side level filter then format + dispatch. Skips all format
 * work when no logger is registered or the level is below threshold. */
#define HSB_LOG_FMT(_level, ...)                                                                    \
    do {                                                                                            \
        ::hololink::module::LoggingInterfaceV1* _hsb_logger = ::hololink::module::hsb_logger_cache; \
        if (_hsb_logger != nullptr                                                                  \
            && static_cast<int>(_level)                                                             \
                >= static_cast<int>(_hsb_logger->level())) {                                        \
            _hsb_logger->log(_level, HOLOLINK_MODULE_FILE, __LINE__,                                \
                static_cast<const char*>(__FUNCTION__),                                             \
                ::fmt::format(__VA_ARGS__).c_str());                                                \
        }                                                                                           \
    } while (0)

#define HSB_LOG_TRACE(...) HSB_LOG_FMT(::hololink::module::LogLevel::Trace, __VA_ARGS__)
#define HSB_LOG_DEBUG(...) HSB_LOG_FMT(::hololink::module::LogLevel::Debug, __VA_ARGS__)
#define HSB_LOG_INFO(...) HSB_LOG_FMT(::hololink::module::LogLevel::Info, __VA_ARGS__)
#define HSB_LOG_WARN(...) HSB_LOG_FMT(::hololink::module::LogLevel::Warning, __VA_ARGS__)
#define HSB_LOG_ERROR(...) HSB_LOG_FMT(::hololink::module::LogLevel::Error, __VA_ARGS__)

#endif // HOLOLINK_MODULE_LOGGING_HPP
