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

/**
 * @brief Priority of a log message.
 *
 * The numeric values are ordered so that a message is considered enabled
 * when its level is greater than or equal to the sink's threshold
 * (see LoggingInterfaceV1::level).
 */
enum class LogLevel : int {
    Trace = 10, ///< Finest-grained diagnostics.
    Debug = 20, ///< Information useful when debugging.
    Info = 30, ///< Normal operational messages.
    Warning = 40, ///< Something unexpected that the program can continue past.
    Error = 50, ///< A failure.
};

/**
 * @brief Application-wide sink for log messages.
 *
 * The host application implements this interface (see logging_impl.cpp)
 * and publishes it to loaded modules; this way all the loaded modules send
 * their logging messages to the same application receiver.
 *
 * Code within this framework normally logs through the `HSB_LOG_TRACE`,
 * `HSB_LOG_DEBUG`, `HSB_LOG_INFO`, `HSB_LOG_WARN` and `HSB_LOG_ERROR`
 * macros, which compare against level() before formatting the message and
 * then call log() with the call site's file, line and function.
 */
class LoggingInterfaceV1 : public Service<LoggingInterfaceV1> {
public:
    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    static constexpr const char* type_id = "logging.v1";
    static constexpr const char* release_version = "2.7.0";
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Fetch the process-wide logger from a module.
     *
     * There is only one logger, so no instance identifier is needed.
     *
     * @param module The module to query.
     * @param allow_null If `false` (the default), throws
     *      std::runtime_error when the module doesn't provide a logger
     *      (or @p module is null). If `true`, returns `nullptr` instead.
     * @returns A shared pointer to the logger.
     */
    static std::shared_ptr<LoggingInterfaceV1> get_service(
        std::shared_ptr<Module> module, bool allow_null = false)
    {
        return Service<LoggingInterfaceV1>::get_service(
            std::move(module), "", allow_null);
    }

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    virtual ~LoggingInterfaceV1() = default;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Returns the current setting for log level filtering.
     *
     * A message should be sent to log() only if its level is greater than
     * or equal to this value. Callers check this before building the
     * message so that disabled log statements cost no formatting work.
     * log() does not itself discard messages below this threshold.
     *
     * @returns The current threshold.
     */
    virtual LogLevel level() const = 0;

    /**
     * @brief Deliver one fully formatted log message to the sink.
     *
     * The sink decorates the message with its own metadata (for example
     * severity, timestamp and thread) and writes it to its destination.
     * This may be called concurrently from multiple threads. All pointer
     * arguments must be non-null, NUL-terminated strings, and are only
     * valid for the duration of the call.
     *
     * @param level Severity of the message.
     * @param file Source file of the call site.
     * @param line Line number of the call site.
     * @param function Name of the function containing the call site.
     * @param message The text to log; no further formatting is applied.
     */
    virtual void log(LogLevel level,
        const char* file, unsigned line, const char* function,
        const char* message)
        = 0;
};

/// @cond INCLUDE_INTERNAL_DOCUMENTATION
/* Per-binary cached pointer the HSB_LOG_* macros dispatch through.
 * Each binary that links hololink::module_runtime gets its own copy
 * via RTLD_LOCAL: the host sets it from Adapter; each loaded module
 * sets it inside hololink_module_init after fetching the logger via
 * LoggingInterfaceV1::get_service. */
extern LoggingInterfaceV1* hsb_logger_cache;

void set_hsb_logger_cache(LoggingInterfaceV1* logger);
/** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

} // namespace hololink::module

/// @cond INCLUDE_INTERNAL_DOCUMENTATION
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
/** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

#define HSB_LOG_TRACE(...) HSB_LOG_FMT(::hololink::module::LogLevel::Trace, __VA_ARGS__)
#define HSB_LOG_DEBUG(...) HSB_LOG_FMT(::hololink::module::LogLevel::Debug, __VA_ARGS__)
#define HSB_LOG_INFO(...) HSB_LOG_FMT(::hololink::module::LogLevel::Info, __VA_ARGS__)
#define HSB_LOG_WARN(...) HSB_LOG_FMT(::hololink::module::LogLevel::Warning, __VA_ARGS__)
#define HSB_LOG_ERROR(...) HSB_LOG_FMT(::hololink::module::LogLevel::Error, __VA_ARGS__)

#endif // HOLOLINK_MODULE_LOGGING_HPP
