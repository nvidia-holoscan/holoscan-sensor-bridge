/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Loads a stub module that fetches the host's Reactor + Logging
 * singletons, schedules a Reactor callback, and emits a log line
 * from inside that callback. Verifies (a) the callback executes on
 * the host's Reactor thread and (b) the log line arrives at the
 * host-installed sink with its file/line/function decoration intact.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#include "hololink/core/logging.hpp"
#include "hololink/module/adapter.hpp"

static std::filesystem::path module_path(const char* basename)
{
    const char* dir = std::getenv("HOLOLINK_MODULE_TEST_MODULE_DIR");
    if (!dir) {
        return std::filesystem::path(basename);
    }
    return std::filesystem::path(dir) / basename;
}

namespace {

struct CapturedLog {
    std::string file;
    unsigned line;
    std::string function;
    hololink::logging::HsbLogLevel level;
    std::string message;
};

} // namespace

static std::mutex g_log_mutex;
static std::condition_variable g_log_cv;
static std::vector<CapturedLog> g_log_records;

static void capture_logger(char const* file, unsigned line, const char* function,
    hololink::logging::HsbLogLevel level, const char* message)
{
    std::lock_guard<std::mutex> guard(g_log_mutex);
    g_log_records.push_back(CapturedLog { file, line, function, level, message });
    g_log_cv.notify_all();
}

TEST(HololinkAdapterSingletons, ReactorCallbackEmitsLog)
{
    using hololink::module::Adapter;

    // Install a capture sink on the host's logger and ensure the
    // level lets info lines through. Save the previous values to
    // restore at the end.
    auto previous_logger = hololink::logging::hsb_logger;
    auto previous_level = hololink::logging::hsb_log_level;
    hololink::logging::hsb_logger = &capture_logger;
    hololink::logging::hsb_log_level = hololink::logging::HSB_LOG_LEVEL_INFO;

    {
        std::lock_guard<std::mutex> guard(g_log_mutex);
        g_log_records.clear();
    }

    auto module = Adapter::get_adapter().load_module(
        module_path("hololink_module_singletons_stub_module.so"));
    ASSERT_NE(module, nullptr);

    // The stub schedules its add_callback during init; the host's
    // Reactor thread should run it shortly after. Wait up to a few
    // seconds for the marker line to appear.
    constexpr auto MARKER = "hololink_module_test::singletons-marker";
    bool got_marker = false;
    {
        std::unique_lock<std::mutex> lock(g_log_mutex);
        got_marker = g_log_cv.wait_for(lock, std::chrono::seconds(5), []() {
            return std::any_of(g_log_records.begin(), g_log_records.end(),
                [](const CapturedLog& r) {
                    return r.message.find(MARKER) != std::string::npos;
                });
        });
    }
    EXPECT_TRUE(got_marker);

    // The line arrived through the wrapped sink, so file / function
    // are populated by the HSB_LOG_INFO macro in the .so source.
    if (got_marker) {
        std::lock_guard<std::mutex> guard(g_log_mutex);
        const auto record = std::find_if(g_log_records.begin(), g_log_records.end(),
            [](const CapturedLog& r) { return r.message.find(MARKER) != std::string::npos; });
        ASSERT_NE(record, g_log_records.end());
        EXPECT_EQ(record->level, hololink::logging::HSB_LOG_LEVEL_INFO);
        EXPECT_NE(record->file, std::string());
        EXPECT_NE(record->function, std::string());
    }

    hololink::logging::hsb_logger = previous_logger;
    hololink::logging::hsb_log_level = previous_level;
}
