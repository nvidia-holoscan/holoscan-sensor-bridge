/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Loads a stub module that fetches the host's Reactor + Logging
 * singletons, schedules a Reactor callback, and emits a log line
 * from inside that callback. Verifies (a) the callback executes on
 * the host's Reactor thread and (b) the log line arrives at the host's
 * stderr sink with its file/line/function decoration intact.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include <unistd.h>

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

class StderrCapture {
public:
    StderrCapture()
    {
        int pipe_fds[2];
        if (::pipe(pipe_fds) != 0) {
            throw std::runtime_error("Unable to create stderr capture pipe");
        }
        read_fd_ = pipe_fds[0];
        saved_stderr_ = ::dup(STDERR_FILENO);
        if (saved_stderr_ < 0) {
            ::close(pipe_fds[0]);
            ::close(pipe_fds[1]);
            throw std::runtime_error("Unable to duplicate stderr");
        }
        if (::dup2(pipe_fds[1], STDERR_FILENO) != STDERR_FILENO) {
            ::close(saved_stderr_);
            saved_stderr_ = -1;
            ::close(pipe_fds[0]);
            ::close(pipe_fds[1]);
            throw std::runtime_error("Unable to redirect stderr");
        }
        ::close(pipe_fds[1]);
        try {
            reader_ = std::thread(&StderrCapture::drain, this);
        } catch (...) {
            ::dup2(saved_stderr_, STDERR_FILENO);
            ::close(saved_stderr_);
            saved_stderr_ = -1;
            ::close(read_fd_);
            read_fd_ = -1;
            throw;
        }
    }

    ~StderrCapture()
    {
        std::fflush(stderr);
        if (saved_stderr_ >= 0) {
            ::dup2(saved_stderr_, STDERR_FILENO);
            ::close(saved_stderr_);
        }
        if (reader_.joinable()) {
            reader_.join();
        }
        if (read_fd_ >= 0) {
            ::close(read_fd_);
        }
    }

    std::string read_until(const std::string& marker,
        std::chrono::milliseconds timeout)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait_for(lock, timeout, [this, &marker] {
            return captured_.find(marker) != std::string::npos || reader_done_;
        });
        return captured_;
    }

private:
    void drain()
    {
        char buffer[512];
        ssize_t count;
        while ((count = ::read(read_fd_, buffer, sizeof(buffer))) > 0) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                captured_.append(buffer, static_cast<std::size_t>(count));
            }
            condition_.notify_all();
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            reader_done_ = true;
        }
        condition_.notify_all();
    }

    int read_fd_ = -1;
    int saved_stderr_ = -1;
    std::thread reader_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::string captured_;
    bool reader_done_ = false;
};

} // namespace

TEST(HololinkAdapterSingletons, ReactorCallbackEmitsLog)
{
    using hololink::module::Adapter;

    StderrCapture capture;

    auto module = Adapter::get_adapter().load_module(
        module_path("hololink_module_singletons_stub_module.so"));
    ASSERT_NE(module, nullptr);

    // The stub schedules its add_callback during init; the host's
    // Reactor thread should run it shortly after. Wait up to a few
    // seconds for the marker line to appear.
    constexpr auto MARKER = "hololink_module_test::singletons-marker";
    const std::string captured = capture.read_until(MARKER, std::chrono::seconds(5));
    EXPECT_NE(captured.find(MARKER), std::string::npos);

    // The host logger decorates the line with severity, source, and function.
    EXPECT_NE(captured.find("INFO"), std::string::npos);
    EXPECT_NE(captured.find("hololink_module_singletons_stub_module.cpp"),
        std::string::npos);
    EXPECT_NE(captured.find("operator()"), std::string::npos);
}
