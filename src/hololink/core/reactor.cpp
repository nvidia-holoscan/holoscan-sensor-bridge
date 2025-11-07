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
 */

#include "reactor.hpp"
#include "logging_internal.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

// Utility functions for timespec operations
namespace {
// Compare two timespec values: returns -1, 0, or 1
int timespec_compare(const struct timespec& a, const struct timespec& b)
{
    if (a.tv_sec < b.tv_sec)
        return -1;
    if (a.tv_sec > b.tv_sec)
        return 1;
    if (a.tv_nsec < b.tv_nsec)
        return -1;
    if (a.tv_nsec > b.tv_nsec)
        return 1;
    return 0;
}

// Subtract two timespec values and return difference in milliseconds
int timespec_diff_ms(const struct timespec& a, const struct timespec& b)
{
    long long diff_sec = a.tv_sec - b.tv_sec;
    long long diff_nsec = a.tv_nsec - b.tv_nsec;
    return static_cast<int>((diff_sec * 1000) + (diff_nsec / 1000000));
}

// Add seconds to timespec
struct timespec timespec_add_seconds(const struct timespec& ts, float seconds)
{
    struct timespec result = ts;
    long long add_nsec = static_cast<long long>(seconds * 1000000000);
    result.tv_sec += add_nsec / 1000000000;
    result.tv_nsec += add_nsec % 1000000000;

    // Handle nsec overflow
    if (result.tv_nsec >= 1000000000) {
        result.tv_sec += result.tv_nsec / 1000000000;
        result.tv_nsec %= 1000000000;
    }

    return result;
}
}

namespace hololink {
namespace core {

    // Static members
    std::shared_ptr<Reactor> Reactor::instance_;
    std::mutex Reactor::instance_mutex_;

    std::shared_ptr<Reactor> Reactor::get_reactor()
    {
        std::lock_guard<std::mutex> lock(instance_mutex_);

        if (!instance_) {
            instance_ = std::shared_ptr<Reactor>(new Reactor());

            // Start the reactor thread
            instance_->thread_ = std::make_unique<std::thread>(&Reactor::run, instance_.get());
        }

        return instance_;
    }

    Reactor::Reactor()
    {
        // Create the wakeup pipe
        int pipe_fds[2];
        if (pipe(pipe_fds) == -1) {
            throw std::runtime_error("Failed to create wakeup pipe: " + std::string(strerror(errno)));
        }

        wakeup_read_fd_ = pipe_fds[0];
        wakeup_write_fd_ = pipe_fds[1];

        // Generate a unique name for this reactor instance
        std::stringstream ss;
        ss << "reactor@" << std::hex << reinterpret_cast<uintptr_t>(this);
        name_ = ss.str();
    }

    Reactor::~Reactor()
    {
        if (running_.load()) {
            stop();
        }

        if (thread_ && thread_->joinable()) {
            thread_->join();
        }

        // Close the wakeup pipe
        if (wakeup_read_fd_ >= 0) {
            close(wakeup_read_fd_);
        }
        if (wakeup_write_fd_ >= 0) {
            close(wakeup_write_fd_);
        }
    }

    struct timespec Reactor::now() const
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        HSB_LOG_TRACE("now() -> {}.{:09d}", ts.tv_sec, ts.tv_nsec);
        return ts;
    }

    void Reactor::add_callback(Callback callback)
    {
        struct timespec zero_time = { 0, 0 };
        add_alarm(zero_time, callback);
    }

    void Reactor::add_fd_callback(int fd, FdCallback callback, short events)
    {
        std::lock_guard<std::mutex> lock(lock_);
        fd_callbacks_[fd] = std::make_pair(callback, events);
        wakeup();
    }

    void Reactor::remove_fd_callback(int fd)
    {
        HSB_LOG_TRACE("remove_fd_callback(fd={})", fd);
        std::lock_guard<std::mutex> lock(lock_);
        fd_callbacks_.erase(fd);
        wakeup();
    }

    Reactor::AlarmHandle Reactor::add_alarm_s(float seconds, Callback callback)
    {
        struct timespec current = now();
        struct timespec target = timespec_add_seconds(current, seconds);
        return add_alarm(target, callback);
    }

    Reactor::AlarmHandle Reactor::add_alarm(const struct timespec& when, Callback callback)
    {
        auto entry = std::make_shared<AlarmEntry>();
        entry->when = when;
        entry->sequence = alarms_added_.fetch_add(1);
        entry->callback = callback;

        {
            std::lock_guard<std::mutex> lock(lock_);
            alarms_.push_back(entry);

            // Keep alarms sorted by time, then by sequence number
            std::sort(alarms_.begin(), alarms_.end(),
                [](const std::shared_ptr<AlarmEntry>& a, const std::shared_ptr<AlarmEntry>& b) {
                    int time_cmp = timespec_compare(a->when, b->when);
                    if (time_cmp != 0) {
                        return time_cmp < 0;
                    }
                    return a->sequence < b->sequence;
                });

            wakeup();
        }

        HSB_LOG_TRACE("add_alarm() -> handle={}, sequence={}", static_cast<const void*>(entry.get()), entry->sequence);
        return entry;
    }

    void Reactor::cancel_alarm(AlarmHandle handle)
    {
        HSB_LOG_TRACE("cancel_alarm(handle={})", static_cast<const void*>(handle.get()));
        if (!handle) {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(lock_);
            auto it = std::find(alarms_.begin(), alarms_.end(), handle);
            if (it != alarms_.end()) {
                alarms_.erase(it);
                wakeup();
            }
        }
    }

    void Reactor::stop()
    {
        HSB_LOG_TRACE("stop()");
        add_callback([this]() { stop_impl(); });
    }

    bool Reactor::is_current_thread() const
    {
        bool result = thread_ && std::this_thread::get_id() == thread_->get_id();
        HSB_LOG_TRACE("is_current_thread() -> {}", result);
        return result;
    }

    void Reactor::run()
    {
        HSB_LOG_TRACE("Starting.");

        while (running_.load()) {
            // Calculate timeout for next alarm
            int timeout_ms = -1; // -1 means wait indefinitely

            {
                std::lock_guard<std::mutex> lock(lock_);
                if (!alarms_.empty()) {
                    struct timespec current_time = now();
                    struct timespec when = alarms_[0]->when;
                    if (timespec_compare(when, current_time) > 0) {
                        timeout_ms = timespec_diff_ms(when, current_time);
                        if (timeout_ms < 0)
                            timeout_ms = 0; // Handle edge case
                    } else {
                        timeout_ms = 0; // Process immediately
                    }
                }
            }

            // Build poll structures
            std::vector<struct pollfd> poll_fds;
            std::vector<int> fd_list;

            // Add wakeup pipe
            poll_fds.push_back({ wakeup_read_fd_, POLLIN, 0 });
            fd_list.push_back(wakeup_read_fd_);

            // Add registered file descriptors
            {
                std::lock_guard<std::mutex> lock(lock_);
                for (const auto& [fd, callback_data] : fd_callbacks_) {
                    poll_fds.push_back({ fd, callback_data.second, 0 }); // Use stored events
                    fd_list.push_back(fd);
                }
            }

            // Poll for ready file descriptors
            HSB_LOG_TRACE("poll(..., timeout_ms={})", timeout_ms);
            int ready_count = poll(poll_fds.data(), poll_fds.size(), timeout_ms);
            HSB_LOG_TRACE("poll(...) ready_count={}", ready_count);

            if (ready_count == -1) {
                if (errno == EINTR) {
                    continue; // Interrupted by signal, continue
                }
                throw std::runtime_error("Poll failed: " + std::string(strerror(errno)));
            }

            // Handle ready file descriptors
            for (size_t i = 0; i < poll_fds.size() && ready_count > 0; ++i) {
                if (poll_fds[i].revents != 0) {
                    ready_count--;
                    int fd = fd_list[i];

                    if (fd == wakeup_read_fd_) {
                        HSB_LOG_TRACE("poll(...) fd={} is wakeup_read_fd_", fd);
                        // Clear the wakeup pipe
                        char buffer[1024];
                        ssize_t bytes_read = read(wakeup_read_fd_, buffer, sizeof(buffer));
                        (void)bytes_read; // Suppress unused variable warning
                    } else {
                        // Call the callback for this file descriptor
                        FdCallback callback;
                        {
                            std::lock_guard<std::mutex> lock(lock_);
                            auto it = fd_callbacks_.find(fd);
                            if (it != fd_callbacks_.end()) {
                                callback = it->second.first; // Get callback from pair
                            }
                        }

                        if (callback) {
                            callback(fd, poll_fds[i].revents);
                        }
                    }
                }
            }

            // Process expired alarms
            std::vector<std::shared_ptr<AlarmEntry>> expired;
            {
                std::lock_guard<std::mutex> lock(lock_);
                struct timespec current_time = now();

                while (!alarms_.empty() && timespec_compare(alarms_[0]->when, current_time) <= 0) {
                    expired.push_back(alarms_[0]);
                    alarms_.erase(alarms_.begin());
                }
            }

            // Execute expired callbacks outside of lock to avoid deadlock
            for (const auto& entry : expired) {
                entry->callback();
            }
        }

        HSB_LOG_TRACE("Done.");
    }

    void Reactor::stop_impl()
    {
        running_.store(false);
    }

    void Reactor::wakeup()
    {
        const char wake_msg[] = "wake-up\n";
        ssize_t bytes_written = write(wakeup_write_fd_, wake_msg, sizeof(wake_msg) - 1);
        (void)bytes_written; // Suppress unused variable warning
    }

} // namespace core
} // namespace hololink
