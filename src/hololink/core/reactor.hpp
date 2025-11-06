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

#pragma once

#include <atomic>
#include <chrono>
#include <ctime>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace hololink {
namespace core {

    /**
     * @brief Event-driven reactor pattern implementation
     *
     * The Reactor class provides an event loop that can handle:
     * - Time-based callbacks (alarms)
     * - File descriptor events
     * - Arbitrary callbacks
     *
     * It runs in a dedicated background thread and provides thread-safe methods
     * for registering callbacks and alarms.
     */
    class Reactor {
    public:
        /**
         * @brief Alarm entry for the reactor
         */
        struct AlarmEntry {
            struct timespec when;
            uint64_t sequence;
            std::function<void()> callback;
        };

        using Callback = std::function<void()>;
        using FdCallback = std::function<void(int fd, short events)>;
        using AlarmHandle = std::shared_ptr<AlarmEntry>;

        /**
         * @brief Get the singleton reactor instance
         * @return Shared pointer to the reactor instance
         */
        static std::shared_ptr<Reactor> get_reactor();

    protected:
        /**
         * @brief Constructor - creates a new reactor instance
         */
        Reactor();

    public:
        /**
         * @brief Destructor - stops the reactor thread
         */
        ~Reactor();

        /**
         * @brief Get current monotonic time
         * @return Current time as struct timespec
         */
        struct timespec now() const;

        /**
         * @brief Add a callback to be executed immediately
         * @param callback Function to call
         */
        void add_callback(Callback callback);

        /**
         * @brief Add a file descriptor callback for poll events
         * @param fd File descriptor to monitor
         * @param callback Function to call when events occur
         * @param events Poll events to monitor (POLLIN by default)
         */
        void add_fd_callback(int fd, FdCallback callback, short events = 0x001); // POLLIN

        /**
         * @brief Remove a file descriptor callback
         * @param fd File descriptor to stop monitoring
         */
        void remove_fd_callback(int fd);

        /**
         * @brief Add an alarm to fire after specified seconds from now
         * @param seconds Seconds to wait before calling callback
         * @param callback Function to call
         * @return Handle that can be used to cancel the alarm
         */
        AlarmHandle add_alarm_s(float seconds, Callback callback);

        /**
         * @brief Add an alarm to fire at a specific time
         * @param when Time to fire the alarm
         * @param callback Function to call
         * @return Handle that can be used to cancel the alarm
         */
        AlarmHandle add_alarm(const struct timespec& when, Callback callback);

        /**
         * @brief Cancel a previously scheduled alarm
         * @param handle Handle returned from add_alarm or add_alarm_s
         */
        void cancel_alarm(AlarmHandle handle);

        /**
         * @brief Stop the reactor (gracefully)
         */
        void stop();

        /**
         * @brief Check if the current thread is the reactor thread
         * @return True if called from within the reactor thread
         */
        bool is_current_thread() const;

    private:
        /**
         * @brief Main reactor loop (runs in background thread)
         */
        void run();

        /**
         * @brief Stop callback for graceful shutdown
         */
        void stop_impl();

        /**
         * @brief Wake up the reactor thread by writing to the wakeup pipe
         */
        void wakeup();

        mutable std::mutex lock_;
        std::vector<std::shared_ptr<AlarmEntry>> alarms_;
        std::atomic<uint64_t> alarms_added_ { 0 };
        std::map<int, std::pair<FdCallback, short>> fd_callbacks_;

        int wakeup_read_fd_;
        int wakeup_write_fd_;

        std::unique_ptr<std::thread> thread_;
        std::atomic<bool> running_ { true };
        std::string name_;

        static std::shared_ptr<Reactor> instance_;
        static std::mutex instance_mutex_;
    };

} // namespace core
} // namespace hololink
