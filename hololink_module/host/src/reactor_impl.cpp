/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hololink/module/reactor.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <map>
#include <memory>
#include <mutex>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "hololink/module/logging.hpp" // HSB_LOG_*

namespace hololink::module {

// Compare two timespec values: returns -1, 0, or 1.
static int timespec_compare(const struct timespec& a, const struct timespec& b)
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

// Difference a - b in milliseconds.
static int timespec_diff_ms(const struct timespec& a, const struct timespec& b)
{
    long long diff_sec = a.tv_sec - b.tv_sec;
    long long diff_nsec = a.tv_nsec - b.tv_nsec;
    return static_cast<int>((diff_sec * 1000) + (diff_nsec / 1000000));
}

// Add a (possibly fractional) number of seconds to a timespec.
static struct timespec timespec_add_seconds(const struct timespec& ts, float seconds)
{
    struct timespec result = ts;
    long long add_nsec = static_cast<long long>(seconds * 1000000000);
    result.tv_sec += add_nsec / 1000000000;
    result.tv_nsec += add_nsec % 1000000000;
    if (result.tv_nsec >= 1000000000) {
        result.tv_sec += result.tv_nsec / 1000000000;
        result.tv_nsec %= 1000000000;
    }
    return result;
}

/* Concrete alarm token: it is the opaque ReactorV1::AlarmEntry handed back
 * to callers (so cancel_alarm can identify it) and also carries the
 * scheduling payload the poll loop needs. (Named *Impl because the
 * unqualified name AlarmEntry resolves to the inherited base inside
 * Reactor.) */
class AlarmEntryImpl : public ReactorV1::AlarmEntry {
public:
    struct timespec when;
    uint64_t sequence;
    std::shared_ptr<ReactorV1::Callback> callback;
};

/* Host-side Reactor: the ReactorV1 service backed by a dedicated poll
 * thread. poll() waits over a self-pipe plus the registered fds and
 * dispatches fd callbacks and expired alarms sequentially, so handlers
 * never overlap. Created once and never freed (see make_reactor_impl) —
 * the poll thread must not be joined during static destruction, when
 * callbacks or other statics it touches may already be gone. */
class Reactor : public ReactorV1 {
public:
    Reactor()
        : thread_()
    {
        int pipe_fds[2];
        if (pipe(pipe_fds) == -1) {
            throw std::runtime_error("Failed to create wakeup pipe: " + std::string(strerror(errno)));
        }
        wakeup_read_fd_ = pipe_fds[0];
        wakeup_write_fd_ = pipe_fds[1];
        // Non-blocking so a burst of wakeup() writes can never block (and
        // deadlock, since wakeup() runs under lock_) when the pipe fills.
        set_nonblocking(wakeup_read_fd_);
        set_nonblocking(wakeup_write_fd_);

        std::stringstream ss;
        ss << "reactor@" << std::hex << reinterpret_cast<uintptr_t>(this);
        name_ = ss.str();

        std::thread run_thread(&Reactor::run, this);
        thread_ = std::move(run_thread);
    }

    ~Reactor() override
    {
        // Normally unreachable — the singleton is intentionally leaked — but
        // kept correct for any non-leaked instance.
        shutdown();
        if (wakeup_read_fd_ >= 0) {
            close(wakeup_read_fd_);
        }
        if (wakeup_write_fd_ >= 0) {
            close(wakeup_write_fd_);
        }
    }

    struct timespec now() const override
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        HSB_LOG_TRACE("now() -> {}.{:09d}", ts.tv_sec, ts.tv_nsec);
        return ts;
    }

    void add_callback(std::shared_ptr<Callback> callback) override
    {
        struct timespec zero_time = { 0, 0 };
        add_alarm(zero_time, std::move(callback));
    }

    void add_fd_callback(int fd, std::shared_ptr<FdCallback> callback,
        short events) override
    {
        std::lock_guard<std::mutex> lock(lock_);
        fd_callbacks_[fd] = std::make_pair(std::move(callback), events);
        wakeup();
    }

    void remove_fd_callback(int fd) override
    {
        HSB_LOG_TRACE("remove_fd_callback(fd={})", fd);
        std::lock_guard<std::mutex> lock(lock_);
        fd_callbacks_.erase(fd);
        wakeup();
    }

    AlarmHandle add_alarm_s(float seconds, std::shared_ptr<Callback> callback) override
    {
        struct timespec current = now();
        struct timespec target = timespec_add_seconds(current, seconds);
        return add_alarm(target, std::move(callback));
    }

    AlarmHandle add_alarm(const struct timespec& when,
        std::shared_ptr<Callback> callback) override
    {
        auto entry = std::make_shared<AlarmEntryImpl>();
        entry->when = when;
        entry->sequence = alarms_added_.fetch_add(1);
        entry->callback = std::move(callback);

        {
            std::lock_guard<std::mutex> lock(lock_);
            alarms_.push_back(entry);

            // Keep alarms sorted by time, then by sequence number.
            std::sort(alarms_.begin(), alarms_.end(),
                [](const std::shared_ptr<AlarmEntryImpl>& a, const std::shared_ptr<AlarmEntryImpl>& b) {
                    int time_cmp = timespec_compare(a->when, b->when);
                    if (time_cmp != 0) {
                        return time_cmp < 0;
                    }
                    return a->sequence < b->sequence;
                });

            wakeup();
        }

        HSB_LOG_TRACE("add_alarm() -> handle={}, sequence={}",
            static_cast<const void*>(entry.get()), entry->sequence);
        return entry;
    }

    void cancel_alarm(AlarmHandle handle) override
    {
        HSB_LOG_TRACE("cancel_alarm(handle={})", static_cast<const void*>(handle.get()));
        if (!handle) {
            return;
        }
        std::lock_guard<std::mutex> lock(lock_);
        auto it = std::find_if(alarms_.begin(), alarms_.end(),
            [&](const std::shared_ptr<AlarmEntryImpl>& e) { return e.get() == handle.get(); });
        if (it != alarms_.end()) {
            alarms_.erase(it);
            wakeup();
        }
    }

    bool is_current_thread() const override
    {
        bool result = std::this_thread::get_id() == thread_.get_id();
        HSB_LOG_TRACE("is_current_thread() -> {}", result);
        return result;
    }

    // Stop the poll thread and join it. Called during Adapter teardown,
    // while the services the thread's callbacks and logging use are still
    // alive, so the leaked, never-destructed reactor cannot run callbacks
    // or log through already-freed objects later in static destruction.
    // Pending alarms/fd callbacks are dropped so the final loop iteration
    // dispatches nothing. Idempotent.
    void shutdown()
    {
        {
            std::lock_guard<std::mutex> lock(lock_);
            running_.store(false);
            alarms_.clear();
            fd_callbacks_.clear();
        }
        wakeup();
        // Guard against a self-join: if a future callback ever triggers
        // teardown from the poll thread, joining it would throw.
        if (thread_.joinable() && !is_current_thread()) {
            thread_.join();
        }
    }

private:
    void run()
    {
        HSB_LOG_TRACE("Starting.");

        while (running_.load()) {
            // Compute the timeout until the next alarm.
            int timeout_ms = -1; // -1 means wait indefinitely

            {
                std::lock_guard<std::mutex> lock(lock_);
                if (!alarms_.empty()) {
                    struct timespec current_time = now();
                    struct timespec when = alarms_[0]->when;
                    if (timespec_compare(when, current_time) > 0) {
                        timeout_ms = timespec_diff_ms(when, current_time);
                        if (timeout_ms < 0)
                            timeout_ms = 0;
                    } else {
                        timeout_ms = 0; // Process immediately
                    }
                }
            }

            std::vector<struct pollfd> poll_fds;
            std::vector<int> fd_list;

            // Always watch the wakeup pipe.
            poll_fds.push_back({ wakeup_read_fd_, POLLIN, 0 });
            fd_list.push_back(wakeup_read_fd_);

            {
                std::lock_guard<std::mutex> lock(lock_);
                for (const auto& [fd, callback_data] : fd_callbacks_) {
                    poll_fds.push_back({ fd, callback_data.second, 0 });
                    fd_list.push_back(fd);
                }
            }

            HSB_LOG_TRACE("poll(..., timeout_ms={})", timeout_ms);
            int ready_count = poll(poll_fds.data(), poll_fds.size(), timeout_ms);
            HSB_LOG_TRACE("poll(...) ready_count={}", ready_count);

            if (ready_count == -1) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error("Poll failed: " + std::string(strerror(errno)));
            }

            for (size_t i = 0; i < poll_fds.size(); ++i) {
                if (poll_fds[i].revents == 0) {
                    continue;
                }
                int fd = fd_list[i];
                if (fd == wakeup_read_fd_) {
                    HSB_LOG_TRACE("poll(...) fd={} is wakeup_read_fd_", fd);
                    char buffer[1024];
                    ssize_t bytes_read;
                    do {
                        bytes_read = read(wakeup_read_fd_, buffer, sizeof(buffer));
                    } while (bytes_read == -1 && errno == EINTR);
                    // EAGAIN/EWOULDBLOCK just means the pipe is already drained.
                    (void)bytes_read;
                } else {
                    std::shared_ptr<FdCallback> callback;
                    {
                        std::lock_guard<std::mutex> lock(lock_);
                        auto it = fd_callbacks_.find(fd);
                        if (it != fd_callbacks_.end()) {
                            callback = it->second.first;
                        }
                    }
                    if (callback) {
                        (*callback)(fd, poll_fds[i].revents);
                    }
                }
            }

            // Collect expired alarms under the lock, dispatch outside it.
            std::vector<std::shared_ptr<AlarmEntryImpl>> expired;
            {
                std::lock_guard<std::mutex> lock(lock_);
                struct timespec current_time = now();

                while (!alarms_.empty() && timespec_compare(alarms_[0]->when, current_time) <= 0) {
                    expired.push_back(alarms_[0]);
                    alarms_.erase(alarms_.begin());
                }
            }

            for (const auto& entry : expired) {
                (*entry->callback)();
            }
        }

        HSB_LOG_TRACE("Done.");
    }

    void wakeup()
    {
        const char wake_msg[] = "wake-up\n";
        ssize_t bytes_written;
        do {
            bytes_written = write(wakeup_write_fd_, wake_msg, sizeof(wake_msg) - 1);
        } while (bytes_written == -1 && errno == EINTR);
        // EAGAIN/EWOULDBLOCK means the pipe is full of undrained wakeups, so
        // the poll thread is already guaranteed to wake; dropping this one is
        // harmless.
        (void)bytes_written;
    }

    static void set_nonblocking(int fd)
    {
        int flags = fcntl(fd, F_GETFL, 0);
        if (flags == -1) {
            throw std::runtime_error(
                "fcntl(F_GETFL) failed: " + std::string(strerror(errno)));
        }
        if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) == -1) {
            throw std::runtime_error(
                "fcntl(F_SETFL) failed: " + std::string(strerror(errno)));
        }
    }

    mutable std::mutex lock_;
    std::vector<std::shared_ptr<AlarmEntryImpl>> alarms_;
    std::atomic<uint64_t> alarms_added_ { 0 };
    std::map<int, std::pair<std::shared_ptr<FdCallback>, short>> fd_callbacks_;

    int wakeup_read_fd_;
    int wakeup_write_fd_;

    std::thread thread_;
    std::atomic<bool> running_ { true };
    std::string name_;
};

// The reactor is a process singleton that lives for the whole application.
// One reference is leaked (a heap shared_ptr that is never deleted) so the
// object is never destructed — avoiding destruction-order issues with the
// poll thread. Its thread is instead stopped explicitly via
// shutdown_reactor_impl() during Adapter teardown.
static const std::shared_ptr<Reactor>& reactor_singleton()
{
    static std::shared_ptr<Reactor>* const leaked
        = new std::shared_ptr<Reactor>(std::make_shared<Reactor>());
    return *leaked;
}

std::shared_ptr<ReactorV1> make_reactor_impl()
{
    return reactor_singleton();
}

void shutdown_reactor_impl()
{
    reactor_singleton()->shutdown();
}

} // namespace hololink::module
