/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// hololink::core::Reactor, which Enumerator::register_ip() runs on and Hololink::reset_framework()
// checks: callbacks and alarms (relative, absolute, cancelled, in order, added from several
// threads), fd callbacks (called, removed), and the reset_framework() check that tests/conftest.py
// runs after every test. These were the HSDK4 Python tests in tests/test_reactor.py; HSDK5 doesn't
// bind hololink.Reactor.

#include <gtest/gtest.h>

#include <poll.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "hololink/core/reactor.hpp"

namespace {

using hololink::core::Reactor;

// Generous: waits end as soon as their event happens, and only a lost callback or a stuck reactor
// should take this long.
constexpr auto kTimeout = std::chrono::seconds(10);

std::chrono::nanoseconds to_duration(const struct timespec& ts)
{
    return std::chrono::seconds(ts.tv_sec) + std::chrono::nanoseconds(ts.tv_nsec);
}

struct timespec later(struct timespec ts, std::chrono::nanoseconds delay)
{
    const auto total = to_duration(ts) + delay;
    const auto whole = std::chrono::duration_cast<std::chrono::seconds>(total);
    ts.tv_sec = static_cast<time_t>(whole.count());
    ts.tv_nsec = static_cast<long>((total - whole).count());
    return ts;
}

// Returns once the reactor has run everything queued before this call.
void drain(Reactor& reactor)
{
    auto done = std::make_shared<std::promise<void>>();
    auto future = done->get_future();
    reactor.add_callback([done] { done->set_value(); });
    ASSERT_EQ(future.wait_for(kTimeout), std::future_status::ready);
}

// While a Hold lives, the reactor thread waits inside one of its callbacks, so alarms added or
// cancelled meanwhile stay in its queue: none can fire before the test is done with them, however
// long the test thread is descheduled.
struct Hold {
    std::promise<void> release;
    ~Hold() { release.set_value(); }
};

std::unique_ptr<Hold> hold(Reactor& reactor)
{
    auto held = std::make_unique<Hold>();
    auto entered = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();
    reactor.add_callback([entered, release = held->release.get_future().share()] {
        entered->set_value();
        release.wait();
    });
    // Until the callback runs, an alarm that is already due could join the reactor's current batch,
    // where cancel_alarm() no longer finds it.
    EXPECT_EQ(entered_future.wait_for(kTimeout), std::future_status::ready);
    return held;
}

TEST(HololinkReactor, GetReactorReturnsTheSameInstance)
{
    auto first = Reactor::get_reactor();
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first.get(), Reactor::get_reactor().get());
}

TEST(HololinkReactor, NowIsAMonotonicClock)
{
    auto reactor = Reactor::get_reactor();
    const auto before = to_duration(reactor->now());
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    const auto elapsed = to_duration(reactor->now()) - before;
    EXPECT_GE(elapsed, std::chrono::milliseconds(10));
    EXPECT_LT(elapsed, kTimeout);
}

TEST(HololinkReactor, RunsACallbackOnTheReactorThread)
{
    auto reactor = Reactor::get_reactor();
    EXPECT_FALSE(reactor->is_current_thread());

    auto on_reactor_thread = std::make_shared<std::promise<bool>>();
    auto result = on_reactor_thread->get_future();
    reactor->add_callback([reactor, on_reactor_thread] {
        on_reactor_thread->set_value(reactor->is_current_thread());
    });
    ASSERT_EQ(result.wait_for(kTimeout), std::future_status::ready);
    EXPECT_TRUE(result.get());
}

TEST(HololinkReactor, FiresARelativeAlarmNoSoonerThanItsDelay)
{
    auto reactor = Reactor::get_reactor();
    auto fired = std::make_shared<std::promise<struct timespec>>();
    auto fired_at = fired->get_future();

    const auto start = to_duration(reactor->now());
    auto handle = reactor->add_alarm_s(0.1f, [reactor, fired] { fired->set_value(reactor->now()); });
    ASSERT_NE(handle, nullptr);
    ASSERT_EQ(fired_at.wait_for(kTimeout), std::future_status::ready);

    const auto elapsed = to_duration(fired_at.get()) - start;
    EXPECT_GE(elapsed, std::chrono::milliseconds(100));
    EXPECT_LT(elapsed, kTimeout);
    EXPECT_NO_THROW(reactor->reset_framework());
}

TEST(HololinkReactor, FiresAnAbsoluteAlarmNoSoonerThanItsTime)
{
    auto reactor = Reactor::get_reactor();
    auto fired = std::make_shared<std::promise<struct timespec>>();
    auto fired_at = fired->get_future();

    const struct timespec when = later(reactor->now(), std::chrono::milliseconds(100));
    reactor->add_alarm(when, [reactor, fired] { fired->set_value(reactor->now()); });
    ASSERT_EQ(fired_at.wait_for(kTimeout), std::future_status::ready);

    const auto late = to_duration(fired_at.get()) - to_duration(when);
    EXPECT_GE(late, std::chrono::nanoseconds(0));
    EXPECT_LT(late, kTimeout);
}

TEST(HololinkReactor, DoesNotFireACancelledAlarm)
{
    auto reactor = Reactor::get_reactor();
    auto fired = std::make_shared<std::atomic<bool>>(false);
    auto after = std::make_shared<std::promise<void>>();
    auto after_fired = after->get_future();
    {
        auto held = hold(*reactor);
        // Due now, so it would run as soon as the reactor is released.
        const struct timespec when = reactor->now();
        auto handle = reactor->add_alarm(when, [fired] { *fired = true; });
        reactor->cancel_alarm(handle);
        // Cancelling removed it, so nothing is left pending.
        EXPECT_NO_THROW(reactor->reset_framework());
        EXPECT_NO_THROW(reactor->cancel_alarm(handle));
        EXPECT_NO_THROW(reactor->cancel_alarm(nullptr));
        // Due at the same time, so it runs after the cancelled alarm would have.
        reactor->add_alarm(when, [after] { after->set_value(); });
    }
    ASSERT_EQ(after_fired.wait_for(kTimeout), std::future_status::ready);
    EXPECT_FALSE(*fired);
}

TEST(HololinkReactor, FiresAlarmsInTimeOrderThenInTheOrderAdded)
{
    auto reactor = Reactor::get_reactor();
    auto mutex = std::make_shared<std::mutex>();
    auto order = std::make_shared<std::vector<std::string>>();
    auto all_fired = std::make_shared<std::promise<void>>();
    auto done = all_fired->get_future();
    auto record = [mutex, order, all_fired](std::string name) {
        return [mutex, order, all_fired, name] {
            std::lock_guard lock(*mutex);
            order->push_back(name);
            if (order->size() == 5) {
                all_fired->set_value();
            }
        };
    };

    {
        // Added out of time order; the last two share one time.
        auto held = hold(*reactor);
        const struct timespec start = reactor->now();
        const struct timespec tie = later(start, std::chrono::milliseconds(200));
        reactor->add_alarm(later(start, std::chrono::milliseconds(150)), record("third"));
        reactor->add_alarm(later(start, std::chrono::milliseconds(50)), record("first"));
        reactor->add_alarm(later(start, std::chrono::milliseconds(100)), record("second"));
        reactor->add_alarm(tie, record("fourth"));
        reactor->add_alarm(tie, record("fifth"));
    }
    ASSERT_EQ(done.wait_for(kTimeout), std::future_status::ready);

    std::lock_guard lock(*mutex);
    EXPECT_EQ(*order, (std::vector<std::string> { "first", "second", "third", "fourth", "fifth" }));
}

TEST(HololinkReactor, CallsAnFdCallbackWhenTheFdIsReadable)
{
    auto reactor = Reactor::get_reactor();
    int fds[2];
    ASSERT_EQ(::pipe(fds), 0);

    auto called = std::make_shared<std::promise<std::tuple<int, short, std::string>>>();
    auto result = called->get_future();
    auto once = std::make_shared<std::once_flag>();
    reactor->add_fd_callback(
        fds[0],
        [called, once](int fd, short events) {
            char buffer[16];
            const ssize_t count = ::read(fd, buffer, sizeof(buffer));
            const std::string data(buffer, count > 0 ? static_cast<size_t>(count) : 0);
            std::call_once(*once, [&] { called->set_value(std::make_tuple(fd, events, data)); });
        },
        POLLIN);
    ASSERT_EQ(::write(fds[1], "test", 4), 4);
    const bool ready = result.wait_for(kTimeout) == std::future_status::ready;

    reactor->remove_fd_callback(fds[0]);
    // The reactor's next poll() leaves the fd out; only then close it.
    drain(*reactor);
    ::close(fds[0]);
    ::close(fds[1]);

    ASSERT_TRUE(ready);
    const auto [fd, events, data] = result.get();
    EXPECT_EQ(fd, fds[0]);
    EXPECT_TRUE(events & POLLIN);
    EXPECT_EQ(data, "test");
}

TEST(HololinkReactor, DoesNotCallARemovedFdCallback)
{
    auto reactor = Reactor::get_reactor();
    int fds[2];
    ASSERT_EQ(::pipe(fds), 0);

    auto called = std::make_shared<std::atomic<bool>>(false);
    reactor->add_fd_callback(
        fds[0], [called](int, short) { *called = true; }, POLLIN);
    reactor->remove_fd_callback(fds[0]);
    ASSERT_EQ(::write(fds[1], "test", 4), 4);
    // The reactor has polled since the write, and no longer polls the fd.
    drain(*reactor);
    ::close(fds[0]);
    ::close(fds[1]);

    EXPECT_FALSE(*called);
}

TEST(HololinkReactor, RunsCallbacksAndAlarmsAddedFromManyThreads)
{
    auto reactor = Reactor::get_reactor();
    constexpr int kThreads = 3;
    constexpr int kCallbacksPerThread = 10;
    // Each thread adds kCallbacksPerThread callbacks and one alarm.
    constexpr int kExpected = kThreads * (kCallbacksPerThread + 1);

    auto mutex = std::make_shared<std::mutex>();
    auto runs = std::make_shared<std::vector<int>>(kThreads, 0);
    auto total = std::make_shared<int>(0);
    auto all_ran = std::make_shared<std::promise<void>>();
    auto done = all_ran->get_future();
    auto record = [mutex, runs, total, all_ran](int thread) {
        std::lock_guard lock(*mutex);
        ++(*runs)[thread];
        if (++*total == kExpected) {
            all_ran->set_value();
        }
    };

    std::vector<std::thread> threads;
    for (int thread = 0; thread < kThreads; ++thread) {
        threads.emplace_back([reactor, record, thread] {
            for (int i = 0; i < kCallbacksPerThread; ++i) {
                reactor->add_callback([record, thread] { record(thread); });
            }
            reactor->add_alarm_s(0.1f, [record, thread] { record(thread); });
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    ASSERT_EQ(done.wait_for(kTimeout), std::future_status::ready);

    std::lock_guard lock(*mutex);
    EXPECT_EQ(*runs, std::vector<int>(kThreads, kCallbacksPerThread + 1));
}

TEST(HololinkReactor, ResetFrameworkFailsWhileAnAlarmIsPending)
{
    auto reactor = Reactor::get_reactor();
    auto handle = reactor->add_alarm_s(10.0f, [] {});
    EXPECT_THROW(reactor->reset_framework(), std::runtime_error);
    reactor->cancel_alarm(handle);
    EXPECT_NO_THROW(reactor->reset_framework());
}

} // namespace
