/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The hololink_module reactor hands a poll result only to the registration it polled. An fd can
// be removed and closed while the reactor runs another fd's callback, and its number reused by a
// new registration; the old fd's readiness, from the same poll(), must not reach the new one --
// a new callback, or the same callback registered again.

#include <gtest/gtest.h>

#include <fcntl.h>
#include <poll.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "hololink/module/adapter.hpp"
#include "hololink/module/reactor.hpp"

namespace hololink::module {
// The process reactor, as the Adapter gets it (reactor_impl.cpp).
std::shared_ptr<ReactorV1> make_reactor_impl();
} // namespace hololink::module

namespace {

using hololink::module::ReactorV1;

struct Pipe {
    int read_fd = -1;
    int write_fd = -1;

    Pipe()
    {
        int fds[2];
        if (::pipe2(fds, O_NONBLOCK) != 0) {
            throw std::runtime_error("pipe2 failed");
        }
        read_fd = fds[0];
        write_fd = fds[1];
    }
    Pipe(const Pipe&) = delete;
    Pipe& operator=(const Pipe&) = delete;
    ~Pipe() { close(); }

    void close()
    {
        if (read_fd >= 0) {
            ::close(read_fd);
            read_fd = -1;
        }
        if (write_fd >= 0) {
            ::close(write_fd);
            write_fd = -1;
        }
    }
};

// Calls of a callback that reads one byte per call: those that found one, and those that found
// none, which only a stale poll result can cause.
struct Reads {
    std::atomic<int> with_data { 0 };
    std::atomic<int> stale { 0 };
};

std::shared_ptr<ReactorV1::FdCallback> make_reader(const std::shared_ptr<Reads>& reads)
{
    return std::make_shared<ReactorV1::FdCallback>([reads](int fd, short) {
        char byte;
        if (::read(fd, &byte, 1) == 1) {
            reads->with_data++;
        } else {
            reads->stale++;
        }
    });
}

template <typename Predicate>
bool wait_for(Predicate predicate, std::chrono::milliseconds timeout)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

// Makes one poll() report both a and the earlier fd x readable, holds the reactor in x's callback,
// and meanwhile removes a, closes it and registers a new pipe under a's fd number, with the same
// callback when \p same_callback. The reactor must then skip a's result.
void check_reused_fd_number(bool same_callback)
{
    hololink::module::Adapter::get_adapter(); // sets up the reactor's logging
    auto reactor = hololink::module::make_reactor_impl();

    Pipe x;
    Pipe a;
    ASSERT_LT(x.read_fd, a.read_fd) << "the reactor dispatches in fd order; x must come first";

    std::mutex mutex;
    std::condition_variable condition;
    bool x_running = false;
    bool x_release = false;
    auto x_callback = std::make_shared<ReactorV1::FdCallback>([&](int fd, short) {
        char byte;
        (void)::read(fd, &byte, 1);
        std::unique_lock lock(mutex);
        x_running = true;
        condition.notify_all();
        condition.wait(lock, [&] { return x_release; });
    });
    auto reads = std::make_shared<Reads>();
    auto a_callback = make_reader(reads);
    reactor->add_fd_callback(x.read_fd, x_callback, POLLIN);
    reactor->add_fd_callback(a.read_fd, a_callback, POLLIN);

    // Make both readable while the reactor is busy in a callback, so its next poll() sees both.
    std::promise<void> gate_running;
    std::promise<void> gate_release;
    std::shared_future<void> released = gate_release.get_future().share();
    reactor->add_callback(std::make_shared<ReactorV1::Callback>([&gate_running, released] {
        gate_running.set_value();
        released.wait();
    }));
    ASSERT_EQ(gate_running.get_future().wait_for(std::chrono::seconds(5)), std::future_status::ready);
    ASSERT_EQ(::write(a.write_fd, "a", 1), 1);
    ASSERT_EQ(::write(x.write_fd, "x", 1), 1);
    gate_release.set_value();

    {
        std::unique_lock lock(mutex);
        ASSERT_TRUE(condition.wait_for(lock, std::chrono::seconds(5), [&] { return x_running; }))
            << "the reactor did not run x's callback";
    }

    // a's result from that poll() is next. Replace a under the same fd number (b is made while a
    // is open, so that it gets other numbers to move from).
    const int number = a.read_fd;
    Pipe b;
    reactor->remove_fd_callback(number);
    a.close();
    ASSERT_EQ(::dup2(b.read_fd, number), number);
    ::close(b.read_fd);
    b.read_fd = number;
    reactor->add_fd_callback(number, same_callback ? a_callback : make_reader(reads), POLLIN);

    {
        std::lock_guard lock(mutex);
        x_release = true;
    }
    condition.notify_all();

    // b is empty: a call now could only come from a's stale result.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_EQ(reads->stale.load(), 0) << "a's poll result reached the registration that replaced it";

    // b is watched: its data reaches its callback.
    ASSERT_EQ(::write(b.write_fd, "b", 1), 1);
    EXPECT_TRUE(wait_for([&] { return reads->with_data.load() == 1; }, std::chrono::seconds(5)));
    EXPECT_EQ(reads->stale.load(), 0);

    reactor->remove_fd_callback(number);
    reactor->remove_fd_callback(x.read_fd);
}

} // namespace

TEST(HololinkModuleReactor, SkipsTheResultOfAnFdReplacedByANewCallback)
{
    check_reused_fd_number(/*same_callback=*/false);
}

TEST(HololinkModuleReactor, SkipsTheResultOfAnFdReplacedByTheSameCallback)
{
    check_reused_fd_number(/*same_callback=*/true);
}
