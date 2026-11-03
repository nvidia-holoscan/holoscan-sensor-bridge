/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// SpiDaemonThread against a fake SPI daemon on the loopback interface; no board or VM needed.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>

#include <gtest/gtest.h>

#include <hololink/core/hololink.hpp>
#include <hololink/core/jesd.hpp>

namespace {

constexpr uint16_t kSpiDaemonPort = 8400;
constexpr uint8_t kJesdMessage = 1;
// JESD204_OP_LINK_SETUP in SpiDaemonThread's state list; dispatched to JESDConfig::configure().
constexpr uint8_t kJesdLinkSetup = 7;

class FakeJesdConfig : public hololink::JESDConfig {
public:
    void power_on() override { }
    void setup_clocks() override { }
    void configure() override { configured = true; }
    void run() override { }

    std::atomic<bool> configured { false };
};

// Connects the way the VM's SPI daemon does, retrying until the thread listens.
int connect_daemon()
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
        const int fd = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in address {};
        address.sin_family = AF_INET;
        address.sin_port = htons(kSpiDaemonPort);
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0) {
            // Fail rather than hang if another listener on the port took the connection.
            const timeval timeout { .tv_sec = 5, .tv_usec = 0 };
            setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
            return fd;
        }
        close(fd);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return -1;
}

class SpiDaemonThreadTest : public ::testing::Test {
protected:
    // Starts the thread and returns the connected fake daemon's socket.
    int start()
    {
        auto daemon = std::async(std::launch::async, connect_daemon);
        thread_ = std::make_unique<hololink::SpiDaemonThread>(hololink_, config_);
        thread_->run();
        return daemon.get();
    }

    // Never started: no I/O, and the fake daemon sends no SPI transactions.
    hololink::Hololink hololink_ { "127.0.0.1", 8192, "jesd_test", false };
    FakeJesdConfig config_;
    std::unique_ptr<hololink::SpiDaemonThread> thread_;
};

TEST_F(SpiDaemonThreadTest, DestroyWhileConnectedJoinsThread)
{
    const int daemon_fd = start();
    ASSERT_GE(daemon_fd, 0);

    // A JESD state transition: a 5-byte message (type, 4-byte union) and a 3-byte reply
    // (big-endian length 1, then JESD204_STATE_CHANGE_DONE).
    const uint8_t message[5] = { kJesdMessage, kJesdLinkSetup, 0, 0, 0 };
    ASSERT_EQ(send(daemon_fd, message, sizeof(message), 0), ssize_t(sizeof(message)));
    uint8_t reply[3] = {};
    ASSERT_EQ(recv(daemon_fd, reply, sizeof(reply), MSG_WAITALL), ssize_t(sizeof(reply)));
    EXPECT_EQ(reply[0], 0);
    EXPECT_EQ(reply[1], 1);
    EXPECT_EQ(reply[2], 1);
    EXPECT_TRUE(config_.configured);

    // The daemon stays connected, so the thread is blocked in recv(): destruction must wake and
    // join it (a joinable std::thread's destructor terminates the process).
    const auto begin = std::chrono::steady_clock::now();
    thread_.reset();
    EXPECT_LT(std::chrono::steady_clock::now() - begin, std::chrono::seconds(2));

    uint8_t byte = 0;
    EXPECT_EQ(recv(daemon_fd, &byte, 1, 0), 0) << "the daemon should see the connection close";
    close(daemon_fd);
}

TEST_F(SpiDaemonThreadTest, DestroyAfterDaemonDisconnects)
{
    const int daemon_fd = start();
    ASSERT_GE(daemon_fd, 0);
    EXPECT_FALSE(thread_->finished());
    close(daemon_fd);
    // The thread ends on its own; it's still joinable.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!thread_->finished() && (std::chrono::steady_clock::now() < deadline)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_TRUE(thread_->finished());
    thread_.reset();
}

TEST_F(SpiDaemonThreadTest, RunThrowsWhenItCantListen)
{
    // A listener without SO_REUSEPORT makes the thread's bind() fail.
    const int blocker = socket(AF_INET, SOCK_STREAM, 0);
    ASSERT_GE(blocker, 0);
    const int enable = 1;
    setsockopt(blocker, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));
    sockaddr_in address {};
    address.sin_family = AF_INET;
    address.sin_port = htons(kSpiDaemonPort);
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    ASSERT_EQ(bind(blocker, reinterpret_cast<sockaddr*>(&address), sizeof(address)), 0);
    ASSERT_EQ(listen(blocker, 1), 0);

    thread_ = std::make_unique<hololink::SpiDaemonThread>(hololink_, config_);
    EXPECT_THROW(thread_->run(), std::runtime_error);
    EXPECT_TRUE(thread_->finished());
    thread_.reset();
    close(blocker);
}

TEST_F(SpiDaemonThreadTest, StopIsRepeatable)
{
    const int daemon_fd = start();
    ASSERT_GE(daemon_fd, 0);
    thread_->stop();
    thread_->stop();
    thread_.reset();
    close(daemon_fd);
}

TEST_F(SpiDaemonThreadTest, DestroyWithoutRun)
{
    thread_ = std::make_unique<hololink::SpiDaemonThread>(hololink_, config_);
    thread_.reset();
}

} // namespace
