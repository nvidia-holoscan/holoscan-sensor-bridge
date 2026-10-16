/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Slot transitions with no verbs and no CUDA.
 */

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <stdexcept>

#include <hololink/operators/cpu_roce_transceiver/cpu_roce_frame_transmitter_internal.hpp>

namespace hololink::operators::frame_tx {
namespace {

    TEST(CpuRoceFrameTransmitterSlot, TimeoutStaysInFlightUntilCompletion)
    {
        std::uint8_t storage[128] = {};
        auto pool = Pool::create(1, storage, 128, 64);
        const auto index = pool->claim().value();
        ASSERT_TRUE(pool->enqueue(index, 40, 0, nullptr));
        std::uint32_t popped = 0;
        std::uint32_t bytes = 0;
        std::uint32_t trailer = 0;
        void* ready = nullptr;
        ASSERT_TRUE(pool->pop_queued(&popped, &bytes, &trailer, &ready));
        pool->mark_inflight(popped, 9, bytes);
        EXPECT_FALSE(pool->claim().has_value());
        EXPECT_EQ(pool->inflight(), 1u);

        bool success = false;
        EXPECT_TRUE(pool->retire(popped, true, &success));
        EXPECT_TRUE(success);
        const auto stats = pool->stats();
        EXPECT_EQ(stats.bytes, 40u);
        EXPECT_EQ(stats.frames, 1u);
        EXPECT_EQ(stats.last_psn, 9u);
        EXPECT_EQ(stats.errors, 0u);
        EXPECT_EQ(pool->inflight(), 0u);
        const auto again = pool->claim();
        ASSERT_TRUE(again.has_value());
        pool->release_loaned(*again);
    }

    TEST(CpuRoceFrameTransmitterSlot, StopDropsQueuedFrames)
    {
        std::uint8_t storage[256] = {};
        auto pool = Pool::create(2, storage, 128, 64);
        const auto first = pool->claim().value();
        const auto second = pool->claim().value();
        ASSERT_TRUE(pool->enqueue(first, 8, 0, nullptr));
        ASSERT_TRUE(pool->enqueue(second, 8, 0, nullptr));
        EXPECT_EQ(pool->drop_queued(), 2u);
        EXPECT_EQ(pool->stats().errors, 2u);
        EXPECT_EQ(pool->queued(), 0u);
        const auto again = pool->claim();
        ASSERT_TRUE(again.has_value());
        pool->release_loaned(*again);
    }

    TEST(CpuRoceFrameTransmitterSlot, SynchronousPostCountsBytes)
    {
        std::uint8_t storage[128] = {};
        auto pool = Pool::create(1, storage, 128, 64);
        const auto index = pool->claim().value();
        pool->mark_inflight(index, 4, 40);
        bool success = false;
        EXPECT_TRUE(pool->retire(index, true, &success));
        EXPECT_TRUE(success);
        EXPECT_EQ(pool->stats().bytes, 40u);
        EXPECT_EQ(pool->stats().frames, 1u);
    }

    TEST(CpuRoceFrameTransmitterSlot, CloseRetiresInFlightWithoutCountingSuccess)
    {
        std::uint8_t storage[128] = {};
        auto pool = Pool::create(1, storage, 128, 64);
        const auto index = pool->claim().value();
        ASSERT_TRUE(pool->enqueue(index, 16, 0, nullptr));
        std::uint32_t popped = 0;
        std::uint32_t bytes = 0;
        std::uint32_t trailer = 0;
        void* ready = nullptr;
        ASSERT_TRUE(pool->pop_queued(&popped, &bytes, &trailer, &ready));
        pool->mark_inflight(popped, 3, bytes);
        pool->retire_inflight();
        EXPECT_EQ(pool->inflight(), 0u);
        EXPECT_EQ(pool->stats().frames, 0u);
        EXPECT_EQ(pool->stats().bytes, 0u);
        const auto again = pool->claim();
        ASSERT_TRUE(again.has_value());
        pool->release_loaned(*again);
    }

    TEST(CpuRoceFrameTransmitterSlot, LateReleaseAfterTheFacadeIsDropped)
    {
        std::uint8_t storage[128] = {};
        std::weak_ptr<Pool> weak;
        std::uint32_t index = 0;
        {
            auto pool = Pool::create(1, storage, 128, 64);
            weak = pool;
            index = pool->claim().value();
            pool.reset();
            ASSERT_FALSE(weak.expired());
            weak.lock()->release_loaned(index);
        }
        EXPECT_TRUE(weak.expired());
    }

    TEST(CpuRoceFrameTransmitterSlot, ValidationErrorIsNotTerminal)
    {
        EXPECT_FALSE(failure_is_terminal(std::invalid_argument("length")));
        EXPECT_FALSE(failure_is_terminal(std::out_of_range("slot")));
        EXPECT_TRUE(failure_is_terminal(std::runtime_error("verbs")));
    }

} // namespace
} // namespace hololink::operators::frame_tx
