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

#include <gtest/gtest.h>

#include <hololink/module/operators/ingress_stall.hpp>

using hololink::module::operators::IngressStall;

// A poll that returns nothing now and then is routine and must stay silent.
TEST(IngressStall, IsSilentBelowTheThreshold)
{
    IngressStall stall(1000);

    EXPECT_FALSE(stall.on_timeout(100));
    EXPECT_FALSE(stall.on_timeout(100));
    EXPECT_EQ(stall.on_frame(), 0u);
    EXPECT_FALSE(stall.stalled());
}

// Crossing the threshold reports once, not once per poll.
TEST(IngressStall, ReportsOnceWhenTheThresholdIsCrossed)
{
    IngressStall stall(1000);

    for (int i = 0; i < 9; i++) {
        EXPECT_FALSE(stall.on_timeout(100)) << "poll " << i;
    }
    EXPECT_TRUE(stall.on_timeout(100)); // 10 x 100ms == 1000ms
    EXPECT_TRUE(stall.stalled());
    EXPECT_EQ(stall.consecutive_timeouts(), 10u);

    for (int i = 0; i < 20; i++) {
        EXPECT_FALSE(stall.on_timeout(100)) << "poll " << i;
    }
    EXPECT_EQ(stall.consecutive_timeouts(), 30u);
}

// The threshold is a duration, so a slow poll reports on its first empty poll
// and a fast one takes many -- "no data for a second" at either rate.
TEST(IngressStall, ThresholdIsADurationNotACount)
{
    IngressStall slow(1000);
    EXPECT_TRUE(slow.on_timeout(1500));

    IngressStall fast(1000);
    for (int i = 0; i < 99; i++) {
        EXPECT_FALSE(fast.on_timeout(10));
    }
    EXPECT_TRUE(fast.on_timeout(10));
}

// Accumulate the actual duration of each empty poll rather than assuming every
// poll has the duration of the most recent one.
TEST(IngressStall, AccumulatesMixedPollDurations)
{
    IngressStall stall(1000);

    EXPECT_FALSE(stall.on_timeout(200));
    EXPECT_FALSE(stall.on_timeout(300));
    EXPECT_EQ(stall.consecutive_timeout_ms(), 500u);
    EXPECT_TRUE(stall.on_timeout(500));
    EXPECT_EQ(stall.consecutive_timeout_ms(), 1000u);

    stall.on_frame();
    EXPECT_EQ(stall.consecutive_timeout_ms(), 0u);
}

// A frame ends the stall and hands back its length for the recovery message.
TEST(IngressStall, FrameClosesTheStallAndReportsItsLength)
{
    IngressStall stall(1000);

    EXPECT_TRUE(stall.on_timeout(1500));
    EXPECT_FALSE(stall.on_timeout(1500));
    EXPECT_EQ(stall.on_frame(), 2u);
    EXPECT_FALSE(stall.stalled());
    EXPECT_EQ(stall.consecutive_timeouts(), 0u);
    EXPECT_EQ(stall.consecutive_timeout_ms(), 0u);
}

// Recovering must clear the run: otherwise the next single empty poll would
// re-cross the threshold immediately and warn about a stall that is not one.
TEST(IngressStall, RecoveryResetsTheRunSoOneEmptyPollIsQuiet)
{
    IngressStall stall(1000);

    for (int i = 0; i < 10; i++) {
        stall.on_timeout(100);
    }
    ASSERT_TRUE(stall.stalled());
    ASSERT_EQ(stall.on_frame(), 10u);

    EXPECT_FALSE(stall.on_timeout(100));
    EXPECT_EQ(stall.consecutive_timeouts(), 1u);
}

// A frame with no stall in progress has nothing to report.
TEST(IngressStall, FrameWithoutAStallReportsNothing)
{
    IngressStall stall(1000);

    EXPECT_EQ(stall.on_frame(), 0u);
    stall.on_timeout(100);
    EXPECT_EQ(stall.on_frame(), 0u);
}

// Two stalls in a run are each reported.
TEST(IngressStall, ReportsEverySeparateStall)
{
    IngressStall stall(1000);

    EXPECT_TRUE(stall.on_timeout(1000));
    EXPECT_EQ(stall.on_frame(), 1u);
    EXPECT_TRUE(stall.on_timeout(1000));
    EXPECT_EQ(stall.on_frame(), 1u);
}

// The frame count separates "never delivered anything" from "stopped after N".
TEST(IngressStall, CountsFramesReceived)
{
    IngressStall stall(1000);

    EXPECT_EQ(stall.frames_received(), 0u);
    stall.on_frame();
    stall.on_frame();
    EXPECT_EQ(stall.frames_received(), 2u);

    stall.on_timeout(1500);
    EXPECT_EQ(stall.frames_received(), 2u);
}

// reset() drops a stall in progress, e.g. across a receiver restart.
TEST(IngressStall, ResetDropsAStallInProgress)
{
    IngressStall stall(1000);

    EXPECT_TRUE(stall.on_timeout(1500));
    stall.reset();
    EXPECT_FALSE(stall.stalled());
    EXPECT_EQ(stall.consecutive_timeouts(), 0u);
    EXPECT_EQ(stall.on_frame(), 0u);

    stall.reset();
    EXPECT_EQ(stall.frames_received(), 0u);
}
