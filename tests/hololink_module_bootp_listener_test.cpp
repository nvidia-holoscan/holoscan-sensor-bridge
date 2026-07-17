/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Smoke test for Adapter::start_bootp_listener / stop_bootp_listener.
 * Uses port=0 so the kernel assigns an ephemeral port — sidesteps any
 * collision with whatever else might happen to be bound on the test
 * box. Verifies the lifecycle calls don't throw, that repeat starts /
 * stops are idempotent, and that the listener can be restarted after
 * stop. End-to-end synthetic-bootp-packet exercise is a follow-up
 * (the bootp v2 packet format is large; the legacy parser the listener
 * delegates to is already covered by hololink::Enumerator's own tests).
 */

#include <gtest/gtest.h>

#include "hololink/module/adapter.hpp"

TEST(HololinkAdapterBootpListener, StartStopIsIdempotent)
{
    auto& adapter = hololink::module::Adapter::get_adapter();

    // The Adapter constructor already started the listener on the
    // default port. Stop it so we can drive the start/stop lifecycle
    // explicitly against an ephemeral port (avoids colliding with
    // anything else on the test box).
    EXPECT_NO_THROW(adapter.stop_bootp_listener());

    // First start: should bind an ephemeral port. Second start: no-op
    // (the listener is already running).
    EXPECT_NO_THROW(adapter.start_bootp_listener(0));
    EXPECT_NO_THROW(adapter.start_bootp_listener(0));

    // Stop once, then again — both should be no-ops, no exception.
    EXPECT_NO_THROW(adapter.stop_bootp_listener());
    EXPECT_NO_THROW(adapter.stop_bootp_listener());

    // Restart after stop to prove stop fully cleaned up.
    EXPECT_NO_THROW(adapter.start_bootp_listener(0));
    EXPECT_NO_THROW(adapter.stop_bootp_listener());
}
