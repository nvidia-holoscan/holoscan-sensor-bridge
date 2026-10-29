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

#pragma once

#include <cstdint>

namespace hololink::module::operators {

/**
 * Decides when a run of empty receive polls is worth reporting.
 *
 * Every receiver polls for a frame and gets nothing from time to time; that is
 * routine and must not be logged per occurrence. A *run* of empty polls is
 * different: downstream sees only a pipeline that stopped producing, so the
 * receiver is the only place that can name the cause. This reports the edges --
 * once when a stall starts, once when it ends -- so a log distinguishes a hiccup
 * from a wedge without a line per poll.
 *
 * The threshold is a duration rather than a count because receivers poll at very
 * different rates (100ms to 1500ms across this codebase), and "no data for a
 * second" means the same thing at every rate while "three empty polls" does not.
 *
 * Callers own the wording, so each receiver can name what it is waiting for; the
 * policy of when to speak lives here. Not thread-safe: like the frame counters it
 * sits beside, it belongs to the one thread doing the polling.
 */
class IngressStall {
public:
    static constexpr uint64_t DEFAULT_WARN_AFTER_MS = 1000;

    explicit IngressStall(uint64_t warn_after_ms = DEFAULT_WARN_AFTER_MS)
        : warn_after_ms_(warn_after_ms)
    {
    }

    /**
     * Record an empty poll of the given duration. Returns true exactly once per
     * stall, on the poll that crosses the threshold -- the caller should log a
     * warning then, and say nothing on the polls that return false.
     */
    bool on_timeout(uint64_t poll_ms)
    {
        ++consecutive_timeouts_;
        consecutive_timeout_ms_ += poll_ms;
        if (reported_ || consecutive_timeout_ms_ < warn_after_ms_) {
            return false;
        }
        reported_ = true;
        return true;
    }

    /**
     * Record a received frame. Returns the length of the stall it just ended, or
     * 0 if no stall was being reported -- so a caller can log the recovery and
     * say how long it lasted. Always clears the run, so the next empty poll
     * starts counting from zero rather than re-crossing the threshold at once.
     */
    uint64_t on_frame()
    {
        ++frames_received_;
        const uint64_t closed = reported_ ? consecutive_timeouts_ : 0;
        consecutive_timeouts_ = 0;
        consecutive_timeout_ms_ = 0;
        reported_ = false;
        return closed;
    }

    /** Empty polls since the last received frame. */
    uint64_t consecutive_timeouts() const { return consecutive_timeouts_; }

    /** Total duration represented by the empty polls since the last frame. */
    uint64_t consecutive_timeout_ms() const { return consecutive_timeout_ms_; }

    /**
     * Frames received since the last reset. Worth putting in the stall message:
     * it separates "never delivered anything" from "stopped after N frames".
     */
    uint64_t frames_received() const { return frames_received_; }

    /** Whether a stall is currently being reported. */
    bool stalled() const { return reported_; }

    /** Forget any stall in progress and the frame count, e.g. on a restart. */
    void reset()
    {
        consecutive_timeouts_ = 0;
        consecutive_timeout_ms_ = 0;
        frames_received_ = 0;
        reported_ = false;
    }

private:
    uint64_t warn_after_ms_ = DEFAULT_WARN_AFTER_MS;
    uint64_t consecutive_timeouts_ = 0;
    uint64_t consecutive_timeout_ms_ = 0;
    uint64_t frames_received_ = 0;
    bool reported_ = false;
};

} // namespace hololink::module::operators
