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

#ifndef SRC_HOLOLINK_CORE_NEWEST_FRAME_SLOT
#define SRC_HOLOLINK_CORE_NEWEST_FRAME_SLOT

#include <cstdint>
#include <utility>

namespace hololink::core {

/**
 * Hands frames from a receiver's capture thread to the compute() that publishes them, keeping
 * only the newest frame, with one notification per frame compute() can take.
 *
 * The capture thread store()s each frame it receives and compute() take()s the newest one. A
 * frame that a newer one replaces before compute() takes it is never published; take_replaced()
 * counts it. store() asks for a notification only when the slot was empty, so exactly one
 * notification is pending for each frame compute() can take. An OnNotified compute() runs once
 * per notification, so posting one per stored frame instead leaves one behind for every replaced
 * frame: each finds the slot empty, or, if compute() waits for the next frame, takes that one
 * early, and the receiver then runs that many frames ahead of its consumers for the rest of the
 * run.
 *
 * Not thread-safe: the owner calls it under the lock that guards its buffers.
 */
template <typename Frame>
class NewestFrameSlot {
public:
    struct Stored {
        // The frame this one replaced, for the caller to recycle; null if the slot was empty.
        Frame* replaced = nullptr;
        // The slot was empty: post one notification, so that compute() takes this frame.
        bool notify = false;
    };

    /** Capture thread: store \p frame, the newest. */
    Stored store(Frame* frame)
    {
        Stored stored { frame_, frame_ == nullptr };
        if (stored.replaced != nullptr) {
            ++replaced_;
        }
        frame_ = frame;
        return stored;
    }

    /** compute(): take the newest frame, emptying the slot; null if there is none. */
    Frame* take() { return std::exchange(frame_, nullptr); }

    /** The frame waiting for compute(), if any, without taking it. */
    Frame* peek() const { return frame_; }

    /** Frames replaced before compute() took them, since the last call. */
    uint64_t take_replaced() { return std::exchange(replaced_, 0); }

    /** Empty the slot and forget the count, e.g. on a restart. Returns the frame it held. */
    Frame* reset()
    {
        replaced_ = 0;
        return std::exchange(frame_, nullptr);
    }

private:
    Frame* frame_ = nullptr;
    uint64_t replaced_ = 0;
};

} // namespace hololink::core

#endif /* SRC_HOLOLINK_CORE_NEWEST_FRAME_SLOT */
