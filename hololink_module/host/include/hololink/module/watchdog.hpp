/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_WATCHDOG_HPP
#define HOLOLINK_MODULE_WATCHDOG_HPP

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>

#include "hololink/module/reactor.hpp"

namespace hololink::module {

/* A deadline timer on ReactorV1. arm() (or tap()) starts or refreshes the
 * deadline; if it elapses without a tap, on_timeout fires once on the
 * reactor thread (re-arm from there to keep watching). Tap on every event
 * you expect — e.g. a received frame — so a gap longer than timeout_s is
 * treated as a loss. timeout_s <= 0 disables it.
 *
 * Thread-safe: arm / tap / cancel may be called from any thread while the
 * timeout dispatches on the reactor thread.
 *  - A generation counter drops a timeout that fired just as a concurrent
 *    tap/cancel superseded it (the alarm slipped past cancel_alarm into
 *    the reactor's dispatch batch), so an event landing exactly at the
 *    deadline doesn't cause a spurious timeout.
 *  - The alarm callback holds the shared state, not the Watchdog, and
 *    checks an `alive` flag, so a timeout in flight when the Watchdog is
 *    destroyed is a no-op rather than a use-after-free.
 *  - Deadlock-free: ReactorV1::cancel_alarm only erases the pending entry
 *    (never waits on a running callback) and callbacks dispatch outside
 *    the reactor lock, so holding the Watchdog's mutex across arm/cancel
 *    can't cycle with the timeout path.
 *
 * on_timeout itself must outlive any possible firing; a caller that
 * captures `this` should cancel() (or destroy the Watchdog) before it is
 * torn down. */
class Watchdog {
public:
    Watchdog(std::shared_ptr<ReactorV1> reactor, double timeout_s,
        std::function<void()> on_timeout)
        : state_(std::make_shared<State>())
    {
        state_->reactor = std::move(reactor);
        state_->timeout_s = timeout_s;
        state_->on_timeout = std::move(on_timeout);
    }

    ~Watchdog() { stop(/*mark_dead=*/true); }

    Watchdog(const Watchdog&) = delete;
    Watchdog& operator=(const Watchdog&) = delete;

    // Start or refresh the deadline.
    void arm()
    {
        State& st = *state_;
        std::lock_guard<std::mutex> lock(st.mutex);
        cancel_pending(st);
        const uint64_t generation = ++st.generation;
        if (!st.reactor || st.timeout_s <= 0.0) {
            return;
        }
        // Capture the shared state (not `this`) so a firing that races
        // destruction finds live state and a false `alive`.
        std::shared_ptr<State> state = state_;
        auto callback = std::make_shared<ReactorV1::Callback>(
            [state, generation]() { on_alarm(state, generation); });
        st.handle = st.reactor->add_alarm_s(
            static_cast<float>(st.timeout_s), callback);
    }

    // A watched event happened: refresh the deadline.
    void tap() { arm(); }

    // Stop watching; no further timeout fires.
    void cancel() { stop(/*mark_dead=*/false); }

private:
    struct State {
        std::mutex mutex;
        std::shared_ptr<ReactorV1> reactor;
        double timeout_s = 0.0;
        std::function<void()> on_timeout;
        ReactorV1::AlarmHandle handle;
        uint64_t generation = 0;
        bool alive = true;
    };

    // Caller holds st.mutex.
    static void cancel_pending(State& st)
    {
        if (st.reactor && st.handle) {
            st.reactor->cancel_alarm(st.handle);
        }
        st.handle.reset();
    }

    void stop(bool mark_dead)
    {
        State& st = *state_;
        std::lock_guard<std::mutex> lock(st.mutex);
        ++st.generation;
        if (mark_dead) {
            st.alive = false;
        }
        cancel_pending(st);
    }

    static void on_alarm(std::shared_ptr<State> state, uint64_t generation)
    {
        std::function<void()> on_timeout;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (!state->alive || generation != state->generation) {
                return;
            }
            state->handle.reset();
            on_timeout = state->on_timeout;
        }
        if (on_timeout) {
            on_timeout();
        }
    }

    std::shared_ptr<State> state_;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_WATCHDOG_HPP
