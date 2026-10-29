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

/**
 * @brief Detects when an expected event stops happening.
 *
 * A Watchdog is a deadline timer that runs on the ReactorV1. While it is
 * armed, each call to tap() restarts the countdown. If the countdown
 * reaches zero before the next tap, the Watchdog calls its
 * `on_timeout` function once, on the reactor thread, and then waits
 * until it is armed again. Call tap() each time the event being watched
 * occurs (for example, a frame arrives) so that a gap longer than the
 * timeout is reported as a loss.
 *
 * Every method may be called from any thread. `on_timeout` runs on the
 * reactor thread, with no Watchdog lock held, so it may itself call
 * arm(), tap() or cancel(); a typical `on_timeout` re-arms in order to
 * keep watching.
 *
 * A Watchdog cannot be copied.
 */
class Watchdog {
public:
    /**
     * @brief Create a Watchdog. It is idle until arm() or tap() is
     * called.
     *
     * @param reactor The reactor that supplies the timer and runs
     *      `on_timeout`. If null, the Watchdog never times out.
     * @param timeout_s Seconds without a tap before `on_timeout` is
     *      called. If zero or negative, the Watchdog is disabled and
     *      never times out.
     * @param on_timeout Function to call when the timeout elapses. It
     *      must remain valid for as long as a timeout can still occur,
     *      so if it refers to an object that is to be destroyed, call
     *      cancel() (or destroy the Watchdog) first.
     */
    Watchdog(std::shared_ptr<ReactorV1> reactor, double timeout_s,
        std::function<void()> on_timeout)
        : state_(std::make_shared<State>())
    {
        state_->reactor = std::move(reactor);
        state_->timeout_s = timeout_s;
        state_->on_timeout = std::move(on_timeout);
    }

    /**
     * @brief Cancel any pending timeout and destroy the Watchdog.
     *
     * No timeout starts after this returns. A call to `on_timeout` that
     * is already running on the reactor thread is not waited for.
     */
    ~Watchdog() { stop(/*mark_dead=*/true); }

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    Watchdog(const Watchdog&) = delete;
    Watchdog& operator=(const Watchdog&) = delete;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Start watching, or restart the countdown from the full
     * timeout.
     *
     * Replaces any countdown already in progress. Does nothing if the
     * Watchdog has no reactor or is disabled by a timeout of zero or
     * less.
     */
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

    /**
     * @brief Report that the watched event happened.
     *
     * Restarts the countdown, exactly as arm() does. It also starts
     * watching if the Watchdog was idle, including after a timeout or
     * cancel().
     *
     * If a timeout was about to be delivered at the same moment,
     * this prevents it, so an event arriving right at the deadline does
     * not cause a spurious timeout.
     */
    void tap() { arm(); }

    /**
     * @brief Stop watching.
     *
     * A timeout that has not yet begun is prevented; it is not delivered
     * even if it was about to fire. The Watchdog can be started again
     * with arm() or tap(). A call to `on_timeout` that is already running
     * on the reactor thread is not waited for.
     */
    void cancel() { stop(/*mark_dead=*/false); }

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
private:
    /* Concurrency notes:
     *  - A generation counter drops a timeout that fired just as a
     *    concurrent tap/cancel superseded it (the alarm slipped past
     *    cancel_alarm into the reactor's dispatch batch).
     *  - The alarm callback holds the shared state, not the Watchdog, and
     *    checks an `alive` flag, so a timeout in flight when the Watchdog
     *    is destroyed is a no-op rather than a use-after-free.
     *  - Deadlock-free: ReactorV1::cancel_alarm only erases the pending
     *    entry (never waits on a running callback) and callbacks dispatch
     *    outside the reactor lock, so holding the Watchdog's mutex across
     *    arm/cancel can't cycle with the timeout path. */
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
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_WATCHDOG_HPP
