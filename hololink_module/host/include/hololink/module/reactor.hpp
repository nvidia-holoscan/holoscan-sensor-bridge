/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_REACTOR_HPP
#define HOLOLINK_MODULE_REACTOR_HPP

#include <ctime>
#include <functional>
#include <memory>

#include "module.hpp"
#include "service.hpp"

namespace hololink::module {

/**
 * @brief Reactor provides an event loop that runs deferred, timed and
 * file-descriptor-driven callbacks in a single thread.
 *
 * Reactor is a singleton object, owned by the host and shared to every
 * module. All callbacks it dispatches run one sequentially in a
 * dedicated thread, so callbacks never run concurrently with each other
 * and need no locking against one another.  Callbacks must return
 * promptly, since each one delays all the others, and must not let an
 * exception escape.
 *
 * Every method may be called from any thread, including from within a
 * callback. Pending callbacks and alarms are discarded when the host
 * shuts down.
 */
class ReactorV1 : public Service<ReactorV1> {
public:
    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    static constexpr const char* type_id = "reactor.v1";
    static constexpr const char* release_version = "2.7.0";
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Fetch the process-wide reactor from a module.
     *
     * There is only one reactor, so no instance identifier is needed.
     *
     * @param module The module to query.
     * @param allow_null If `false` (the default), throws
     *      std::runtime_error when the module doesn't provide a reactor
     *      (or @p module is null). If `true`, returns `nullptr` instead.
     * @returns A shared pointer to the reactor.
     */
    static std::shared_ptr<ReactorV1> get_service(
        std::shared_ptr<Module> module, bool allow_null = false)
    {
        return Service<ReactorV1>::get_service(
            std::move(module), "", allow_null);
    }

    /** A function run once on the reactor thread. */
    using Callback = std::function<void()>;

    /**
     * A function run on the reactor thread when a watched file descriptor
     * becomes ready. `fd` is the descriptor, and `events` is the set of
     * `poll(2)` events (`POLLIN`, `POLLHUP`, ...) that occurred on it.
     */
    using FdCallback = std::function<void(int fd, short events)>;

    /**
     * @brief Opaque identifier of a pending alarm.
     *
     * Obtained from add_alarm() or add_alarm_s(), and passed to
     * cancel_alarm(). Applications neither create nor derive from it.
     */
    class AlarmEntry {
    public:
        /// @cond INCLUDE_INTERNAL_DOCUMENTATION
        virtual ~AlarmEntry() = default;
        /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION
    };
    /** Handle to a pending alarm. */
    using AlarmHandle = std::shared_ptr<AlarmEntry>;

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    virtual ~ReactorV1() = default;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief The current time on the reactor's clock.
     *
     * This is a monotonic clock (not wall-clock time); it is the clock
     * add_alarm() interprets its `when` argument against.
     *
     * @returns The current time.
     */
    virtual struct timespec now() const = 0;

    /**
     * @brief Queues a callback on the reactor thread for
     *      execution as soon as possible.
     *
     * Callbacks queued this way run in the order they were added.
     *
     * @param callback The function to run. The reactor keeps it alive
     *      until it has run.
     */
    virtual void add_callback(std::shared_ptr<Callback> callback) = 0;

    /**
     * @brief Call a function whenever a file descriptor has events of
     * interest.
     *
     * Readiness is level-triggered, as for `poll(2)`: the callback keeps
     * being called for as long as the condition persists, so it should
     * consume or clear it. There can be only one callback per file
     * descriptor; adding another for the same `fd` replaces the earlier
     * one. The reactor does not own or close `fd`.
     *
     * @param fd The file descriptor to watch.
     * @param callback The function to call. The reactor keeps it alive
     *      until it is removed or replaced and any call already in
     *      progress has finished.
     * @param events The `poll(2)` events to wait for, e.g. `POLLIN`.
     */
    virtual void add_fd_callback(int fd, std::shared_ptr<FdCallback> callback,
        short events)
        = 0;

    /**
     * @brief Stop watching a file descriptor.
     *
     * After this returns no new calls to the file descriptor's callback
     * will be scheduled. If called from a thread other than the reactor thread, a
     * call that is already running may still be finishing. Does nothing
     * if `fd` is not being watched. Remove the callback before closing
     * `fd`.
     *
     * @param fd The file descriptor passed to add_fd_callback().
     */
    virtual void remove_fd_callback(int fd) = 0;

    /**
     * @brief Run a callback once, after a delay.
     *
     * @param seconds Delay from now, in seconds.
     * @param callback The function to run. The reactor keeps it alive
     *      until it has run or the alarm is cancelled.
     * @returns A handle for cancel_alarm().
     */
    virtual AlarmHandle add_alarm_s(float seconds,
        std::shared_ptr<Callback> callback)
        = 0;

    /**
     * @brief Run a callback once, at a given time.
     *
     * Alarms that are due run in order of their `when`.  A `when` that has
     * already passed runs as soon as possible.
     *
     * @param when Time to run at, on the clock reported by now().
     * @param callback The function to run. The reactor keeps it alive
     *      until it has run or the alarm is cancelled.
     * @returns A handle for cancel_alarm().
     */
    virtual AlarmHandle add_alarm(const struct timespec& when,
        std::shared_ptr<Callback> callback)
        = 0;

    /**
     * @brief Cancel a scheduled alarm.
     *
     * Does nothing if `handle` is null, or if the alarm has already run
     * or been cancelled.
     *
     * @param handle The handle returned when the alarm was added.
     */
    virtual void cancel_alarm(AlarmHandle handle) = 0;

    /**
     * @brief Determine if the caller is running in the reactor thread.
     *
     * @returns `true` when called in the reactor thread.
     */
    virtual bool is_current_thread() const = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_REACTOR_HPP
