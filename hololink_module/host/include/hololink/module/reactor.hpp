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

/* Process-wide event-loop singleton. Every callback the Reactor
 * dispatches runs sequentially on a single thread, so handlers never
 * observe concurrent execution of one another and require no
 * internal locking against each other. The singleton is owned by
 * the host module and reachable from any module via the locator. */
class ReactorV1 : public Service<ReactorV1> {
public:
    static constexpr const char* type_id = "reactor.v1";

    // Singleton: hides the inherited three-arg form, passes "" instance_id.
    static std::shared_ptr<ReactorV1> get_service(
        std::shared_ptr<Module> module, bool allow_null = false)
    {
        return Service<ReactorV1>::get_service(
            std::move(module), "", allow_null);
    }

    using Callback = std::function<void()>;
    using FdCallback = std::function<void(int fd, short events)>;

    /* Opaque alarm token returned by add_alarm[_s] and accepted by
     * cancel_alarm. Implementations subclass this. */
    class AlarmEntry {
    public:
        virtual ~AlarmEntry() = default;
    };
    using AlarmHandle = std::shared_ptr<AlarmEntry>;

    virtual ~ReactorV1() = default;

    /* Current monotonic time. */
    virtual struct timespec now() const = 0;

    /* Queue a callback for the next time the reactor thread idles.
     * The Reactor holds the shared_ptr until dispatch completes. */
    virtual void add_callback(std::shared_ptr<Callback> callback) = 0;

    /* Watch fd for the supplied poll events and dispatch callback on
     * each event. The Reactor holds the shared_ptr until
     * remove_fd_callback returns AND any in-flight dispatch has
     * finished. */
    virtual void add_fd_callback(int fd, std::shared_ptr<FdCallback> callback,
        short events)
        = 0;
    virtual void remove_fd_callback(int fd) = 0;

    /* One-shot alarms. The Reactor holds the shared_ptr until
     * dispatch returns or cancel_alarm completes. */
    virtual AlarmHandle add_alarm_s(float seconds,
        std::shared_ptr<Callback> callback)
        = 0;
    virtual AlarmHandle add_alarm(const struct timespec& when,
        std::shared_ptr<Callback> callback)
        = 0;
    virtual void cancel_alarm(AlarmHandle handle) = 0;

    /* True iff called from the reactor's dispatch thread. Useful for
     * debug assertions in code that expects to run only on the
     * single sequencing thread. */
    virtual bool is_current_thread() const = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_REACTOR_HPP
