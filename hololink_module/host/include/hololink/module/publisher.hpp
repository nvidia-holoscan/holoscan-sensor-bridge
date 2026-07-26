/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_PUBLISHER_HPP
#define HOLOLINK_MODULE_PUBLISHER_HPP

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "logging.hpp"
#include "module.hpp"
#include "reactor.hpp"
#include "service.hpp"
#include "service_locator.h"

namespace hololink::module {

/* Publisher owns the local-binary registry of services published from
 * this binary. It exports the static C-ABI thunks the peer uses to
 * call into us, register_service() (called via the ServicePublisher
 * helper), callbacks() (returned to the peer at init time), and
 * self_module() (a Module that loops back into the local registry,
 * used for in-binary lookups).
 *
 * Exactly one Publisher exists per process binary. Each module .so
 * has its own private RTLD_LOCAL'd Publisher; the host process has
 * one. The base constructor enforces this by stamping `current_` and
 * throwing if a second instance is constructed in the same binary. */
class Publisher : public std::enable_shared_from_this<Publisher> {
public:
    Publisher(const Publisher&) = delete;
    Publisher& operator=(const Publisher&) = delete;
    virtual ~Publisher();

    /* Module whose get_service walks the local registry directly. Use
     * it when in-binary code needs typed T::get_service(module, ...)
     * lookups against locally-registered services. The returned Module
     * keeps this Publisher alive for as long as it (or any service
     * handle obtained through it) is held; consumers may safely retain
     * it past the Publisher creator's lifetime. */
    std::shared_ptr<Module> self_module();

    /* Module fronting the host's service registry via the
     * hololink_module_init_t callbacks (get_service /
     * release_service). Used by module-internal code that needs to
     * reach host-published services (LoggingInterfaceV1, ReactorV1,
     * etc.). Non-null after a successful setup() on a module-side
     * Publisher; null on the host-side `HostPublisher` (which doesn't
     * bridge a binary boundary) and on any Publisher before setup()
     * has populated it. */
    std::shared_ptr<Module> host_module() const { return host_module_; }

    /* The host's logger, fetched during setup(). Same instance the
     * HSB_LOG_* macros dispatch through — setup() wired the
     * per-binary HSB_LOG cache. Null on the host-side Publisher and
     * before setup() has populated it. */
    std::shared_ptr<LoggingInterfaceV1> logger() const { return logger_; }

    /* The host's reactor, fetched during setup(). Null on the
     * host-side Publisher and before setup() has populated it. */
    std::shared_ptr<ReactorV1> reactor() const { return reactor_; }

    /* C-ABI callback set the peer uses to call into us. */
    hololink_module_services_t callbacks() const;

    /* Invalidate a single cached service so the next lookup reconstructs it;
     * a service passes `this` to invalidate itself. The Publisher keeps no
     * notion of how services relate — callers drive any cascade. Static:
     * Publisher is accessed via the per-binary `current_`. */
    static void invalidate(const void* service);

    /* Module-side bootstrap + "module init is done" entry point.
     * Validates `init` (non-null, matching api_version, non-null
     * get_service/release_service callbacks) and returns a result
     * struct with status = HOLOLINK_MODULE_INVALID_PARAMETER on
     * validation failure — does not throw. On a valid init, builds
     * the host Module from the init callbacks, fetches the host's
     * LoggingInterfaceV1 (wiring this binary's HSB_LOG cache), and
     * fetches the host's ReactorV1. All three are then reachable
     * via host_module() / logger() / reactor(). Default body
     * returns callbacks() once the bootstrap is done; subclasses
     * override to perform any eager-publish work (e.g.
     * HsbLitePublisher publishes the FrameMetadataInterfaceV1 +
     * EnumerationInterfaceV1 singletons) before returning
     * callbacks(). Called by every supplement's hololink_module_init. */
    virtual hololink_module_services_t setup(
        const hololink_module_init_t* init);

    /* Override hook for binary-specific lazy construction. Invoked by
     * lookup() on a registry miss. The override default-constructs the
     * matching impl (cheap; no resource acquisition), publishes it
     * under instance_id via ServicePublisher<T>, and returns true.
     * Returning false signals "no constructor for this type_id" —
     * the lookup then surfaces a miss to the caller. Binaries with
     * no lazy services implement this as `return false;`. */
    virtual bool construct_service(
        const std::string& instance_id,
        const std::string& type_id)
        = 0;

    /* Pure class-query: returns true iff `type_id` appears in T's
     * reported set (i.e. T::for_each_type_id calls back at least once
     * with a matching string). Static because no Publisher state is
     * involved; callable either via instance or fully-qualified. The
     * iteration halts on the first match. */
    template <typename T>
    static bool has_type_id(const std::string& type_id)
    {
        bool found = false;
        T::for_each_type_id([&](const char* tid) {
            if (type_id == tid) {
                found = true;
                return false; // halt
            }
            return true;
        });
        return found;
    }

protected:
    /* The Publisher's bootstrap state (host Module + logger +
     * reactor) is populated by `setup(init)`; the constructor just
     * registers the per-binary `current_` slot. One Publisher per
     * binary — module-side supplements construct theirs in
     * hololink_module_init; the host-side `HostPublisher` (in
     * `host/src/adapter.cpp`) constructs in `Adapter::Adapter()`
     * and never calls setup (it has no init to bootstrap from). */
    Publisher();

private:
    class SelfModule;
    friend class SelfModule;

    struct Key {
        std::string instance_id;
        std::string type_id;
        bool operator==(const Key& other) const noexcept
        {
            return instance_id == other.instance_id && type_id == other.type_id;
        }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const noexcept
        {
            std::hash<std::string> h;
            return h(k.instance_id) ^ (h(k.type_id) << 1);
        }
    };

    /* Called by ServicePublisher<T>::publish. */
    void register_service(const std::string& instance_id,
        const std::string& type_id,
        std::shared_ptr<void> service);

    static hololink_module_service_t get_service_thunk(
        const char* instance_id, const char* type_id);
    static void release_service_thunk(
        hololink_module_service_t instance);

    /* Look up a service in the local registry. On a miss invokes
     * the construct_service() virtual; the override publishes the
     * new impl into the registry, and this function re-reads the
     * entry and returns it. Returns an empty pointer when neither
     * path produces a service. Used by both the static thunk (for
     * cross-binary peer calls) and SelfModule (for in-binary
     * lookups). */
    std::shared_ptr<void> lookup(const std::string& instance_id,
        const std::string& type_id);

    /* One Publisher per binary. Each module .so absorbs its own
     * private copy of this translation unit via
     * hololink::module_runtime + RTLD_LOCAL, so the static is unique
     * to that binary and the C-ABI thunks export from each module
     * route to that module's Publisher. */
    static Publisher* current_;

    // Recursive because lookup() holds it across construct_service()
    // (so two concurrent lookups for the same key can't both invoke
    // construction and publish duplicate impls), and construct_service
    // overrides publish via ServicePublisher::publish, which calls
    // register_service, which re-acquires this mutex on the same
    // thread.
    std::recursive_mutex registry_mutex_;
    std::unordered_map<Key, std::shared_ptr<void>, KeyHash> registry_;

    // Strong references backing the raw pointers handed to the host via
    // get_service_thunk (the C-ABI carries only the pointer, so host
    // handles are owning by virtue of these), keyed by that pointer with
    // a live-handle count. An instance is kept alive by registry_ (cache)
    // or by any outstanding handle here. Guarded by its own mutex, never
    // nested with registry_mutex_.
    struct Outstanding {
        std::shared_ptr<void> service;
        std::size_t count;
    };
    std::mutex outstanding_mutex_;
    std::unordered_map<const void*, Outstanding> outstanding_;

    // Strong: the Publisher keeps the SelfModule alive for its
    // whole lifetime so that Service<T>::module_ (which is
    // a weak_ptr) never expires while any service published by this
    // Publisher is still reachable. The reverse edge — SelfModule's
    // pointer back to the Publisher — is weak to break the cycle.
    std::mutex self_module_mutex_;
    std::shared_ptr<SelfModule> self_module_;

    // Bootstrap state populated by setup(); empty before setup
    // runs and on the host-side Publisher (which has no init).
    std::shared_ptr<Module> host_module_;
    std::shared_ptr<LoggingInterfaceV1> logger_;
    std::shared_ptr<ReactorV1> reactor_;

    template <typename T>
    friend class ServicePublisher;
};

/* Helper used to publish services. Stamps the service's shared
 * ServiceBase::module_ with the Publisher's self_module() so child
 * accessors (e.g. HololinkInterface::get_i2c) resolve children in
 * the same binary the parent was published in.
 *
 * The single-arg publish walks T::for_each_type_id and registers the
 * same impl under every key the chain reports. For V1 interfaces
 * whose for_each_type_id emits just the interface's type_id, that's
 * a single register_service call; for concrete impls whose
 * for_each_type_id chains through the interface to the impl's own
 * type_id, the same call registers under each.
 *
 * The two-arg overload registers under one explicit type_id (used by
 * the iteration body, and by callers who need to publish under just
 * one key without consulting the chain). */
template <typename T>
class ServicePublisher {
public:
    explicit ServicePublisher(std::shared_ptr<Publisher> publisher)
        : publisher_(std::move(publisher))
    {
    }

    void publish(const std::string& instance_id, std::shared_ptr<T> impl)
    {
        T::for_each_type_id([&](const char* tid) {
            publish(instance_id, tid, impl);
            return true;
        });
    }

    void publish(const std::string& instance_id,
        const char* type_id,
        std::shared_ptr<T> impl)
    {
        impl->ServiceBase::module_ = publisher_->self_module();
        publisher_->register_service(instance_id, type_id, impl);
    }

private:
    std::shared_ptr<Publisher> publisher_;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_PUBLISHER_HPP
