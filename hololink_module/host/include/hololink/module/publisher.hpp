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

/**
 * @brief Provide support for loadable modules that publish
 *      a specific set of services.
 *
 * This support includes
 *  - Lazy service instantiation with caching: The first time
 *      a request for a specific resource is observed results
 *      in a call to construct_service().  The return value is
 *      cached and returned in subsequent calls to lookup().
 *  - Caches of common services provided by the application layer,
 *      including Logging and Reactor, and the Module presented
 *      by the application.
 *
 * Exactly one Publisher may exist in an individual binary (shared
 * object or application).  Each module shared object has its own,
 * separate from the host's.
 *
 * Lookups are thread-safe. Lookups of the same service made
 * concurrently do not call construct_service() more than once.
 */
class Publisher : public std::enable_shared_from_this<Publisher> {
public:
    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    Publisher(const Publisher&) = delete;
    Publisher& operator=(const Publisher&) = delete;
    virtual ~Publisher();
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief A Module through which this Publisher's own services can be
     * looked up.
     *
     * Use it when code in the same binary needs the typed
     * `T::get_service(module, ...)` lookup for a service published here.
     * The returned Module keeps this Publisher alive for as long as it,
     * or any service obtained through it, is held, so it may be retained
     * after the code that created the Publisher has finished.
     *
     * @returns The same Module on every call.
     */
    std::shared_ptr<Module> self_module();

    /**
     * @brief A Module through which a module can look up services
     * published by the host (for example LoggingInterfaceV1 and
     * ReactorV1).
     *
     * @returns The host's Module once setup() has succeeded. Null before
     *      then, and always null on the host's own Publisher.
     */
    std::shared_ptr<Module> host_module() const { return host_module_; }

    /**
     * @brief The host's logger.
     *
     * This is the logger that the `HSB_LOG_*` macros use in this binary.
     *
     * @returns The logger once setup() has succeeded. Null before then,
     *      and always null on the host's own Publisher.
     */
    std::shared_ptr<LoggingInterfaceV1> logger() const { return logger_; }

    /**
     * @brief The host's reactor.
     *
     * @returns The reactor once setup() has succeeded. Null before then,
     *      and always null on the host's own Publisher.
     */
    std::shared_ptr<ReactorV1> reactor() const { return reactor_; }

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    /* C-ABI callback set the peer uses to call into us. */
    hololink_module_services_t callbacks() const;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Discard a published service so that the next lookup
     * creates a fresh one.
     *
     * Clears the instance cache for this service; a later call to lookup for
     * this object will return a newly constructed instance.  Note that this
     * decrements the shared_ptr reference count, so the actual instance is
     * destructed only after the last reference is dropped.  This method does
     * nothing if `service` is null or was not published.
     *
     * A service can invalidate itself by passing `this`.
     *
     * @param service Address of the service object, as published.
     */
    static void invalidate(const void* service);

    /**
     * @brief Connect a module to the host. Called from the module's
     * `hololink_module_init` entry point.
     *
     * Checks `init` and, if it is acceptable, makes the host's Module,
     * logger and reactor available through host_module(), logger() and
     * reactor(), and connects this binary's `HSB_LOG_*` macros to the
     * host's logger. Subclasses may override this to publish services
     * that must exist before any lookup; an override should call this
     * implementation first and return its result unchanged if the
     * status is not HOLOLINK_MODULE_OK.
     *
     * @param init Initialization data passed by the host to the module.
     * @returns The services structure to return from
     *      `hololink_module_init`. Its status is
     *      HOLOLINK_MODULE_INVALID_PARAMETER, without throwing, if `init`
     *      is null, was built for a different API version, or lacks its
     *      lookup callbacks.
     * @throws std::runtime_error if the host does not provide a logger or
     *      reactor.
     */
    virtual hololink_module_services_t setup(
        const hololink_module_init_t* init);

    /**
     * @brief Create a service on demand.
     *
     * Called when a lookup finds no cached service for the given
     * identifiers. An implementation that can supply the service
     * constructs it, publishes it under `instance_id` with
     * ServicePublisher, and returns `true`. Construction should be
     * cheap and must not acquire hardware or other resources; that
     * belongs in the service's later configuration. An implementation
     * that has no such service returns `false`, and the lookup then
     * reports the service as not found.
     *
     * Calls are serialized with other lookups, and publishing other
     * services within this call is permitted.
     *
     * @param instance_id Instance id requested by the lookup.
     * @param type_id Type id requested by the lookup.
     * @returns `true` if a service was published for these identifiers.
     */
    virtual bool construct_service(
        const std::string& instance_id,
        const std::string& type_id)
        = 0;

    /**
     * @brief Report whether a service class is published under a type
     * id.
     *
     * A service class may support multiple type ids (e.g. it supports
     * several versions of an interface or different, maybe unrelated interfaces).
     * This method, for a given service class T, returns `true` if the
     * requested type_id is in the list of those supported by T.
     *
     * @tparam T Service class to test.
     * @param type_id Type id requested by a lookup.
     * @returns `true` if `type_id` is one of the type ids of `T`.
     */
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
     * binary — modules construct a subclass of this in
     * hololink_module_init; the host-side `HostPublisher` (in
     * `host/src/adapter.cpp`) constructs in `Adapter::Adapter()`
     * and never calls setup (it has no init to bootstrap from). */

    /**
     * @brief Construct the binary's Publisher.
     *
     * Intended to be called from a derived class: a module's Publisher
     * is created in its `hololink_module_init`, the host's when it
     * starts up.
     *
     * @throws std::runtime_error if a Publisher already exists in this
     *      binary.
     */
    Publisher();

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
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
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION
};

/**
 * @brief Publishes service implementations into a Publisher, making them
 * available to lookups.
 *
 * Use it from a Publisher's own code: either to publish services eagerly
 * (for example in Publisher::setup()) or from
 * Publisher::construct_service() to publish the service that was just
 * asked for.
 *
 * Publishing does two things. It records the implementation in the
 * Publisher, which keeps it alive and returns it from later lookups
 * until Publisher::invalidate() discards it. And it makes the
 * Publisher's self_module() the service's owning Module (see
 * ServiceBase::module()), so that the service's accessors for related
 * services, such as a device's `get_i2c`, look them up in the same
 * binary the service was published from.
 *
 * @tparam T The service class to publish. It must declare or inherit a
 *      `release_version`.
 */
template <typename T>
class ServicePublisher {
public:
    /**
     * @param publisher The Publisher to publish into.
     */
    explicit ServicePublisher(std::shared_ptr<Publisher> publisher)
        : publisher_(std::move(publisher))
    {
    }

    /**
     * @brief Publish a service under every type id that `T` provides.
     *
     * A service that can be used as several interfaces is made available
     * under the type id of each, so that a lookup for any of them
     * returns this object.
     *
     * @param instance_id Instance id to publish under; lookups must use
     *      the same value.
     * @param impl The service. Must not be null.
     * @throws std::runtime_error if any of the (instance id, type id)
     *      pairs is already published. Pairs published before the
     *      collision was found remain published.
     */
    void publish(const std::string& instance_id, std::shared_ptr<T> impl)
    {
        T::for_each_type_id([&](const char* tid) {
            publish(instance_id, tid, impl);
            return true;
        });
    }

    /**
     * @brief Publish a service under one type id.
     *
     * Use this to publish under a single type id, rather than all of the
     * type ids that `T` provides.
     *
     * @param instance_id Instance id to publish under; lookups must use
     *      the same value.
     * @param type_id Type id to publish under.
     * @param impl The service. Must not be null.
     * @throws std::runtime_error if this (instance id, type id) pair is
     *      already published.
     */
    void publish(const std::string& instance_id,
        const char* type_id,
        std::shared_ptr<T> impl)
    {
        check_release_version<T>();
        impl->ServiceBase::module_ = publisher_->self_module();
        publisher_->register_service(instance_id, type_id, impl);
    }

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
private:
    std::shared_ptr<Publisher> publisher_;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_PUBLISHER_HPP
