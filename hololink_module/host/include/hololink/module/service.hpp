/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_SERVICE_HPP
#define HOLOLINK_MODULE_SERVICE_HPP

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "enumeration_metadata.hpp"
#include "module.hpp"

namespace hololink::module {

template <typename T>
class ServicePublisher;

/* Non-template base shared (via virtual inheritance) by every Service<T>
 * a class derives from. Collapses what would otherwise be one
 * Service<T>::module_ per Service<T> instantiation into a single weak_ptr
 * per object — so an impl that derives from both Service<Interface> and
 * Service<Impl> (the type_id-keyed pattern that lets the impl be fetched
 * cast-free) still has one module(), one module_, one stamping site. */
class ServiceBase {
public:
    std::shared_ptr<Module> module() const { return module_.lock(); }

protected:
    ServiceBase() = default;
    ~ServiceBase() = default;
    ServiceBase(const ServiceBase&) = default;
    ServiceBase& operator=(const ServiceBase&) = default;

private:
    std::weak_ptr<Module> module_;

    template <typename T>
    friend class ServicePublisher;
};

/* CRTP base for every interface or impl reachable through the locator.
 * Derived classes supply `static constexpr const char* type_id` and
 * inherit the static get_service template that aliases the typed pointer.
 *
 * Inherits ServiceBase virtually so a class that picks up multiple
 * Service<T> bases (interface T plus impl T') still has one ServiceBase
 * subobject — one `module_`, one `module()` accessor, no diamond. */
template <typename Derived>
class Service : public virtual ServiceBase {
public:
    static std::shared_ptr<Derived> get_service(
        std::shared_ptr<Module> module,
        const char* instance_id,
        bool allow_null = false)
    {
        if (!module && allow_null) {
            return {};
        }
        if (!module) {
            throw std::runtime_error("Invalid module pointer passed to get_service.");
        }
        std::shared_ptr<const void> service = module->get_service(instance_id, Derived::type_id, allow_null);
        if (!service) {
            return {};
        }
        // The publisher stored a shared_ptr<Derived>; aliasing the
        // typed pointer over its control block keeps the same lifetime.
        Derived* typed = const_cast<Derived*>(static_cast<const Derived*>(service.get()));
        return std::shared_ptr<Derived>(std::move(service), typed);
    }

    /* Iterate over every type_id this class is published under.
     *
     * The callback signature is `bool(const char*)` — returning false
     * halts the walk; true continues. Used by
     * ServicePublisher<T>::publish to walk the chain when registering
     * an impl, and by Publisher::has_type_id to recognize which
     * type_ids dispatch here.
     *
     * Chain walking is driven by an optional `Derived::ServiceAlias`
     * typedef: an impl class declares
     *   using ServiceAlias = <ParentClass>;
     * and Service<Derived>::for_each_type_id first walks
     * ParentClass::for_each_type_id, then emits Derived::type_id.
     * Classes that don't declare ServiceAlias (V1 interfaces at the
     * root of a chain) emit just their own type_id and terminate. */
    template <typename Callback>
    static bool for_each_type_id(Callback&& cb)
    {
        if constexpr (has_service_alias<Derived>::value) {
            if (!Derived::ServiceAlias::for_each_type_id(cb)) {
                return false;
            }
        }
        return cb(Derived::type_id);
    }

protected:
    template <typename U, typename = void>
    struct has_service_alias : std::false_type {
    };
    template <typename U>
    struct has_service_alias<U, std::void_t<typename U::ServiceAlias>>
        : std::true_type {
    };

    Service() = default;
    ~Service() = default;
    Service(const Service&) = default;
    Service& operator=(const Service&) = default;
};

/* Service whose instance is constructed lazily from EnumerationMetadata.
 *
 * Each Derived supplies:
 *   - `static std::string locator_id(const EnumerationMetadata&)`
 *      — derives the instance_id used to key the cache.
 *   - `virtual void configure(const EnumerationMetadata&)`
 *      — one-shot resource materialization. Runs exactly once per
 *        instance: the framework's metadata-form get_service wraps
 *        it in std::call_once on the instance's configure_once_
 *        latch, so impls don't need to guard against repeat calls.
 *        The metadata of the *first* get_service(metadata) caller
 *        wins; later callers' metadata is ignored (consistent with
 *        the cache-dedup semantics — the same instance can't be
 *        configured twice with different inputs).
 *
 * The metadata-form `get_service` resolves the module via the host-side
 * Adapter, asks the module-side Publisher to look up or construct the
 * impl (cache lives in the Publisher), then calls `configure(metadata)`
 * exactly once before returning. The instance_id-form `get_service`
 * inherited from Service<Derived> is a pure cache lookup — it works
 * only after a metadata-form call has populated the cache. */
template <typename Derived>
class ConfigurableService : public Service<Derived> {
public:
    using Service<Derived>::get_service;

    /* Defined out-of-line in the host-side `service.cpp` so this header
     * does not pull adapter.hpp (and the Adapter singleton) into module
     * binaries that include this file but never call into Adapter. */
    static std::shared_ptr<Derived> get_service(
        const EnumerationMetadata& metadata,
        bool allow_null = false);

    /* Ensure the service's configuration reflects `metadata`. The default
     * implementation runs configure(metadata) exactly once via
     * std::call_once on the per-instance configure_once_ latch — first
     * caller runs configure, concurrent callers block until done,
     * subsequent callers return immediately; if configure throws, the
     * latch resets so a follow-up call retries. This default assumes the
     * metadata for a given service key is stable across resolutions.
     *
     * Virtual so services whose configuration genuinely varies per
     * resolution can broaden the contract to "reconfigure when the
     * metadata differs from what was last applied." The motivating case
     * is data channels: a per-(serial,data_channel) channel is a
     * process-lifetime cached singleton, but its host / data-plane
     * addressing is derived from the enumeration metadata the application
     * resolved it with — so a value an earlier resolution cached (e.g. a
     * different test's channel for the same key) must not override what
     * the application asks for now. Such impls override this to apply the
     * supplied metadata each resolution.
     *
     * Two routes call this:
     *   1. The framework's metadata-form get_service (host-side).
     *   2. Sibling impls inside a module .so that fetch a cached anchor
     *      via the (module, instance_id) form and ensure_configured it. */
    virtual void ensure_configured(const EnumerationMetadata& metadata)
    {
        std::call_once(configure_once_, [&]() {
            this->configure(metadata);
        });
    }

    /* One-shot materialization of impl-side resources from metadata.
     * Default no-op — impl classes that have nothing to materialize
     * (a singleton with all state in its default-constructed members)
     * keep the default. By default the framework runs this at most once
     * per instance via ensure_configured; impls don't need their own
     * std::call_once / atomic-bool guard. Don't call this method directly
     * from outside the framework — use ensure_configured. */
    virtual void configure(const EnumerationMetadata& /*metadata*/) { }

protected:
    ConfigurableService() = default;
    ~ConfigurableService() = default;
    ConfigurableService(const ConfigurableService&) = delete;
    ConfigurableService& operator=(const ConfigurableService&) = delete;

private:
    /* Latch the framework's metadata-form get_service flips on the
     * first lookup so configure(metadata) runs exactly once per
     * instance. Not copyable / movable — std::once_flag isn't either,
     * which is why the base's copy operations are deleted. */
    std::once_flag configure_once_;
};

/* Bridge between the templated ConfigurableService<Derived>::get_service
 * and the Adapter singleton. Resolves the module from metadata, asks
 * the Module to get-or-construct the (instance_id, type_id) entry, and
 * returns the typed-erased service pointer. The caller (the template
 * instantiation) does the static_cast back to Derived and the
 * configure call. */
std::shared_ptr<const void> get_or_construct_service_via_metadata(
    const EnumerationMetadata& metadata,
    const std::string& instance_id,
    const char* type_id,
    bool allow_null);

template <typename Derived>
std::shared_ptr<Derived> ConfigurableService<Derived>::get_service(
    const EnumerationMetadata& metadata, bool allow_null)
{
    const std::string instance_id = Derived::locator_id(metadata);
    std::shared_ptr<const void> service
        = get_or_construct_service_via_metadata(
            metadata, instance_id, Derived::type_id, allow_null);
    if (!service) {
        return {};
    }
    Derived* typed = const_cast<Derived*>(static_cast<const Derived*>(service.get()));
    auto typed_ptr = std::shared_ptr<Derived>(std::move(service), typed);
    typed_ptr->ensure_configured(metadata);
    return typed_ptr;
}

} // namespace hololink::module

#endif // HOLOLINK_MODULE_SERVICE_HPP
