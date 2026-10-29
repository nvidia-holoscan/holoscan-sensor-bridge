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
#include "release.hpp"

namespace hololink::module {

template <typename T>
class ServicePublisher;

/// @cond INCLUDE_INTERNAL_DOCUMENTATION
/* Compile-time checks that ensure service interfaces include a
 * `release_version` member.  Valid values for `release_version` are
 *
 *  - "x.y.z" (e.g. "2.7.0") this interface is frozen and must not change
 *  - UNRELEASED this interface has not been included in a release
 *      yet and is allowed to change.  None of these are allowed into
 *      a released tree.
 *  - TEST_ONLY is used for services that are only used in test fixtures;
 *      these are allowed to change and are not allowed to be used
 *      in actual application or example code.
 */
template <typename U, typename = void>
struct has_release_version : std::false_type {
};
template <typename U>
struct has_release_version<U, std::void_t<decltype(U::release_version)>>
    : std::true_type {
};
template <typename T>
constexpr void check_release_version()
{
    static_assert(has_release_version<T>::value,
        "Service class has no 'static constexpr const char* release_version' to "
        "declare or inherit. An interface declares its own next to type_id: the "
        "version of the release that froze it, UNRELEASED (release.hpp) until one "
        "does, or TEST_ONLY for a fixture application code must not use. An "
        "implementation inherits its interface's.");
}
/** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

/**
 * @brief Holds the handle to the Module a service was published with.
 *
 * A service reaches the other services it cooperates with (for example
 * a data channel finding its device) through this Module, so that they
 * all come from the same module as the service itself.
 *
 * Services use virtual inheritance of ServiceBase to ensure they have
 * exactly one pointer to the owner Module, even when a service
 * implements several interfaces.
 *
 * Applications do not use this class directly; it is a base of every
 * interface derived from Service.
 */
class ServiceBase {
public:
    /**
     * @brief The Module that this service was published from.
     *
     * The service holds no strong reference to its Module, so this does
     * not extend the Module's lifetime.
     *
     * @returns The owning Module, or an empty pointer if the service has
     *      not been published or its Module no longer exists.
     */
    std::shared_ptr<Module> module() const { return module_.lock(); }

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
protected:
    ServiceBase() = default;
    ~ServiceBase() = default;
    ServiceBase(const ServiceBase&) = default;
    ServiceBase& operator=(const ServiceBase&) = default;

private:
    std::weak_ptr<Module> module_;

    template <typename T>
    friend class ServicePublisher;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION
};

/** Service provides a `get_service` method which provides a type-safe
 * way to fetch a <Derived> service implementation from a module.  Note that
 * the same service instance may implement any number of Service
 * interfaces, so an interface is provided allowing a visitor
 * to see the list of service names that these objects support. */
template <typename Derived>
class Service : public virtual ServiceBase {
public:
    /**
     * @brief Type-safe way to fetch a std::shared_ptr to an
     *      instance of <Derived> from a Module.
     *
     * Note that Module usually caches instances, it's likely
     * that calls to get_service with the same locator_id will
     * return pointers to the same implementation object.
     *
     * @param module is the shared object that owns the instance
     *      we wish to fetch
     * @param instance_id Names the specific object, usually
     *      expressed as a set of "name=value;..." pairs,
     *      as defined by the implementations themselves.  This
     *      name is normally fetched by calling
     *      `Derived::locator_id(enumeration_metadata)`.
     * @param allow_null The default value of `false` causes
     *      this routine to throw a std::runtime_error if the
     *      service cannot be found within the module; this would
     *      be expected to terminate the calling program.  If
     *      the application wants to handle the error, pass in
     *      `true` here and get_service will return `nullptr`
     *      when this module doesn't provide the requested service.
     * @returns A shared pointer to the service instance.
     */
    static std::shared_ptr<Derived> get_service(
        std::shared_ptr<Module> module,
        const char* instance_id,
        bool allow_null = false)
    {
        check_release_version<Derived>();
        if (!module && allow_null) {
            return {};
        }
        if (!module) {
            throw std::runtime_error("Invalid module pointer passed to get_service.");
        }
        // Note that module->get_service raises std::runtime_error per the
        // rules above.
        std::shared_ptr<const void> service = module->get_service(instance_id, Derived::type_id, allow_null);
        if (!service) {
            return {};
        }
        // The publisher stored a shared_ptr<Derived>; aliasing the
        // typed pointer over its control block keeps the same lifetime.
        Derived* typed = const_cast<Derived*>(static_cast<const Derived*>(service.get()));
        return std::shared_ptr<Derived>(std::move(service), typed);
    }

    /**
     * @brief Iterate over every type_id supported by this instance.
     *
     * The callback signature is `bool(const char*)` — returning false
     * halts the walk; true continues.  Iteration is depth-first,
     * so you'll see the most specific implementation types first.
     */
    template <typename Callback>
    static bool for_each_type_id(Callback&& cb)
    {
        check_release_version<Derived>();
        if constexpr (has_service_alias<Derived>::value) {
            if (!Derived::ServiceAlias::for_each_type_id(cb)) {
                return false;
            }
        }
        return cb(Derived::type_id);
    }

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
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
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION
};

/**
 * @brief A service whose instance is constructed from EnumerationMetadata.
 *
 * Services are instances addressed by instance_id; ConfigurableService
 * constructs the instance_id from fields in EnumerationData, via Derived::locator_id,
 * (frequently of the form "serial=X;instance=Y" or another string uniquely
 * identifying the implementation object).  When the Module doesn't
 * yet have an instance of that service, EnumerationMetadata will be
 * used to initialize it.
 *
 * @tparam Derived The service class being resolved.
 */
template <typename Derived>
class ConfigurableService : public Service<Derived> {
public:
    using Service<Derived>::get_service;

    /**
     * @brief Fetch a service, configured by EnumerationMetadata.
     *
     * Module usually caches service objects; so when a specific service
     * instance isn't found, it will be constructed with the current values
     * in EnumerationMetadata.  Later calls to get_service are likely to
     * return that cached instance, ignoring changes to any enumeration
     * metadata that aren't specifically related to service addressing.
     *
     * @param metadata Addressing and configuration for the service.
     * @param allow_null Use `true` to return nullptr when the service cannot
     *      be resolved; the default `false` value raises a std::runtime_error.
     * @return A shared pointer to the requested service (<Derived>).
     * @throws std::runtime_error when allow_null==false and the service
     *      lookup (or construction) fails.
     */
    static std::shared_ptr<Derived> get_service(
        const EnumerationMetadata& metadata,
        bool allow_null = false);

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    /**
     * @brief Bring the instance's configuration up to date with `metadata`.
     *
     * Runs configure() once per instance: the first caller runs it, callers
     * arriving while it runs wait for it, and later callers return
     * immediately. If configure() throws, the next call runs it again.
     */
    virtual void ensure_configured(const EnumerationMetadata& metadata)
    {
        std::call_once(configure_once_, [&]() {
            this->configure(metadata);
        });
    }
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Use `metadata` contents to initialize this instance.
     *
     * This routine is guaranteed to be called the first time a
     * call to get_service(metadata) is called on a specific service
     * instance.
     */
    virtual void configure(const EnumerationMetadata& /*metadata*/) { }

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
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
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION
};

/// @cond INCLUDE_INTERNAL_DOCUMENTATION
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

/* Implements ConfigurableService::get_service, defined
 * in the template above. */
template <typename Derived>
std::shared_ptr<Derived> ConfigurableService<Derived>::get_service(
    const EnumerationMetadata& metadata, bool allow_null)
{
    check_release_version<Derived>();
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
/** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

} // namespace hololink::module

#endif // HOLOLINK_MODULE_SERVICE_HPP
