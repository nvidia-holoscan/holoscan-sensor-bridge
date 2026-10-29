/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_BASE_HPP
#define HOLOLINK_MODULE_BASE_HPP

#include <memory>

#include "service_locator.h"

namespace hololink::module {

/**
 * @brief Module encapsulates a set of services, normally
 *      associated with a specific HSB implementation.
 *
 * The group of services for a specific Module are normally
 * implemented in a single shared object file, but an application
 * also provides a Module instance that encapsulates services
 * that those loaded modules will use (e.g. Logging).
 *
 * Applications hold a `std::shared_ptr<Module>` and pass it to the
 * type-safe `get_service` function of the interface they want (e.g.
 * `Service<T>::get_service`), which calls Module::get_service
 * and returns a correctly typed pointer.
 *
 * A service returned from a Module remains valid for as long as the
 * caller holds it, independent of how long the caller keeps the Module
 * itself.
 */
class Module : public std::enable_shared_from_this<Module> {
public:
    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    virtual ~Module() = default;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Look up a service by its untyped identifiers.
     *
     * This is the untyped lookup that the type-safe `get_service`
     * functions are built on; applications always call those
     * instead. If a service matching both identifiers is available, it
     * is returned; repeated calls with the same identifiers may return
     * the same instance. A service that has not been used before may be
     * created by the module as part of the lookup.
     *
     * @param instance_id Names the specific service instance; its format
     *      is defined by the service type.
     * @param type_id Names the type of the service interface,
     *      corresponding with the 'type_id' member of a Service class
     *      (e.g. `"mipi_dphy.v1"`).
     * @param allow_null If `false` (the default), a service that cannot
     *      be found causes a std::runtime_error to be thrown. If `true`,
     *      an empty pointer is returned instead.
     * @returns The service, as a type-erased pointer to the object
     *      registered under `type_id`, or nullptr when it is
     *      not found and `allow_null` is `true`.
     * @throws std::runtime_error if `instance_id` or `type_id` is null,
     *      or if the service is not found and `allow_null` is `false`.
     */
    virtual std::shared_ptr<const void> get_service(
        const char* instance_id,
        const char* type_id,
        bool allow_null = false)
        = 0;
};

/// @cond INCLUDE_INTERNAL_DOCUMENTATION
/**
 * @brief Implementation of Module where get_service delegates
 *      to the set of callbacks published by a loaded .so module.
 */
class LoadedModule : public Module {
public:
    /**
     * @brief Wrap a peer's callback pair in a new LoadedModule.
     *
     * @param get_service Peer callback used to look up services. Must be
     *        non-NULL.
     * @param release_service Peer callback invoked with each handle
     *        previously returned by @p get_service once it is no longer
     *        referenced. Must be non-NULL.
     * @param keeper Optional opaque object whose lifetime is tied to this
     *        module; it is destroyed after the last reference to the
     *        module is released. Typically an RAII dlopen handle supplied
     *        by the host's .so loader, so this class needs no dlfcn
     *        dependency. May be null.
     * @returns The new module.
     * @throws std::runtime_error if @p get_service or @p release_service
     *         is NULL.
     */
    static std::shared_ptr<LoadedModule> create(
        hololink_module_get_service get_service,
        hololink_module_release_service release_service,
        std::shared_ptr<void> keeper = nullptr);

    /**
     * @brief Look up a service by delegating to the peer's get_service
     * callback.
     *
     * The returned shared_ptr's deleter calls the peer's release_service
     * callback with the handle and releases a strong reference to this
     * module.
     *
     * @param instance_id Identifies the service instance. Must be
     *        non-NULL.
     * @param type_id Identifies the service type. Must be non-NULL.
     * @param allow_null If true, a lookup that finds no matching service
     *        returns an empty shared_ptr; if false, it throws.
     * @returns The service handle, or an empty shared_ptr when no service
     *          matches and @p allow_null is true.
     * @throws std::runtime_error if @p instance_id or @p type_id is NULL,
     *         or if no service matches and @p allow_null is false.
     */
    std::shared_ptr<const void> get_service(
        const char* instance_id,
        const char* type_id,
        bool allow_null = false) override;

private:
    LoadedModule(hololink_module_get_service get_service,
        hololink_module_release_service release_service,
        std::shared_ptr<void> keeper);

    hololink_module_get_service get_service_;
    hololink_module_release_service release_service_;
    std::shared_ptr<void> keeper_;
};
/** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

} // namespace hololink::module

#endif // HOLOLINK_MODULE_BASE_HPP
