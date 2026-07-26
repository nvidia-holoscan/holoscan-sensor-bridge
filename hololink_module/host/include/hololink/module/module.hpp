/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_BASE_HPP
#define HOLOLINK_MODULE_BASE_HPP

#include <memory>

#include "service_locator.h"

namespace hololink::module {

/* Abstract base for any service-locator entry point. Concrete
 * subclasses provide their own get_service strategy:
 *
 *   - LoadedModule wraps a (get_service, release_service) callback
 *     pair coming from a peer binary.
 *   - Publisher::SelfModule (private nested) looks services up
 *     directly in the local Publisher's registry, used by in-binary
 *     code that wants the same typed T::get_service(module, ...) path
 *     consumer code uses.
 *
 * There are many Module instances per process — one per peer
 * relationship (LoadedModule), plus one self-Module per binary
 * (Publisher::SelfModule). Consumers always hold a shared_ptr<Module>
 * regardless of which subclass is underneath. */
class Module : public std::enable_shared_from_this<Module> {
public:
    virtual ~Module() = default;

    /* Look up a service. On a registry hit the cached instance is
     * returned; on a miss the module attempts construction via a
     * type_id-keyed default constructor registered with the
     * Publisher (used by ConfigurableService<T>::get_service to
     * materialize impls lazily). Throws when no entry results
     * unless allow_null=true. The returned shared_ptr's deleter
     * keeps this Module alive (and through it, whatever resources
     * back the service) for as long as the caller holds it.
     * Higher-level callers go through Service<T>::get_service. */
    virtual std::shared_ptr<const void> get_service(
        const char* instance_id,
        const char* type_id,
        bool allow_null = false)
        = 0;
};

/* Concrete Module that wraps a (get_service, release_service)
 * callback pair from a peer binary. Optionally owns a "lifetime
 * keeper" (e.g. an RAII dlopen handle for the host-loaded-.so case)
 * — when the last shared_ptr<LoadedModule> drops, the keeper is
 * destroyed too. The keeper is opaque so this class has no dlfcn
 * dependency. */
class LoadedModule : public Module {
public:
    /* Wrap a peer's callback pair. Both callbacks must be non-NULL.
     * The optional keeper is held until this Module is destroyed and
     * is provided so the host's .so loader can attach a dlopen-handle
     * RAII without LoadedModule having to know about dlopen. */
    static std::shared_ptr<LoadedModule> create(
        hololink_module_get_service get_service,
        hololink_module_release_service release_service,
        std::shared_ptr<void> keeper = nullptr);

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

} // namespace hololink::module

#endif // HOLOLINK_MODULE_BASE_HPP
