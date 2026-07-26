/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hololink/module/abi_check.h"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/logging.hpp"
#include "hololink/module/module.hpp"
#include "hololink/module/publisher.hpp"
#include "hololink/module/reactor.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace hololink::module {

// ---------------------------------------------------------------------------
// LoadedModule
// ---------------------------------------------------------------------------

LoadedModule::LoadedModule(hololink_module_get_service get_service,
    hololink_module_release_service release_service,
    std::shared_ptr<void> keeper)
    : get_service_(get_service)
    , release_service_(release_service)
    , keeper_(std::move(keeper))
{
}

std::shared_ptr<LoadedModule> LoadedModule::create(
    hololink_module_get_service get_service,
    hololink_module_release_service release_service,
    std::shared_ptr<void> keeper)
{
    if (!get_service || !release_service) {
        throw std::runtime_error(
            "While constructing a LoadedModule: get_service and release_service must both be non-NULL");
    }
    return std::shared_ptr<LoadedModule>(
        new LoadedModule(get_service, release_service, std::move(keeper)));
}

std::shared_ptr<const void> LoadedModule::get_service(
    const char* instance_id, const char* type_id, bool allow_null)
{
    if (!instance_id || !type_id) {
        throw std::runtime_error(
            "While looking up a service: instance_id and type_id must both be non-NULL");
    }

    hololink_module_service_t handle = get_service_(instance_id, type_id);

    if (!handle) {
        if (allow_null) {
            return {};
        }
        throw std::runtime_error(std::string(
                                     "While looking up service '")
            + type_id + "' (instance_id='" + instance_id
            + "'): no matching service is registered");
    }

    // The deleter keeps a strong reference to this Module so the
    // backing .so cannot be dropped while a service handle from it
    // is still alive.
    auto self = shared_from_this();
    auto release = release_service_;
    return std::shared_ptr<const void>(handle,
        [self, release](const void* p) { release(p); });
}

// ---------------------------------------------------------------------------
// Publisher
// ---------------------------------------------------------------------------

class Publisher::SelfModule : public Module {
public:
    explicit SelfModule(std::weak_ptr<Publisher> publisher)
        : publisher_(std::move(publisher))
    {
    }

    std::shared_ptr<const void> get_service(
        const char* instance_id, const char* type_id, bool allow_null) override
    {
        if (!instance_id || !type_id) {
            throw std::runtime_error(
                "While looking up a service: instance_id and type_id must both be non-NULL");
        }

        auto publisher = publisher_.lock();
        if (!publisher) {
            if (allow_null) {
                return {};
            }
            throw std::runtime_error(std::string(
                                         "While looking up service '")
                + type_id + "' (instance_id='" + instance_id
                + "'): the Publisher that owns this SelfModule has been destroyed");
        }

        std::shared_ptr<void> stored = publisher->lookup(instance_id, type_id);
        if (!stored) {
            if (allow_null) {
                return {};
            }
            throw std::runtime_error(std::string(
                                         "While looking up service '")
                + type_id + "' (instance_id='" + instance_id
                + "'): no matching service is registered");
        }
        return std::shared_ptr<const void>(std::move(stored), stored.get());
    }

private:
    // Weak: the Publisher strongly owns this SelfModule (see
    // Publisher::self_module_), so a strong back-pointer here would
    // cycle. Locking on each get_service call is correct — the
    // Publisher always outlives every legitimate lookup.
    std::weak_ptr<Publisher> publisher_;
};

Publisher* Publisher::current_ = nullptr;

Publisher::Publisher()
{
    if (current_) {
        throw std::runtime_error(
            "While constructing a Publisher: a Publisher already exists in this binary "
            "(only one Publisher per process binary is supported)");
    }
    current_ = this;
}

hololink_module_services_t Publisher::setup(
    const hololink_module_init_t* init)
{
    hololink_module_services_t result {};
    if (!init || init->api_version != HOLOLINK_MODULE_API_VERSION
        || !init->get_service || !init->release_service) {
        result.status = HOLOLINK_MODULE_INVALID_PARAMETER;
        return result;
    }
    host_module_ = LoadedModule::create(
        init->get_service, init->release_service);
    logger_ = LoggingInterfaceV1::get_service(host_module_);
    set_hsb_logger_cache(logger_.get());
    reactor_ = ReactorV1::get_service(host_module_);
    return callbacks();
}

Publisher::~Publisher()
{
    if (current_ == this) {
        current_ = nullptr;
    }
}

void Publisher::register_service(const std::string& instance_id,
    const std::string& type_id,
    std::shared_ptr<void> service)
{
    std::lock_guard<std::recursive_mutex> guard(registry_mutex_);
    auto [it, inserted] = registry_.try_emplace(
        Key { instance_id, type_id }, std::move(service));
    if (!inserted) {
        // The same (instance_id, type_id) pair was already published in
        // this binary. The common cause is a derived class that inherits
        // its V1 interface's `type_id` instead of declaring its own —
        // ServicePublisher<DerivedImpl>::publish then reads
        // `DerivedImpl::type_id` and silently uses the V1's value,
        // colliding with the V1 publisher's prior registration.
        throw std::runtime_error(
            "While publishing a service: an entry with instance_id='"
            + instance_id + "' and type_id='" + type_id
            + "' is already published in this binary "
              "(check that each impl class declares its own type_id "
              "rather than inheriting one from its V1 interface)");
    }
}

hololink_module_services_t Publisher::callbacks() const
{
    return hololink_module_services_t {
        HOLOLINK_MODULE_OK,
        &Publisher::get_service_thunk,
        &Publisher::release_service_thunk,
    };
}

std::shared_ptr<Module> Publisher::self_module()
{
    std::lock_guard<std::mutex> guard(self_module_mutex_);
    if (!self_module_) {
        self_module_ = std::shared_ptr<SelfModule>(
            new SelfModule(weak_from_this()));
    }
    return self_module_;
}

std::shared_ptr<void> Publisher::lookup(const std::string& instance_id,
    const std::string& type_id)
{
    // registry_mutex_ is recursive so it can be held across the
    // construct_service() call: the override publishes via
    // ServicePublisher::publish, which re-enters register_service on
    // the same thread. Holding the lock across construction
    // serializes concurrent lookups for the same key so they don't
    // both invoke construct_service.
    std::lock_guard<std::recursive_mutex> guard(registry_mutex_);
    const auto cached = registry_.find(Key { instance_id, type_id });
    if (cached != registry_.end()) {
        return cached->second;
    }
    if (!construct_service(instance_id, type_id)) {
        return {};
    }
    const auto constructed = registry_.find(Key { instance_id, type_id });
    if (constructed == registry_.end()) {
        return {};
    }
    return constructed->second;
}

hololink_module_service_t Publisher::get_service_thunk(
    const char* instance_id, const char* type_id)
{
    Publisher* publisher = current_;
    if (!publisher || !instance_id || !type_id) {
        return nullptr;
    }
    std::shared_ptr<void> stored = publisher->lookup(instance_id, type_id);
    if (!stored) {
        return nullptr;
    }
    // Record a strong reference so the instance survives for as long as
    // the host holds the returned pointer, independent of the cache.
    void* raw = stored.get();
    {
        std::lock_guard<std::mutex> guard(publisher->outstanding_mutex_);
        auto [it, inserted] = publisher->outstanding_.try_emplace(
            raw, Outstanding { stored, 0 });
        ++it->second.count;
    }
    return raw;
}

void Publisher::release_service_thunk(hololink_module_service_t instance)
{
    Publisher* publisher = current_;
    if (!publisher || !instance) {
        return;
    }
    // Release one reference; the last release frees the instance unless
    // the cache still holds it.
    std::lock_guard<std::mutex> guard(publisher->outstanding_mutex_);
    const auto it = publisher->outstanding_.find(instance);
    if (it == publisher->outstanding_.end()) {
        return;
    }
    if (--it->second.count == 0) {
        publisher->outstanding_.erase(it);
    }
}

void Publisher::invalidate(const void* service)
{
    Publisher* publisher = current_;
    if (!publisher || !service) {
        return;
    }
    // Match all entries, since a service may be registered under more than
    // one type_id.
    std::lock_guard<std::recursive_mutex> guard(publisher->registry_mutex_);
    for (auto it = publisher->registry_.begin();
         it != publisher->registry_.end();) {
        if (it->second.get() == service) {
            it = publisher->registry_.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace hololink::module

/* ABI check symbol exported by every module that links
 * hololink::module_runtime. The host calls this before
 * hololink_module_init and rejects the load on mismatch. */
extern "C" hololink_module_abi_check_t hololink_module_get_abi_check(void)
{
    hololink_module_abi_check_t check;
    check.magic = HOLOLINK_MODULE_ABI_MAGIC;
    check.api_version = HOLOLINK_MODULE_API_VERSION;
    check.struct_size = sizeof(hololink_module_abi_check_t);
    check.size_of_enumeration_metadata = sizeof(hololink::module::EnumerationMetadata);
    check.align_of_enumeration_metadata = alignof(hololink::module::EnumerationMetadata);
    check.size_of_std_string = sizeof(std::string);
    check.align_of_std_string = alignof(std::string);
    return check;
}
