/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Test fixture: a module .so that publishes a TestServiceV1 whose
 * answer() value comes from a HostTestServiceV1 the host pre-publishes
 * — proving the locator works in both directions across the .so
 * boundary.
 */

#include <memory>

#include "hololink/module/module.hpp"
#include "hololink/module/publisher.hpp"
#include "hololink/module/service.hpp"
#include "hololink/module/service_locator.h"

// Anonymous namespace holds file-local TYPES only — function-local
// classes / anonymous namespaces are the only tools that give types
// internal linkage. File-local variables use `static` at file scope
// below.
namespace {

class HostTestServiceV1
    : public hololink::module::Service<HostTestServiceV1> {
public:
    static constexpr const char* type_id = "hololink_module_test.host.v1";
    virtual ~HostTestServiceV1() = default;
    virtual int answer() const = 0;
};

class TestServiceV1
    : public hololink::module::Service<TestServiceV1> {
public:
    static constexpr const char* type_id = "hololink_module_test.module.v1";
    virtual ~TestServiceV1() = default;
    virtual int answer() const = 0;
};

class TestServiceImpl : public TestServiceV1 {
public:
    explicit TestServiceImpl(int value)
        : value_(value)
    {
    }
    int answer() const override { return value_; }

private:
    int value_;
};

} // namespace

// Held alive for the lifetime of the loaded module .so.
static std::shared_ptr<hololink::module::Publisher> g_publisher;
static std::shared_ptr<TestServiceImpl> g_test_service;

namespace {
class TestPublisher : public hololink::module::Publisher {
public:
    bool construct_service(
        const std::string& /*instance_id*/,
        const std::string& /*type_id*/) override
    {
        return false;
    }
};
} // namespace

extern "C" hololink_module_services_t
hololink_module_init(const hololink_module_init_t* init)
{
    using hololink::module::ServicePublisher;

    auto publisher = std::make_shared<TestPublisher>();
    g_publisher = publisher;

    auto result = publisher->setup(init);
    if (result.status != HOLOLINK_MODULE_OK) {
        return result;
    }

    // Look up the host-published "host" service and surface its value.
    auto host_service = HostTestServiceV1::get_service(
        publisher->host_module(), "");
    const int value = host_service ? host_service->answer() : -1;

    g_test_service = std::make_shared<TestServiceImpl>(value);
    ServicePublisher<TestServiceV1>(g_publisher).publish("", g_test_service);

    return result;
}
