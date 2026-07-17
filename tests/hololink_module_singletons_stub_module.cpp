/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Test fixture: a module .so that, during init, schedules a Reactor
 * callback (via Publisher::reactor()) and emits an HSB_LOG_INFO line
 * from the callback. Proves the host singletons round-trip across the
 * .so boundary and that the per-binary HSB_LOG cache works.
 */

#include <memory>

#include "hololink/module/logging.hpp"
#include "hololink/module/publisher.hpp"
#include "hololink/module/reactor.hpp"
#include "hololink/module/service_locator.h"

// Held alive for the lifetime of the loaded module .so.
static std::shared_ptr<hololink::module::Publisher> g_publisher;
static std::shared_ptr<hololink::module::ReactorV1::Callback> g_callback;

namespace {
// All services this stub exposes are eagerly published below; there
// is no lazy construction path. construct_service is implemented as
// an explicit no-op to satisfy Publisher's pure virtual.
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
    using hololink::module::ReactorV1;

    auto publisher = std::make_shared<TestPublisher>();
    g_publisher = publisher;

    auto result = publisher->setup(init);
    if (result.status != HOLOLINK_MODULE_OK) {
        return result;
    }

    // Schedule a callback that emits a log line via the cached
    // logger. The reactor holds the inner lambda; the lambda
    // captures g_callback, so the shared_ptr keeps this stub's
    // Callback object alive until dispatch completes.
    g_callback = std::make_shared<ReactorV1::Callback>([]() {
        HSB_LOG_INFO("hololink_module_test::singletons-marker from {}",
            "singletons-stub");
    });
    publisher->reactor()->add_callback(g_callback);

    return result;
}
