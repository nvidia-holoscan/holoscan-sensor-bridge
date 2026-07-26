/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Test fixture: a module .so that publishes an EnumerationInterfaceV1
 * whose update_metadata stamps a marker field into the metadata.
 * Used by the enumeration test to verify Adapter::enumerate finds
 * this stub by UUID and reaches its update_metadata across the .so
 * boundary.
 */

#include <cstdint>
#include <memory>

#include "hololink/module/enumeration.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/publisher.hpp"
#include "hololink/module/service_locator.h"

namespace {

class EnumerationImpl : public hololink::module::EnumerationInterfaceV1 {
public:
    hololink_module_status_t update_metadata(
        hololink::module::EnumerationMetadata& metadata,
        const uint8_t* /*raw_packet*/, size_t /*raw_packet_len*/) override
    {
        metadata["enriched_by_stub"] = std::string("yes");
        metadata["control_port"] = static_cast<int64_t>(8192);
        return HOLOLINK_MODULE_OK;
    }
};

} // namespace

// Held alive for the lifetime of the loaded module .so.
static std::shared_ptr<hololink::module::Publisher> g_publisher;
static std::shared_ptr<EnumerationImpl> g_enumeration;

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
    using hololink::module::EnumerationInterfaceV1;
    using hololink::module::ServicePublisher;

    g_publisher = std::make_shared<TestPublisher>();

    auto result = g_publisher->setup(init);
    if (result.status != HOLOLINK_MODULE_OK) {
        return result;
    }

    g_enumeration = std::make_shared<EnumerationImpl>();
    ServicePublisher<EnumerationInterfaceV1>(g_publisher).publish("", g_enumeration);

    return result;
}
