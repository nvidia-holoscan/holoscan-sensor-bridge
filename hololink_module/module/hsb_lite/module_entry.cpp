/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <string>

#include "hololink/module/publisher.hpp"
#include "hololink/module/service_locator.h"

#include "hsb_lite_publisher.hpp"

/* Canonical HSB-Lite supplement Publisher. Inherits the canonical
 * branch chain + enumeration body from module_core::HsbLitePublisher;
 * the only thing this supplement contributes is its own identity. */
class HsbLitePublisher : public hololink::module::module_core::HsbLitePublisher {
protected:
    std::string module_name() const override { return "hsb_lite"; }
};

// Held alive for the lifetime of this loaded module .so. The host's
// LoadedModule keeps the .so resident; the publisher outlives every
// service handle the host obtains through it.
static std::shared_ptr<hololink::module::Publisher> g_publisher;

extern "C" hololink_module_services_t
hololink_module_init(const hololink_module_init_t* init)
{
    auto publisher = std::make_shared<HsbLitePublisher>();
    g_publisher = publisher;
    return publisher->setup(init);
}
