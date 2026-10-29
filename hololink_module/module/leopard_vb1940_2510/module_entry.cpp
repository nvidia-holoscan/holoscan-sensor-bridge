/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>

#include "hololink/module/publisher.hpp"
#include "hololink/module/service_locator.h"

#include "leopard_vb1940_2510_publisher.hpp"

// Held alive for the lifetime of this loaded module .so. The host's
// LoadedModule keeps the .so resident; the publisher outlives every
// service handle the host obtains through it.
static std::shared_ptr<hololink::module::Publisher> g_publisher;

extern "C" hololink_module_services_t
hololink_module_init(const hololink_module_init_t* init)
{
    auto publisher = std::make_shared<
        hololink::module::leopard_vb1940::LeopardVb1940_2510_Publisher>();
    g_publisher = publisher;
    return publisher->setup(init);
}
