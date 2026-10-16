/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// hsb_pcie module entry point: hololink_module_init for hololink_<UUID>.so. Same shape as
// module/hsb_lite/module_entry.cpp with HsbPciePublisher.

#include <memory>

#include "hololink/module/publisher.hpp"
#include "hololink/module/service_locator.h"

#include "hsb_pcie_publisher.hpp"

// Held alive for the lifetime of this loaded module .so. The host's
// LoadedModule keeps the .so resident; the publisher outlives every
// service handle the host obtains through it.
static std::shared_ptr<hololink::module::Publisher> g_publisher;

extern "C" hololink_module_services_t
hololink_module_init(const hololink_module_init_t* init)
{
    auto publisher = std::make_shared<hololink::module::hsb_pcie::HsbPciePublisher>();
    g_publisher = publisher;
    return publisher->setup(init);
}
