/*
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

#ifndef HOLOLINK_MODULE_TAUROTECH_DA326_TAUROTECH_DA326_HPP
#define HOLOLINK_MODULE_TAUROTECH_DA326_TAUROTECH_DA326_HPP

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/module.hpp"
#include "hololink/module/service.hpp"
#include "hololink/module/status.h"

namespace hololink::module::taurotech_da326 {

/* TauroTech DA326 board supplement. Exposes board-specific ops:
 * camera reset release, deserializer power cycle, power check, and
 * I2C clock setup. All register accesses go through HololinkInterfaceV1. */
class TauroTechDa326InterfaceV1
    : public ConfigurableService<TauroTechDa326InterfaceV1> {
public:
    static constexpr const char* type_id = "taurotech_da326.v1";

    static std::string locator_id(const EnumerationMetadata& metadata)
    {
        return "serial=" + metadata.get<std::string>("serial_number");
    }

    virtual ~TauroTechDa326InterfaceV1() = default;

    virtual hololink_module_status_t release_reset() = 0;
    virtual hololink_module_status_t power_cycle() = 0;
    virtual hololink_module_status_t check_power() = 0;
    virtual hololink_module_status_t setup_clock() = 0;

    /* Returns the HololinkInterfaceV1 backing this board so callers can
     * reach I2C buses and data channels through the module surface. */
    virtual std::shared_ptr<HololinkInterfaceV1> hololink() = 0;
};

} // namespace hololink::module::taurotech_da326

#endif // HOLOLINK_MODULE_TAUROTECH_DA326_TAUROTECH_DA326_HPP
