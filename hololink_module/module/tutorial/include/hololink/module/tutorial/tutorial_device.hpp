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

#ifndef HOLOLINK_MODULE_TUTORIAL_TUTORIAL_DEVICE_HPP
#define HOLOLINK_MODULE_TUTORIAL_TUTORIAL_DEVICE_HPP

#include <memory>
#include <string>

#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/service.hpp"
#include "hololink/module/status.h"

namespace hololink::module::tutorial {

/* Tutorial device status service. This is the one board-specific
 * capability the Tutorial device adds on top of the canonical HSB
 * services: a status LED. The application fetches it by metadata; the
 * implementation lives in the module and drives the LED through the
 * board's control plane. */
class TutorialDeviceInterfaceV1
    : public ConfigurableService<TutorialDeviceInterfaceV1> {
public:
    static constexpr const char* type_id = "tutorial_device.v1";

    static std::string locator_id(const EnumerationMetadata& metadata)
    {
        return "serial=" + metadata.get<std::string>("serial_number");
    }

    virtual ~TutorialDeviceInterfaceV1() = default;

    /* Turn the board's status LED on (true) or off (false). */
    virtual hololink_module_status_t set_status_led(bool on) = 0;
};

} // namespace hololink::module::tutorial

#endif // HOLOLINK_MODULE_TUTORIAL_TUTORIAL_DEVICE_HPP
