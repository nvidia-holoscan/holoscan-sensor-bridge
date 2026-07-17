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

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "hololink/module/hololink.hpp"
#include "hololink/module/publisher.hpp"
#include "hololink/module/service.hpp"
#include "hololink/module/status.h"
#include "hololink/module/tutorial/tutorial_device.hpp"

#include "hsb_lite_publisher.hpp"

using namespace hololink::module;
using namespace hololink::module::module_core;

// device_status block: one status LED on bit 0. The LED is wired as a
// pull-down, so clearing bit 0 turns it on and setting bit 0 turns it off.
static constexpr uint32_t TUTORIAL_DEVICE_STATUS = 0x42348000;
static constexpr uint32_t STATUS_LED_BIT = 1u << 0;

/* Concrete Tutorial device status service. configure() resolves the
 * board's already-started HololinkInterfaceV1 control plane (the
 * application fetches and start()s it first); every register access
 * goes through it. */
class TutorialDeviceV1
    : public tutorial::TutorialDeviceInterfaceV1,
      public Service<TutorialDeviceV1> {
public:
    static constexpr const char* type_id = "tutorial_device.impl.v1";
    using Service<TutorialDeviceV1>::get_service;
    using Service<TutorialDeviceV1>::for_each_type_id;
    using ServiceAlias = tutorial::TutorialDeviceInterfaceV1;

    void configure(const EnumerationMetadata& metadata) override
    {
        const std::string hololink_id = HololinkInterfaceV1::locator_id(metadata);
        hololink_ = HololinkInterfaceV1::get_service(this->module(), hololink_id.c_str());
    }

    hololink_module_status_t set_status_led(bool on) override
    {
        // Pull-down: clear bit 0 to turn the LED on, set it to turn it off.
        return on ? hololink_->and_uint32(TUTORIAL_DEVICE_STATUS, ~STATUS_LED_BIT)
                  : hololink_->or_uint32(TUTORIAL_DEVICE_STATUS, STATUS_LED_BIT);
    }

private:
    std::shared_ptr<HololinkInterfaceV1> hololink_;
};

/* Tutorial device channel configuration. The board has one network
 * interface — a single data plane shared by all three sensors, one SIF
 * per sensor — so it overrides the base 1:1 sensor-to-data-plane
 * mapping. Sensors 0 and 1 are IMX274 (left/right); sensor 2 is an IMU. */
class TutorialDeviceChannelConfigurationV1
    : public HsbLiteChannelConfigurationV1 {
public:
    void use_sensor(EnumerationMetadata& metadata, int64_t sensor_number) override
    {
        constexpr int64_t TOTAL_SENSORS = 3;
        if (sensor_number < 0 || sensor_number >= TOTAL_SENSORS) {
            throw std::runtime_error(
                "While selecting a Tutorial device sensor: sensor_number "
                + std::to_string(sensor_number)
                + " is out of range (Tutorial device supports 0.."
                + std::to_string(TOTAL_SENSORS - 1) + ")");
        }
        // One physical data plane (0) shared by every sensor, one SIF each.
        hsb_lite_sensor_metadata(metadata, sensor_number,
            /*data_plane=*/0, /*sifs_per_sensor=*/1);
    }
};

/* Tutorial device Publisher. Specializes the canonical HSB-Lite
 * publisher: it keeps every inherited service (control plane, I2C,
 * oscillator, data channels, receivers) so the IMX274 driver runs
 * unchanged, and adds only what the Tutorial device changes — its
 * name, its single-data-plane sensor layout, and the status LED. */
class TutorialDevicePublisher : public HsbLitePublisher {
protected:
    std::string module_name() const override { return "tutorial"; }

    void publish_channel_configuration() override
    {
        auto impl = std::make_shared<TutorialDeviceChannelConfigurationV1>();
        ServicePublisher<HsbLiteChannelConfigurationV1>(shared_from_this())
            .publish("", impl);
    }

    bool construct_overrides(
        const std::string& instance_id,
        const std::string& type_id) override
    {
        if (!Publisher::has_type_id<TutorialDeviceV1>(type_id)) {
            return false;
        }
        auto impl = std::make_shared<TutorialDeviceV1>();
        ServicePublisher<TutorialDeviceV1>(shared_from_this())
            .publish(instance_id, impl);
        return true;
    }
};

// Held alive for the lifetime of this loaded module .so.
static std::shared_ptr<Publisher> g_publisher;

extern "C" hololink_module_services_t
hololink_module_init(const hololink_module_init_t* init)
{
    auto publisher = std::make_shared<TutorialDevicePublisher>();
    g_publisher = publisher;
    return publisher->setup(init);
}
