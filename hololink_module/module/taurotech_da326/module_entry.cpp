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

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "hololink/module/publisher.hpp"
#include "hololink/module/service.hpp"
#include "hololink/module/status.h"
#include "hololink/module/taurotech_da326/taurotech_da326.hpp"

#include "hsb_lite_publisher.hpp"

using namespace hololink::module;
using namespace hololink::module::module_core;

namespace {

// GPIO registers (FPGA absolute addresses).
// Direction register: bit SET = INPUT, bit CLEAR = OUTPUT.
constexpr uint32_t GPIO_DIRECTION_BASE = 0x0000'002C;
constexpr uint32_t GPIO_OUTPUT_BASE = 0x0000'000C;
constexpr uint32_t GPIO_STATUS_BASE = 0x0000'008C;

} // namespace

/* Concrete TauroTechDa326 board service.
 *
 * Follows the same HololinkV1 pattern as HsbLiteV1: configure() fetches
 * the already-created HololinkV1 from the module's service registry (the
 * application must call HololinkInterfaceV1::get_service(metadata) and
 * start() first). All register I/O goes through HololinkInterfaceV1. */
class TauroTechDa326V1
    : public taurotech_da326::TauroTechDa326InterfaceV1,
      public Service<TauroTechDa326V1> {
public:
    static constexpr const char* type_id = "taurotech_da326.module.v1";
    using Service<TauroTechDa326V1>::get_service;
    using Service<TauroTechDa326V1>::for_each_type_id;
    using ServiceAlias = taurotech_da326::TauroTechDa326InterfaceV1;

    void configure(const EnumerationMetadata& metadata) override
    {
        const std::string hl_id = HololinkInterfaceV1::locator_id(metadata);
        hololink_ = HololinkInterfaceV1::get_service(this->module(), hl_id.c_str());
    }

    // --- Board ops via HololinkInterfaceV1 ---

    hololink_module_status_t release_reset() override
    {
        constexpr uint32_t CAMERA_CTRL_REG = 0x00000008;
        constexpr uint32_t CAMERA_RESET_RELEASE = 0x00000003;
        return hololink_->write_uint32({ CAMERA_CTRL_REG }, { CAMERA_RESET_RELEASE });
    }

    hololink_module_status_t power_cycle() override
    {
        hololink_module_status_t s;

        // Set all 11 GPIO pins (0-10) to INPUT (bit set = IN).
        s = hololink_->or_uint32(GPIO_DIRECTION_BASE, 0x7FFu);
        if (s != HOLOLINK_MODULE_OK)
            return s;

        // Pin 0 to OUTPUT (bit clear = OUT) then LOW.
        s = hololink_->and_uint32(GPIO_DIRECTION_BASE, ~(1u << 0u));
        if (s != HOLOLINK_MODULE_OK)
            return s;
        s = hololink_->and_uint32(GPIO_OUTPUT_BASE, ~(1u << 0u));
        if (s != HOLOLINK_MODULE_OK)
            return s;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::this_thread::sleep_for(std::chrono::seconds(1));

        // Pin 0 remains OUTPUT — drive HIGH.
        s = hololink_->or_uint32(GPIO_OUTPUT_BASE, 1u << 0u);
        if (s != HOLOLINK_MODULE_OK)
            return s;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::this_thread::sleep_for(std::chrono::seconds(1));

        return HOLOLINK_MODULE_OK;
    }

    hololink_module_status_t check_power() override
    {
        // Pin 1 to INPUT (bit set = IN).
        hololink_module_status_t s = hololink_->or_uint32(GPIO_DIRECTION_BASE, 1u << 1u);
        if (s != HOLOLINK_MODULE_OK)
            return s;

        // Read pin 1 from STATUS register; LOW (bit clear) means power fault.
        std::vector<uint32_t> vals;
        s = hololink_->read_uint32({ GPIO_STATUS_BASE }, vals);
        if (s != HOLOLINK_MODULE_OK || vals.empty()) {
            return HOLOLINK_MODULE_NETWORK_ERROR;
        }
        if (!(vals[0] & (1u << 1u))) {
            throw std::runtime_error("Power issue detected.");
        }
        return HOLOLINK_MODULE_OK;
    }

    hololink_module_status_t setup_clock() override
    {
        // I2C controller base + clock-count register offset, from hololink::I2C_CTRL / I2C_REG_CLK_CNT.
        constexpr uint32_t I2C_CLK_REG = 0x0300'020C;
        constexpr uint32_t I2C_CLK_DIVIDER = 0x19;
        return hololink_->write_uint32({ I2C_CLK_REG }, { I2C_CLK_DIVIDER });
    }

    std::shared_ptr<HololinkInterfaceV1> hololink() override
    {
        if (!hololink_) {
            throw std::runtime_error(
                "TauroTechDa326V1::hololink: configure() has not been called.");
        }
        return hololink_;
    }

private:
    std::shared_ptr<HololinkInterfaceV1> hololink_;
};

/* TauroTech DA326 module Publisher. */
/* TauroTech DA326 channel-configuration override.
 *
 * The DA326 has two sensors on a single data plane (data_plane=0), one
 * SIF per sensor. The base HsbLiteChannelConfigurationV1 incorrectly maps
 * data_plane == sensor_number (HSB-Lite 1:1); this override keeps
 * data_plane=0 for both sensors so hif_address stays at 0x2000300. */
class TauroTechDa326ChannelConfigurationV1
    : public hololink::module::module_core::HsbLiteChannelConfigurationV1 {
public:
    void use_sensor(hololink::module::EnumerationMetadata& metadata,
        int64_t sensor_number) override
    {
        constexpr int64_t TOTAL_SENSORS = 2;
        if (sensor_number < 0 || sensor_number >= TOTAL_SENSORS) {
            throw std::runtime_error(
                std::string("While selecting a TauroTech DA326 sensor: ")
                + "sensor_number " + std::to_string(sensor_number)
                + " is out of range (DA326 supports 0-1)");
        }
        // Single data plane (0) shared by both sensors; one SIF per sensor.
        hololink::module::module_core::hsb_lite_sensor_metadata(
            metadata, sensor_number, /*data_plane=*/0, /*sifs_per_sensor=*/1);
    }
};

class TauroTechDa326Publisher : public HsbLitePublisher {
protected:
    std::string module_name() const override { return "taurotech_da326"; }

    void publish_channel_configuration() override
    {
        auto impl = std::make_shared<TauroTechDa326ChannelConfigurationV1>();
        hololink::module::ServicePublisher<
            hololink::module::module_core::HsbLiteChannelConfigurationV1>(
            shared_from_this())
            .publish("", impl);
    }

    bool construct_overrides(
        const std::string& instance_id,
        const std::string& type_id) override
    {
        if (!Publisher::has_type_id<TauroTechDa326V1>(type_id)) {
            return false;
        }
        auto impl = std::make_shared<TauroTechDa326V1>();
        ServicePublisher<TauroTechDa326V1>(shared_from_this())
            .publish(instance_id, impl);
        return true;
    }
};

static std::shared_ptr<Publisher> g_publisher;

extern "C" hololink_module_services_t
hololink_module_init(const hololink_module_init_t* init)
{
    auto publisher = std::make_shared<TauroTechDa326Publisher>();
    g_publisher = publisher;
    return publisher->setup(init);
}
