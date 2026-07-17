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

#ifndef HOLOLINK_MODULE_SENSORS_MAX9295D_MAX9295D_HPP
#define HOLOLINK_MODULE_SENSORS_MAX9295D_MAX9295D_HPP

#include <cstdint>
#include <memory>

#include "hololink/module/hololink.hpp"
#include "hololink/module/i2c.hpp"

namespace hololink::module::sensors::max9295d {

class Max9295d {
public:
    static constexpr uint32_t I2C_ADDRESS_DEFAULT = 0x40;

    static constexpr uint16_t DEV_ID_REG = 0x000D;
    static constexpr uint8_t DEV_ID = 0x95;

    // Control Channel address-translation registers.
    static constexpr uint16_t CC_SRC_A_REG = 0x0042;
    static constexpr uint16_t CC_DST_A_REG = 0x0043;
    static constexpr uint16_t CC_SRC_B_REG = 0x0044;
    static constexpr uint16_t CC_DST_B_REG = 0x0045;

    // GPIO blocks. Per-pin stride is 3 registers (GPIO_A / GPIO_B / GPIO_C).
    // Pin N base address = GPIO_PIN_BASE + 3 * N
    static constexpr uint16_t GPIO_PIN_BASE = 0x02BE;

    explicit Max9295d(std::shared_ptr<HololinkInterfaceV1> hololink,
        uint32_t i2c_address = I2C_ADDRESS_DEFAULT);

    uint32_t get_i2c_address() const;
    void set_i2c_address(uint32_t i2c_address);

    uint8_t get_register(uint16_t reg);
    void set_register(uint16_t reg, uint8_t value);

    void configure_cc_address_translation(uint32_t src_a, uint32_t dst_a,
        uint32_t src_b, uint32_t dst_b);

    void route_gmsl_gpio_to_pin(int pin, uint8_t rx_id);

private:
    uint32_t i2c_address_;
    uint32_t i2c_bus_ = 1;
    std::shared_ptr<I2cInterfaceV1> i2c_;
};

} // namespace hololink::module::sensors::max9295d

#endif /* HOLOLINK_MODULE_SENSORS_MAX9295D_MAX9295D_HPP */
