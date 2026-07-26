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

#include "max9295d.hpp"

#include <stdexcept>
#include <vector>

#include <fmt/format.h>

#include <hololink/core/logging_internal.hpp>

namespace hololink::module::sensors::max9295d {

Max9295d::Max9295d(std::shared_ptr<HololinkInterfaceV1> hololink, uint32_t i2c_address)
    : i2c_address_(i2c_address)
{
    i2c_ = hololink->get_i2c<>(i2c_bus_, i2c_address_);
}

uint32_t Max9295d::get_i2c_address() const
{
    return i2c_address_;
}

void Max9295d::set_i2c_address(uint32_t i2c_address)
{
    set_register(0x0000, static_cast<uint8_t>(i2c_address << 1));
    i2c_address_ = i2c_address;
}

uint8_t Max9295d::get_register(uint16_t reg)
{
    std::vector<uint8_t> write_bytes = { uint8_t(reg >> 8), uint8_t(reg & 0xFF) };
    std::vector<uint8_t> read_bytes(1);
    const auto s = i2c_->i2c_transaction(i2c_address_, write_bytes, read_bytes);
    if (s != HOLOLINK_MODULE_OK) {
        throw std::runtime_error(fmt::format(
            "While reading MAX9295D register: i2c_addr=0x{:02X}, register=0x{:04X}, status={}",
            i2c_address_, reg, s));
    }
    return read_bytes[0];
}

void Max9295d::set_register(uint16_t reg, uint8_t value)
{
    std::vector<uint8_t> write_bytes = { uint8_t(reg >> 8), uint8_t(reg & 0xFF), value };
    std::vector<uint8_t> read_bytes;
    const auto s = i2c_->i2c_transaction(i2c_address_, write_bytes, read_bytes);
    if (s != HOLOLINK_MODULE_OK) {
        HSB_LOG_ERROR("Failed: i2c_addr=0x{:02X}, register=0x{:04X}, value=0x{:02X}, status={}",
            i2c_address_, reg, value, s);
        throw std::runtime_error(fmt::format(
            "While writing MAX9295D register: i2c_addr=0x{:02X}, register=0x{:04X}, status={}",
            i2c_address_, reg, s));
    }
}

void Max9295d::configure_cc_address_translation(uint32_t src_a, uint32_t dst_a,
    uint32_t src_b, uint32_t dst_b)
{
    set_register(CC_SRC_A_REG, static_cast<uint8_t>(src_a << 1));
    set_register(CC_DST_A_REG, static_cast<uint8_t>(dst_a << 1));
    set_register(CC_SRC_B_REG, static_cast<uint8_t>(src_b << 1));
    set_register(CC_DST_B_REG, static_cast<uint8_t>(dst_b << 1));
}

void Max9295d::route_gmsl_gpio_to_pin(int pin, uint8_t rx_id)
{
    uint16_t base = GPIO_PIN_BASE + 3u * static_cast<uint16_t>(pin);
    set_register(base, 0x04); // GPIO_A: GPIO_RX_EN=1
    set_register(base + 1, 0x60); // GPIO_B: push-pull, pull-up
    set_register(base + 2, static_cast<uint8_t>(rx_id & 0x1F)); // GPIO_C: GPIO_RX_ID
}

} // namespace hololink::module::sensors::max9295d
