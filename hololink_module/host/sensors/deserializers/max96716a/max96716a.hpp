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

#ifndef HOLOLINK_MODULE_SENSORS_MAX96716A_MAX96716A_HPP
#define HOLOLINK_MODULE_SENSORS_MAX96716A_MAX96716A_HPP

#include <cstdint>
#include <memory>

#include "hololink/module/hololink.hpp"
#include "hololink/module/i2c.hpp"

namespace hololink::module::sensors::max96716a {

class Max96716a {
public:
    static constexpr uint32_t I2C_ADDRESS = 0x28;

    static constexpr uint16_t DEV_ID_REG = 0x000D;
    static constexpr uint8_t DEV_ID = 0xBE;

    static constexpr uint16_t VIDEO_PIPE_SEL = 0x0161;

    // GPIO blocks. Per-pin stride is 3 registers (GPIO_A / GPIO_B / GPIO_C).
    // Pin N base = GPIO_PIN_BASE + 3 * N
    static constexpr uint16_t GPIO_PIN_BASE = 0x02B0;
    static constexpr uint16_t GPIO_ALT_PAGE_OFFSET = 0x5000;

    enum class GmslLink {
        LINK_A = 0,
        LINK_B = 1,
    };

    enum class VideoPipe {
        PIPE_Y = 0,
        PIPE_Z = 1,
    };

    explicit Max96716a(std::shared_ptr<HololinkInterfaceV1> hololink);

    uint8_t get_register(uint16_t reg);
    void set_register(uint16_t reg, uint8_t value);

    void enable_link_exclusive(GmslLink link);
    void enable_both_links();

    uint8_t stream_id_to_pipe_mapping(GmslLink link, int stream_id, VideoPipe pipe);

    void configure_video_pipe();
    void route_pin_to_gmsl_gpio(GmslLink link, int pin, uint8_t tx_id);

private:
    uint32_t i2c_address_ = I2C_ADDRESS;
    uint32_t i2c_bus_ = 1;
    std::shared_ptr<I2cInterfaceV1> i2c_;
};

} // namespace hololink::module::sensors::max96716a

#endif /* HOLOLINK_MODULE_SENSORS_MAX96716A_MAX96716A_HPP */
