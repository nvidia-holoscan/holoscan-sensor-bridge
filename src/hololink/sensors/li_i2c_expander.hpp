/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SENSORS_LI_I2C_EXPANDER_HPP
#define SENSORS_LI_I2C_EXPANDER_HPP

#include <cstdint>

#include <memory>

#include "hololink/core/hololink.hpp"

namespace hololink::sensors {

enum class I2CExpanderOutputEN : uint8_t {
    OUTPUT_1 = 0b0001, // for first camera
    OUTPUT_2 = 0b0010, // for another camera
    OUTPUT_3 = 0b0100,
    OUTPUT_4 = 0b1000,
    DEFAULT = 0b0000
};

class LII2CExpander {
public:
    static constexpr uint32_t I2C_EXPANDER_ADDRESS = 0b01110000;

    LII2CExpander(Hololink& hololink, uint32_t i2c_bus);
    void configure(I2CExpanderOutputEN output_en = I2CExpanderOutputEN::DEFAULT);

private:
    std::shared_ptr<Hololink::I2c> i2c_;
};

} // namespace hololink::sensors

#endif /* SENSORS_LI_I2C_EXPANDER_HPP */
