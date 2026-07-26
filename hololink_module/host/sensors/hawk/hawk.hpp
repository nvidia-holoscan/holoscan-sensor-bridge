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

#ifndef HOLOLINK_MODULE_SENSORS_HAWK_HAWK_HPP
#define HOLOLINK_MODULE_SENSORS_HAWK_HAWK_HPP

#include "../ar0234/ar0234.hpp"
#include "../serializers/max9295d/max9295d.hpp"

#include <array>
#include <memory>

namespace hololink::module::sensors::hawk {

/// One MAX9295D serializer + two AR0234 sensors on one GMSL link.
class Hawk {
public:
    static constexpr uint32_t SENSOR_LEFT_I2C_ADDRESS = ar0234::Ar0234::CAM_A_I2C_ADDRESS;
    static constexpr uint32_t SENSOR_RIGHT_I2C_ADDRESS = ar0234::Ar0234::CAM_B_I2C_ADDRESS;

    explicit Hawk(std::shared_ptr<HololinkInterfaceV1> hololink, bool skip_i2c = false);

    max9295d::Max9295d& serializer() { return *serializer_; }
    const max9295d::Max9295d& serializer() const { return *serializer_; }

    ar0234::Ar0234& sensor_left() { return *sensor_left_; }
    ar0234::Ar0234& sensor_right() { return *sensor_right_; }

    std::array<std::shared_ptr<ar0234::Ar0234>, 2> sensors() const
    {
        return { sensor_left_, sensor_right_ };
    }

    // ---- Per-sensor operations (broadcast to both AR0234s) ----
    void configure(ar0234::ar0234_mode::Mode mode, bool fsync = false);
    void set_mode(ar0234::ar0234_mode::Mode mode);
    void set_exposure_reg(uint16_t value);
    void test_pattern(uint16_t pattern);
    void start();
    void stop();

    // ---- Format/dims -- both sensors are identical AR0234s ----
    void configure_converter(std::shared_ptr<hololink::csi::CsiConverter> converter);
    hololink::csi::PixelFormat pixel_format() const;
    hololink::csi::BayerFormat bayer_format() const;
    int64_t width() const;
    int64_t height() const;

    // ---- Serializer address delegation ----
    uint32_t get_serializer_i2c_address() const;
    void set_serializer_i2c_address(uint32_t i2c_address);

    // ---- Sensor I2C address remap (via serializer CC translation) ----
    void remap_sensor_addresses(uint32_t new_left, uint32_t new_right);

private:
    std::shared_ptr<max9295d::Max9295d> serializer_;
    std::shared_ptr<ar0234::Ar0234> sensor_left_;
    std::shared_ptr<ar0234::Ar0234> sensor_right_;
};

} // namespace hololink::module::sensors::hawk

#endif /* HOLOLINK_MODULE_SENSORS_HAWK_HAWK_HPP */
