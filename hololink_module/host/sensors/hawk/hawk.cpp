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

#include "hawk.hpp"

namespace hololink::module::sensors::hawk {

Hawk::Hawk(std::shared_ptr<HololinkInterfaceV1> hololink, bool skip_i2c)
    : serializer_(std::make_shared<max9295d::Max9295d>(hololink))
    , sensor_left_(std::make_shared<ar0234::Ar0234>(hololink, SENSOR_LEFT_I2C_ADDRESS, ar0234::Ar0234::CAM_I2C_BUS, skip_i2c))
    , sensor_right_(std::make_shared<ar0234::Ar0234>(hololink, SENSOR_RIGHT_I2C_ADDRESS, ar0234::Ar0234::CAM_I2C_BUS, skip_i2c))
{
}

void Hawk::configure(ar0234::ar0234_mode::Mode mode, bool fsync)
{
    for (auto& sensor : sensors()) {
        sensor->configure_camera(mode, fsync);
    }
}

void Hawk::set_mode(ar0234::ar0234_mode::Mode mode)
{
    for (auto& sensor : sensors()) {
        sensor->set_mode(mode);
    }
}

void Hawk::set_exposure_reg(uint16_t value)
{
    for (auto& sensor : sensors()) {
        sensor->set_exposure_reg(value);
    }
}

void Hawk::test_pattern(uint16_t pattern)
{
    for (auto& sensor : sensors()) {
        sensor->test_pattern(pattern);
    }
}

void Hawk::start()
{
    sensor_left_->start();
    try {
        sensor_right_->start();
    } catch (...) {
        try {
            sensor_left_->stop();
        } catch (...) {
        }
        throw;
    }
}

void Hawk::stop()
{
    for (auto& sensor : sensors()) {
        sensor->stop();
    }
}

void Hawk::configure_converter(std::shared_ptr<hololink::csi::CsiConverter> converter)
{
    sensor_left_->configure_converter(converter);
}

hololink::csi::PixelFormat Hawk::pixel_format() const { return sensor_left_->pixel_format(); }
hololink::csi::BayerFormat Hawk::bayer_format() const { return sensor_left_->bayer_format(); }
int64_t Hawk::width() const { return sensor_left_->width(); }
int64_t Hawk::height() const { return sensor_left_->height(); }

uint32_t Hawk::get_serializer_i2c_address() const { return serializer_->get_i2c_address(); }
void Hawk::set_serializer_i2c_address(uint32_t i2c_address)
{
    serializer_->set_i2c_address(i2c_address);
}

void Hawk::remap_sensor_addresses(uint32_t new_left, uint32_t new_right)
{
    serializer_->configure_cc_address_translation(
        /*src_a=*/new_left, /*dst_a=*/SENSOR_LEFT_I2C_ADDRESS,
        /*src_b=*/new_right, /*dst_b=*/SENSOR_RIGHT_I2C_ADDRESS);
    sensor_left_->set_i2c_address(new_left);
    sensor_right_->set_i2c_address(new_right);
}

} // namespace hololink::module::sensors::hawk
