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

#ifndef HOLOLINK_MODULE_SENSORS_AR0234_AR0234_HPP
#define HOLOLINK_MODULE_SENSORS_AR0234_AR0234_HPP

#include "ar0234_mode.hpp"

#include "hololink/module/hololink.hpp"
#include "hololink/module/i2c.hpp"
#include <hololink/core/csi_controller.hpp>
#include <hololink/core/csi_formats.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace hololink::module::sensors::ar0234 {

class Ar0234 {
public:
    static constexpr uint32_t CAM_A_I2C_ADDRESS = 0x10;
    static constexpr uint32_t CAM_B_I2C_ADDRESS = 0x18;
    static constexpr uint32_t CAM_I2C_BUS = 1;
    static constexpr uint16_t DEV_ID_REG = 0x3000;
    static constexpr uint16_t DEV_ID = 0x0A56;

    explicit Ar0234(std::shared_ptr<HololinkInterfaceV1> hololink,
        uint32_t i2c_address = CAM_A_I2C_ADDRESS,
        uint32_t i2c_bus = CAM_I2C_BUS,
        bool skip_i2c = false);

    uint32_t get_i2c_address() const;
    void set_i2c_address(uint32_t i2c_address);

    void configure_camera(ar0234_mode::Mode mode, bool fsync = false);
    void set_mode(ar0234_mode::Mode mode);
    void start();
    void stop();

    void configure_converter(std::shared_ptr<hololink::csi::CsiConverter> converter);

    void set_exposure_reg(uint16_t value = 0x02DC);
    void test_pattern(uint16_t pattern = 2);

    uint16_t get_register(uint16_t reg);
    void set_register(uint16_t reg, uint16_t value);

    hololink::csi::PixelFormat pixel_format() const { return pixel_format_; }
    hololink::csi::BayerFormat bayer_format() const { return hololink::csi::BayerFormat::GRBG; }
    int64_t width() const { return static_cast<int64_t>(width_); }
    int64_t height() const { return static_cast<int64_t>(height_); }

    hololink::csi::PixelFormat get_pixel_format() const { return pixel_format_; }
    hololink::csi::BayerFormat get_bayer_format() const { return hololink::csi::BayerFormat::GRBG; }
    int64_t get_width() const { return static_cast<int64_t>(width_); }
    int64_t get_height() const { return static_cast<int64_t>(height_); }

private:
    template <typename ContainerT>
    void apply_register_table(const ContainerT& table);

    std::shared_ptr<I2cInterfaceV1> i2c_;
    uint32_t i2c_address_;
    uint32_t i2c_bus_;
    bool skip_i2c_;
    bool fsync_ = false;

    ar0234_mode::Mode mode_ = ar0234_mode::Unknown;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    hololink::csi::PixelFormat pixel_format_ = hololink::csi::PixelFormat::RAW_10;
};

} // namespace hololink::module::sensors::ar0234

#endif /* HOLOLINK_MODULE_SENSORS_AR0234_AR0234_HPP */
