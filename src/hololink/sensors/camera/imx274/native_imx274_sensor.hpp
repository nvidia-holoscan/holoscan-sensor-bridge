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

#ifndef SENSORS_CAMERA_IMX274_NATIVE_IMX274_SENSOR_HPP
#define SENSORS_CAMERA_IMX274_NATIVE_IMX274_SENSOR_HPP

#include <array>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hololink/core/csi_controller.hpp>
#include <hololink/core/csi_formats.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/sensors/camera/camera_sensor.hpp>
#include <hololink/sensors/li_i2c_expander.hpp>

#include "imx274_mode.hpp"

namespace hololink::sensors {

class NativeImx274Sensor : public CameraSensor {
public:
    // Driver info
    static constexpr const char* DRIVER_NAME = "IMX274-NATIVE";
    static constexpr uint32_t VERSION = 1;

    // Camera I2C address
    static constexpr uint32_t I2C_ADDRESS = 0b00011010;

    NativeImx274Sensor(DataChannel& data_channel, int expander_configuration = 0, uint32_t i2c_bus = CAM_I2C_BUS);

    ~NativeImx274Sensor() override;

    // Override base methods
    void configure(CameraMode mode) override;
    void set_mode(CameraMode mode) override;
    void start() override;
    void stop() override;

    void configure_converter(std::shared_ptr<hololink::csi::CsiConverter> converter) override;
    // Additional methods
    void setup_clock();
    uint32_t get_version() const;
    void test_pattern(uint8_t pattern = 0);

    // Exposure, analog gain, and digital gain
    void set_exposure_reg(int32_t value = 0x0C);
    void set_analog_gain_reg(int32_t value = 0x0C);
    void set_digital_gain_reg(int32_t value = 0x0000);

protected:
    void initialize_supported_modes();
    uint8_t get_register(uint16_t reg);
    void set_register(uint16_t reg, uint8_t value,
        const std::shared_ptr<Timeout>& timeout = nullptr);
    void configure_camera(CameraMode mode);
    template <typename ContainerT>
    void apply_register_settings(const ContainerT& settings);

private:
    std::unordered_map<int, std::shared_ptr<Imx274FrameFormat>> mode_frame_formats_;

    I2CExpanderOutputEN i2c_expander_configuration_ = I2CExpanderOutputEN::DEFAULT;

    std::shared_ptr<Hololink> hololink_;
    std::shared_ptr<Hololink::I2c> i2c_;
    std::shared_ptr<LII2CExpander> i2c_expander_;

    mutable std::mutex mutex_;
};

} // namespace hololink::sensors

#endif /* SENSORS_CAMERA_IMX274_NATIVE_IMX274_SENSOR_HPP */
