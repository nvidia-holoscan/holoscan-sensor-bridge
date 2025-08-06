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

#ifndef SENSORS_CAMERA_VB1940_NATIVE_VB1940_SENSOR_HPP
#define SENSORS_CAMERA_VB1940_NATIVE_VB1940_SENSOR_HPP

#include <array>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hololink/core/csi_controller.hpp>
#include <hololink/core/csi_formats.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/sensors/camera/camera_sensor.hpp>
#include <hololink/sensors/camera/vb1940/vb1940_mode.hpp>
#include <hololink/sensors/li_i2c_expander.hpp>

namespace hololink::sensors {

// Calibration data structures
struct CalibrationData {
    enum class CameraIndex : uint8_t {
        kLeft = 0,
        kRight = 1
    };
    // values: (fx, fy, cx, cy)
    std::array<std::array<double, 4>, 2> intrinsic_parameters;
    // values: (k1, k2, p1, p2, k3, k4, k5, k6)
    std::array<std::array<double, 8>, 2> distortion_parameters;
    // Rotation parameters (Rx, Ry, Rz)
    std::array<double, 3> R;
    // Translation parameters (Tx, Ty, Tz)
    std::array<double, 3> T;

    std::string to_string() const;
};

class NativeVb1940Sensor : public CameraSensor, public Synchronizable, public std::enable_shared_from_this<NativeVb1940Sensor> {
public:
    // Driver info
    static constexpr const char* DRIVER_NAME = "VB1940-NATIVE";
    static constexpr uint32_t VERSION = 1;

    // Camera I2C address
    static constexpr uint32_t I2C_ADDRESS = 0x10;

    // EEPROM constants
    static constexpr uint32_t EEPROM_I2C_ADDRESS = 0x51;
    static constexpr uint32_t EEPROM_MAX_PAGE_NUM = 256;
    static constexpr uint32_t EEPROM_PAGE_SIZE = 64;
    static constexpr uint32_t CALIB_SIZE = 256;
    static constexpr uint32_t VCL_EN_I2C_ADDRESS_1 = 0x70;
    static constexpr uint32_t VCL_EN_I2C_ADDRESS_2 = 0x71;
    static constexpr uint32_t VCL_PWM_I2C_ADDRESS = 0x21;

    NativeVb1940Sensor(DataChannel& data_channel, std::shared_ptr<Synchronizer> synchronizer = Synchronizer::null_synchronizer());
    ~NativeVb1940Sensor() override;

    void start() override;
    void stop() override;

    // Override base methods
    void configure(CameraMode mode) override;
    void set_mode(CameraMode mode) override;

    void configure_converter(std::shared_ptr<hololink::csi::CsiConverter> converter) override;

    // Additional methods
    void setup_clock();
    uint32_t get_version() const;

    // Register access methods
    uint8_t get_register(uint16_t reg);
    uint32_t get_register_32(uint16_t reg);

    // Exposure and analog gain
    void set_exposure_reg(int32_t value = 0x0014);
    void set_analog_gain_reg(int32_t value = 0x00);

    // Calibration data methods
    CalibrationData get_rgb_calibration_data();
    CalibrationData get_ir_calibration_data();

protected:
    void initialize_supported_modes();

    // VB1940 specific helper methods
    void write_data_in_pages(uint32_t start_addr, const std::vector<uint8_t>& data);
    uint32_t get_device_id();
    void status_check();
    void do_secure_boot();
    void write_certificate();
    void write_fw();
    void write_vt_patch();

    // Register access methods
    void set_register_8(uint16_t reg, uint8_t value, const std::shared_ptr<Timeout>& timeout = nullptr);
    void set_register_16(uint16_t reg, uint16_t value, const std::shared_ptr<Timeout>& timeout = nullptr);
    void set_register_32(uint16_t reg, uint32_t value, const std::shared_ptr<Timeout>& timeout = nullptr);
    void set_register_buffer(uint16_t reg, const std::vector<uint8_t>& data_buffer, const std::shared_ptr<Timeout>& timeout = nullptr);

    void configure_camera(CameraMode mode);
    template <typename ContainerT>
    void apply_register_settings(const ContainerT& settings);

    // EEPROM access methods
    uint8_t get_eeprom_register(uint16_t reg);
    std::vector<uint8_t> get_eeprom_page(uint32_t page_num = 0, uint32_t page_offset = 0, uint32_t data_len = 64);
    void set_eeprom_register(uint16_t reg, uint8_t value, const std::shared_ptr<Timeout>& timeout = nullptr);
    void set_eeprom_page(uint32_t page_num, uint32_t page_offset, const std::vector<uint8_t>& data_buffer, const std::shared_ptr<Timeout>& timeout = nullptr);

    // Calibration parsing methods
    CalibrationData parse_calibration_data(const std::vector<double>& data);
    CalibrationData read_calibration_from_eeprom(bool rgb);

private:
    std::unordered_map<int, std::shared_ptr<Vb1940FrameFormat>> mode_frame_formats_;

    std::shared_ptr<Hololink> hololink_;
    std::shared_ptr<Hololink::I2c> i2c_;

    mutable std::mutex mutex_;
    std::shared_ptr<Synchronizer> synchronizer_;
};

} // namespace hololink::sensors

#endif /* SENSORS_CAMERA_VB1940_NATIVE_VB1940_SENSOR_HPP */
