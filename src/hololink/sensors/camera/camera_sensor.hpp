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

#ifndef SENSORS_CAMERA_CAMERA_SENSOR_HPP
#define SENSORS_CAMERA_CAMERA_SENSOR_HPP

#include "../sensor.hpp"
#include "camera_mode.hpp"

#include <hololink/core/csi_controller.hpp>

#include <memory>
#include <optional>
#include <unordered_set>
#include <vector>

namespace hololink::sensors {

class CameraSensor : public Sensor {
public:
    CameraSensor();
    virtual ~CameraSensor();

    // Start/stop data streaming from the camera sensor
    void start() override;
    void stop() override;

    // Configure the camera sensor settings
    virtual void configure(CameraMode mode);

    // Get the current camera mode
    virtual CameraMode get_mode() const;

    // Set the camera sensor mode
    virtual void set_mode(CameraMode mode);

    // Get supported modes
    virtual const std::unordered_set<CameraMode>& supported_modes() const;

    virtual void configure_converter(std::shared_ptr<hololink::csi::CsiConverter> converter);

    virtual int64_t get_width() const;
    virtual int64_t get_height() const;
    virtual hololink::csi::PixelFormat get_pixel_format() const;
    virtual hololink::csi::BayerFormat get_bayer_format() const;

protected:
    std::unordered_set<CameraMode> supported_modes_;
    // Current camera mode
    std::optional<CameraMode> mode_;

    int64_t width_ = 0;
    int64_t height_ = 0;
    hololink::csi::PixelFormat pixel_format_;
    hololink::csi::BayerFormat bayer_format_;
};

} // namespace hololink::sensors

#endif /* SENSORS_CAMERA_CAMERA_SENSOR_HPP */
