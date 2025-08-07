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

#include "camera_sensor.hpp"

namespace hololink::sensors {

CameraSensor::CameraSensor() = default;
CameraSensor::~CameraSensor() = default;

void CameraSensor::start()
{
}

void CameraSensor::stop()
{
}

void CameraSensor::configure(CameraMode mode) { }

CameraMode CameraSensor::get_mode() const
{
    if (!mode_) {
        return -1;
    }
    return mode_.value();
}

void CameraSensor::set_mode(CameraMode mode)
{
    mode_ = mode;
}

const std::unordered_set<CameraMode>& CameraSensor::supported_modes() const
{
    return supported_modes_;
}

void CameraSensor::configure_converter(std::shared_ptr<hololink::csi::CsiConverter> converter)
{
    throw std::runtime_error("configure_converter is not implemented for CameraSensor");
}

int64_t CameraSensor::get_width() const { return width_; }

int64_t CameraSensor::get_height() const { return height_; }

hololink::csi::PixelFormat CameraSensor::get_pixel_format() const { return pixel_format_; }

hololink::csi::BayerFormat CameraSensor::get_bayer_format() const { return bayer_format_; }

} // namespace hololink::sensors
