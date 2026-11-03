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

namespace {

    // Adapter: captures the geometry that the camera driver computes via the CsiConverter
    // protocol, using CsiToBayerOp-compatible alignment rules. File-scope (not a
    // configure_converter()-local struct) so make_generic_geometry_adapter() below can hand one
    // out to callers with no CameraSensor instance to drive it through.
    class GeometryCaptureConverter : public hololink::csi::CsiConverter {
    public:
        hololink::csi::CsiGeometry& out;
        explicit GeometryCaptureConverter(hololink::csi::CsiGeometry& g)
            : out(g)
        {
        }

        uint32_t receiver_start_byte() override { return 0; }

        uint32_t received_line_bytes(uint32_t n) override { return (n + 7u) & ~7u; }

        uint32_t transmitted_line_bytes(hololink::csi::PixelFormat fmt, uint32_t w) override
        {
            switch (fmt) {
            case hololink::csi::PixelFormat::RAW_8:
                return w;
            case hololink::csi::PixelFormat::RAW_10:
                return w * 5 / 4;
            case hololink::csi::PixelFormat::RAW_12:
                return w * 3 / 2;
            default:
                return w;
            }
        }

        void configure(uint32_t sb, uint32_t bpl, uint32_t pw, uint32_t ph,
            hololink::csi::PixelFormat fmt, uint32_t trailing = 0) override
        {
            out = { sb, bpl, pw, ph, fmt, trailing };
        }
    };

} // namespace

std::shared_ptr<hololink::csi::CsiConverter> CameraSensor::make_generic_geometry_adapter(
    hololink::csi::CsiGeometry& out)
{
    return std::make_shared<GeometryCaptureConverter>(out);
}

void CameraSensor::configure_converter(hololink::csi::CsiGeometry& geom)
{
    auto adapter = make_generic_geometry_adapter(geom);
    configure_converter(std::static_pointer_cast<hololink::csi::CsiConverter>(adapter));
}

int64_t CameraSensor::get_width() const { return width_; }

int64_t CameraSensor::get_height() const { return height_; }

hololink::csi::PixelFormat CameraSensor::get_pixel_format() const { return pixel_format_; }

hololink::csi::BayerFormat CameraSensor::get_bayer_format() const { return bayer_format_; }

} // namespace hololink::sensors
