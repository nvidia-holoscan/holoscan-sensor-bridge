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

#include "camera_mode.hpp"

#include <fmt/format.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

auto fmt::formatter<hololink::sensors::CameraFrameFormat>::format(
    const hololink::sensors::CameraFrameFormat& element,
    fmt::format_context& ctx) const -> decltype(ctx.out())
{
    // Format the camera frame format with all relevant information
    std::string buffer = fmt::format(
        "CameraFrameFormat[mode={} ({}), resolution={}x{}, fps={:.2f}, format={}]",
        element.mode_id(),
        element.mode_name(),
        element.width(),
        element.height(),
        element.frame_rate(),
        static_cast<int>(element.pixel_format()));
    return fmt::format_to(ctx.out(), "{}", buffer);
}

namespace hololink::sensors {

CameraFrameFormat::CameraFrameFormat(CameraMode mode_id,
    const std::string& mode_name,
    int64_t width,
    int64_t height,
    double frame_rate,
    csi::PixelFormat pixel_format)
    : format_ { mode_id, mode_name, width, height, frame_rate, pixel_format }
{
}

const CameraFrameFormat::Format& CameraFrameFormat::format() const
{
    return format_;
}

} // namespace hololink::sensors
