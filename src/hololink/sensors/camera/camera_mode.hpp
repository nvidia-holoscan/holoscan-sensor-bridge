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

#ifndef SENSORS_CAMERA_CAMERA_MODE_HPP
#define SENSORS_CAMERA_CAMERA_MODE_HPP

#include <cstdint>
#include <string>

#include <fmt/format.h>
#include <fmt/ranges.h> // allows fmt to format std::array, std::vector, etc.

#include <hololink/core/csi_formats.hpp>

namespace hololink::sensors {

using CameraMode = int;

class CameraFrameFormat {
public:
    struct Format {
        CameraMode mode_id;
        std::string mode_name;
        int64_t width;
        int64_t height;
        double frame_rate;
        csi::PixelFormat pixel_format;
    };

    CameraFrameFormat(CameraMode mode_id,
        const std::string& mode_name,
        int64_t width,
        int64_t height,
        double frame_rate,
        csi::PixelFormat pixel_format);

    virtual ~CameraFrameFormat() = default;

    // Core properties as simple getters
    virtual const Format& format() const;

    // Convenience accessors
    CameraMode mode_id() const { return format().mode_id; }
    const std::string& mode_name() const { return format().mode_name; }
    int64_t width() const { return format().width; }
    int64_t height() const { return format().height; }
    double frame_rate() const { return format().frame_rate; }
    hololink::csi::PixelFormat pixel_format() const { return format().pixel_format; }

protected:
    Format format_;
};

} // namespace hololink::sensors

/**
 * @brief Formatter for CameraMode
 */
template <>
struct fmt::formatter<hololink::sensors::CameraFrameFormat> : fmt::formatter<fmt::string_view> {
    /**
     * @brief Format function for CameraFrameFormat
     *
     * @param element
     * @param ctx
     * @return auto
     */
    auto format(const hololink::sensors::CameraFrameFormat& element,
        fmt::format_context& ctx) const -> decltype(ctx.out());
};

#endif /* SENSORS_CAMERA_CAMERA_MODE_HPP */
