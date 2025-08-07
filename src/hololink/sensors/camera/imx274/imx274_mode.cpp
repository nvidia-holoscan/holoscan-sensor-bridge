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

#include "imx274_mode.hpp"

namespace hololink::sensors {

Imx274FrameFormat::Imx274FrameFormat(CameraMode mode_id, const std::string& mode_name,
    int64_t width, int64_t height, double frame_rate,
    csi::PixelFormat pixel_format)
    : CameraFrameFormat(mode_id, mode_name, width, height, frame_rate, pixel_format)
{
}

const Imx274FrameFormat::Format& Imx274FrameFormat::format() const
{
    return format_;
}

} // namespace hololink::sensors
