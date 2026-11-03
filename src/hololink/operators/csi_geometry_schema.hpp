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

#pragma once

#include <hololink/core/csi_controller.hpp>
#include <holoscan/core/parameter_schema.hpp>

// CsiGeometry's parameter schema, shared by CsiToBayerOp and PackedFormatConverterOp. It has a
// header of its own rather than living alongside CsiGeometry in csi_controller.hpp, so that the
// lightweight control-plane header doesn't pull in holoscan/core/parameter_schema.hpp's heavier
// transitive includes (down to holoscan/core/domain/tensor.hpp's <cuda/stream>) for every
// consumer of csi_controller.hpp, such as hololink::Hololink's control-plane implementation.
namespace hololink::csi {

HOLOSCAN_PARAM_SCHEMA(
    CsiGeometry,
    holoscan::param_schema("com.nvidia.hololink.csi.CsiGeometry", 1),
    holoscan::field<&CsiGeometry::start_byte>("start_byte")
        .description("Offset of the first CSI payload byte within the received buffer."),
    holoscan::field<&CsiGeometry::bytes_per_line>("bytes_per_line")
        .description("Stride of one CSI line, in bytes."),
    holoscan::field<&CsiGeometry::pixel_width>("pixel_width")
        .description("Frame width in pixels."),
    holoscan::field<&CsiGeometry::pixel_height>("pixel_height")
        .description("Frame height in pixels."),
    holoscan::field<&CsiGeometry::pixel_format>("pixel_format")
        .enum_values({ holoscan::param_enum_value("raw_8", PixelFormat::RAW_8),
            holoscan::param_enum_value("raw_10", PixelFormat::RAW_10),
            holoscan::param_enum_value("raw_12", PixelFormat::RAW_12) })
        .description("CSI pixel packing."),
    holoscan::field<&CsiGeometry::trailing_bytes>("trailing_bytes")
        .description("Trailing status bytes appended after the last CSI line."));

} // namespace hololink::csi
