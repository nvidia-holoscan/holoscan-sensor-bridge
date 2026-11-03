// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoscan/core/parameter_schema.hpp>

#include "hololink/module/csi_converter.hpp"

// Declared in its own header (rather than alongside CsiGeometry in csi_converter.hpp, or
// inside whichever operator header happens to need it first) for the same reason the native
// tree keeps hololink::csi::CsiGeometry's own schema in
// src/hololink/operators/csi_geometry_schema.hpp: csi_converter.hpp is a lightweight
// control-plane header shared by the module sensor drivers (Ar0234, Hawk, Imx274Cam,
// Vb1940Cam, ...), none of which need holoscan/core/parameter_schema.hpp's heavier transitive
// includes. Two operators need this schema (CsiToBayerOp and PackedFormatConverterOp), and a
// shared header avoids an operator-header-to-operator-header dependency between the two.
namespace hololink::module::csi {

HOLOSCAN_PARAM_SCHEMA(
    CsiGeometry,
    holoscan::param_schema("com.nvidia.hololink.module.csi.CsiGeometry", 1),
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

} // namespace hololink::module::csi
