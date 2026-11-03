// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <hololink/core/csi_formats.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace hololink::operators {

// Maps a raw-Bayer CFA phase to the matching schema::ImageT encoding -- shared by every
// Tensor->ImageT gateway operator (CsiToBayerOp, PackedFormatConverterOp) so the mapping is
// defined once instead of once per call site (it was previously duplicated verbatim across six
// example files as a local bayer_encoding_for() helper).
[[nodiscard]] inline holoscan::schema::ImageEncoding bayer_image_encoding(hololink::csi::BayerFormat format)
{
    switch (format) {
    case hololink::csi::BayerFormat::BGGR:
        return holoscan::schema::ImageEncoding_BAYER_BGGR;
    case hololink::csi::BayerFormat::RGGB:
        return holoscan::schema::ImageEncoding_BAYER_RGGB;
    case hololink::csi::BayerFormat::GBRG:
        return holoscan::schema::ImageEncoding_BAYER_GBRG;
    case hololink::csi::BayerFormat::GRBG:
        return holoscan::schema::ImageEncoding_BAYER_GRBG;
    }
    return holoscan::schema::ImageEncoding_CUSTOM;
}

// As above; without a CFA phase the encoding is CUSTOM, so a downstream demosaic uses its own
// configured grid position.
[[nodiscard]] inline holoscan::schema::ImageEncoding bayer_image_encoding(
    const std::optional<hololink::csi::BayerFormat>& format)
{
    return format ? bayer_image_encoding(*format) : holoscan::schema::ImageEncoding_CUSTOM;
}

// Copies every ImageT field through unchanged except `data`, which is reset rather than copied:
// the copy's tensor is about to be replaced by the caller's own output tensor (via emit_tensor()'s
// loan, which ignores the descriptor's `data` field entirely, or via a fresh shared_ptr assignment
// for a plain emit()), and leaving the input's shared_ptr in the copy would only extend that
// tensor's lifetime pointlessly. Mirrors forward_metadata()'s "copy through by default" idiom in
// capture_metadata_utils.hpp, for the same reason: hand-copying fields individually has already
// dropped fields silently in this codebase (see forward_metadata()'s own comment). The input's
// plane_layouts are cleared too: they describe the input's memory, and every caller writes its
// output densely (row-major, no row padding), where they would point at the wrong bytes.
[[nodiscard]] inline holoscan::schema::ImageT forward_image_descriptor(const holoscan::schema::ImageT& in)
{
    holoscan::schema::ImageT out = in;
    out.data.reset();
    out.plane_layouts.clear();
    return out;
}

} // namespace hololink::operators
