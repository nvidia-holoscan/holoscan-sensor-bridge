/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>

#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/vsync.hpp"

#include "hololink/module/csi_converter.hpp"

#include "hololink/module/sensors/vb1940/vb1940_cam.hpp"

namespace py = pybind11;
namespace v1940 = hololink::module::sensors::vb1940;
namespace ad = hololink::module;

PYBIND11_MODULE(_hololink_module_vb1940, m)
{
    m.doc() = "hololink_module module-side VB1940 camera bindings";

    // The core extension registers the V1 base types the camera ctor
    // and configure() depend on (EnumerationMetadata, Vsync, …).
    // Importing it here ensures those registrations are present before
    // pybind tries to resolve cross-module references.
    py::module_::import("hololink_module._hololink_py_module");

    // Per-sensor CSI format enums. Integer values match the legacy
    // hololink::csi::* enums byte-for-byte, so passing
    // camera.bayer_format().value into legacy operators
    // (BayerDemosaicOp, ImageProcessorOp) works without conversion.
    py::enum_<ad::csi::PixelFormat>(m, "PixelFormat")
        .value("RAW_8", ad::csi::PixelFormat::RAW_8)
        .value("RAW_10", ad::csi::PixelFormat::RAW_10)
        .value("RAW_12", ad::csi::PixelFormat::RAW_12);

    py::enum_<ad::csi::BayerFormat>(m, "BayerFormat")
        .value("BGGR", ad::csi::BayerFormat::BGGR)
        .value("RGGB", ad::csi::BayerFormat::RGGB)
        .value("GBRG", ad::csi::BayerFormat::GBRG)
        .value("GRBG", ad::csi::BayerFormat::GRBG);

    py::enum_<v1940::Vb1940_Mode>(m, "Vb1940_Mode")
        .value("VB1940_MODE_2560X1984_30FPS",
            v1940::Vb1940_Mode::VB1940_MODE_2560X1984_30FPS)
        .value("VB1940_MODE_1920X1080_30FPS",
            v1940::Vb1940_Mode::VB1940_MODE_1920X1080_30FPS)
        .value("VB1940_MODE_2560X1984_30FPS_8BIT",
            v1940::Vb1940_Mode::VB1940_MODE_2560X1984_30FPS_8BIT)
        .value("VB1940_MODE_2560X1984_60FPS",
            v1940::Vb1940_Mode::VB1940_MODE_2560X1984_60FPS);

    py::class_<v1940::Vb1940Cam, std::shared_ptr<v1940::Vb1940Cam>>(
        m, "Vb1940Cam")
        .def(py::init([](const ad::EnumerationMetadata& metadata,
                          std::shared_ptr<ad::VsyncInterfaceV1> vsync) {
            return std::make_shared<v1940::Vb1940Cam>(metadata, std::move(vsync));
        }),
            py::arg("metadata"), py::arg("vsync") = nullptr)
        .def("configure", &v1940::Vb1940Cam::configure, py::arg("mode"))
        .def("start", &v1940::Vb1940Cam::start)
        .def("stop", &v1940::Vb1940Cam::stop)
        .def("set_exposure_reg", &v1940::Vb1940Cam::set_exposure_reg,
            py::arg("value") = 0x0014)
        .def("set_analog_gain_reg", &v1940::Vb1940Cam::set_analog_gain_reg,
            py::arg("value") = 0x00)
        .def("width", &v1940::Vb1940Cam::width)
        .def("height", &v1940::Vb1940Cam::height)
        .def("pixel_format", &v1940::Vb1940Cam::pixel_format)
        .def("bayer_format", &v1940::Vb1940Cam::bayer_format)
        // configure_converter takes the module-native CsiConverterV1.
        // No in-tree Python player drives the module VB1940 binding, so
        // this is exposed for completeness; a Python caller would pass a
        // CsiConverterV1 implementation (e.g. a bridge wrapping a legacy
        // CsiToBayerOp).
        .def("configure_converter", &v1940::Vb1940Cam::configure_converter,
            py::arg("converter"));
}
