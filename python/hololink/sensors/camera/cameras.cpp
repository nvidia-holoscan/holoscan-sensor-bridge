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

#include <string>

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <hololink/core/logging.hpp>
#include <hololink/sensors/camera/camera_mode.hpp>
#include <hololink/sensors/camera/camera_sensor.hpp>

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace hololink::sensors {

// Trampoline class for CameraSensor
class PyCameraSensor : public CameraSensor {
public:
    using CameraSensor::CameraSensor; // Inherit constructors

    // Only override the virtual methods
    void configure(CameraMode mode) override
    {
        PYBIND11_OVERRIDE(void, CameraSensor, configure, mode);
    }

    CameraMode get_mode() const override
    {
        PYBIND11_OVERRIDE(CameraMode, CameraSensor, get_mode, );
    }

    void set_mode(CameraMode mode) override
    {
        PYBIND11_OVERRIDE(void, CameraSensor, set_mode, mode);
    }

    void start() override
    {
        PYBIND11_OVERRIDE(void, CameraSensor, start, );
    }

    void stop() override
    {
        PYBIND11_OVERRIDE(void, CameraSensor, stop, );
    }

    const std::unordered_set<CameraMode>& supported_modes() const override
    {
        PYBIND11_OVERRIDE(const std::unordered_set<CameraMode>&,
            CameraSensor,
            supported_modes, );
    }

    void configure_converter(std::shared_ptr<hololink::csi::CsiConverter> converter) override
    {
        PYBIND11_OVERRIDE(void, CameraSensor, configure_converter, converter);
    }

    int64_t get_width() const override
    {
        PYBIND11_OVERRIDE(int64_t, CameraSensor, get_width, );
    }

    int64_t get_height() const override
    {
        PYBIND11_OVERRIDE(int64_t, CameraSensor, get_height, );
    }

    csi::PixelFormat get_pixel_format() const override
    {
        PYBIND11_OVERRIDE(csi::PixelFormat, CameraSensor, get_pixel_format, );
    }

    csi::BayerFormat get_bayer_format() const override
    {
        PYBIND11_OVERRIDE(csi::BayerFormat, CameraSensor, get_bayer_format, );
    }
};

PYBIND11_MODULE(_hololink_camera_sensor, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    // Bind the Format struct
    py::class_<CameraFrameFormat::Format>(m, "CameraFrameFormat_Format")
        .def(py::init<CameraMode, const std::string&, int64_t, int64_t, double, csi::PixelFormat>(),
            py::arg("mode_id"), py::arg("mode_name"), py::arg("width"), py::arg("height"),
            py::arg("frame_rate"), py::arg("pixel_format"))
        .def_property_readonly("mode_id", [](const CameraFrameFormat::Format& f) { return f.mode_id; })
        .def_property_readonly("mode_name", [](const CameraFrameFormat::Format& f) { return f.mode_name; })
        .def_property_readonly("width", [](const CameraFrameFormat::Format& f) { return f.width; })
        .def_property_readonly("height", [](const CameraFrameFormat::Format& f) { return f.height; })
        .def_property_readonly("frame_rate", [](const CameraFrameFormat::Format& f) { return f.frame_rate; })
        .def_property_readonly("pixel_format", [](const CameraFrameFormat::Format& f) { return f.pixel_format; });

    // Bind CameraFrameFormat class
    py::class_<CameraFrameFormat, std::shared_ptr<CameraFrameFormat>>(m, "CameraFrameFormat")
        .def(py::init<CameraMode, const std::string&, int64_t, int64_t, double, csi::PixelFormat>(),
            py::arg("mode_id"),
            py::arg("mode_name"),
            py::arg("width"),
            py::arg("height"),
            py::arg("frame_rate"),
            py::arg("pixel_format"))
        .def_property_readonly("format", &CameraFrameFormat::format)
        .def_property_readonly("mode_id", &CameraFrameFormat::mode_id)
        .def_property_readonly("mode_name", &CameraFrameFormat::mode_name)
        .def_property_readonly("width", &CameraFrameFormat::width)
        .def_property_readonly("height", &CameraFrameFormat::height)
        .def_property_readonly("frame_rate", &CameraFrameFormat::frame_rate)
        .def_property_readonly("pixel_format", &CameraFrameFormat::pixel_format);

    // Bind CameraSensor class
    py::class_<CameraSensor, PyCameraSensor, Sensor, std::shared_ptr<CameraSensor>>(m, "CameraSensor")
        .def(py::init<>())
        .def("configure", &CameraSensor::configure)
        .def("set_mode", &CameraSensor::set_mode)
        .def("start", &CameraSensor::start)
        .def("stop", &CameraSensor::stop)
        .def("supported_modes", [](const CameraSensor& self) {
            return py::cast(self.supported_modes());
        })
        .def_property_readonly("mode", &CameraSensor::get_mode)
        .def_property_readonly("width", &CameraSensor::get_width)
        .def_property_readonly("height", &CameraSensor::get_height)
        .def_property_readonly("pixel_format", &CameraSensor::get_pixel_format)
        .def_property_readonly("bayer_format", &CameraSensor::get_bayer_format)
        .def("configure_converter", &CameraSensor::configure_converter);
}

} // namespace hololink::sensors
