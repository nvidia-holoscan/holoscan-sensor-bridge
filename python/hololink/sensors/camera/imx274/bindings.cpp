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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <hololink/sensors/camera/imx274/imx274_mode.hpp>
#include <hololink/sensors/camera/imx274/native_imx274_sensor.hpp>

namespace py = pybind11;

namespace hololink::sensors {

PYBIND11_MODULE(_native_imx274_camera_sensor, m)
{
    // Bind NativeImx274Sensor class
    py::class_<NativeImx274Sensor, CameraSensor, std::shared_ptr<NativeImx274Sensor>>(m, "NativeImx274Sensor")
        .def(py::init<DataChannel&, int, uint32_t>(),
            py::arg("data_channel"),
            py::arg("expander_configuration") = 0,
            py::arg("i2c_bus") = CAM_I2C_BUS)
        .def("configure", py::overload_cast<CameraMode>(&NativeImx274Sensor::configure))
        .def("start", &NativeImx274Sensor::start)
        .def("stop", &NativeImx274Sensor::stop)
        .def("setup_clock", &NativeImx274Sensor::setup_clock)
        .def("get_version", &NativeImx274Sensor::get_version)
        .def("test_pattern", &NativeImx274Sensor::test_pattern)
        .def("set_exposure_reg", &NativeImx274Sensor::set_exposure_reg)
        .def("set_analog_gain_reg", &NativeImx274Sensor::set_analog_gain_reg)
        .def("set_digital_gain_reg", &NativeImx274Sensor::set_digital_gain_reg)
        .def_property_readonly_static("DRIVER_NAME",
            [](py::object) { return NativeImx274Sensor::DRIVER_NAME; })
        .def_property_readonly_static("VERSION",
            [](py::object) { return NativeImx274Sensor::VERSION; })
        .def_property_readonly_static("I2C_ADDRESS",
            [](py::object) { return NativeImx274Sensor::I2C_ADDRESS; });

    // Bind IMX274 mode enum
    py::module_ imx274_mode_module = m.def_submodule("imx274_mode");
    // Note: pybind11's generated enum type is not PEP 435 compatible (not iterable)
    // See https://github.com/pybind/pybind11/issues/2332
    py::enum_<imx274_mode::Mode>(imx274_mode_module, "Imx274_Mode")
        .value("IMX274_MODE_3840X2160_60FPS", imx274_mode::IMX274_MODE_3840X2160_60FPS)
        .value("IMX274_MODE_1920X1080_60FPS", imx274_mode::IMX274_MODE_1920X1080_60FPS)
        .value("IMX274_MODE_3840X2160_60FPS_12BITS", imx274_mode::IMX274_MODE_3840X2160_60FPS_12BITS)
        .export_values();

    imx274_mode_module.attr("IMX274_TABLE_WAIT_MS") = imx274_mode::IMX274_TABLE_WAIT_MS;
    imx274_mode_module.attr("IMX274_WAIT_MS") = imx274_mode::IMX274_WAIT_MS;
    imx274_mode_module.attr("IMX274_WAIT_MS_START") = imx274_mode::IMX274_WAIT_MS_START;
    imx274_mode_module.attr("REG_EXP_MSB") = imx274_mode::REG_EXP_MSB;
    imx274_mode_module.attr("REG_EXP_LSB") = imx274_mode::REG_EXP_LSB;
    imx274_mode_module.attr("REG_AG_MSB") = imx274_mode::REG_AG_MSB;
    imx274_mode_module.attr("REG_AG_LSB") = imx274_mode::REG_AG_LSB;
    imx274_mode_module.attr("REG_DG") = imx274_mode::REG_DG;

    // Bind Imx274FrameFormat class
    py::class_<Imx274FrameFormat, CameraFrameFormat, std::shared_ptr<Imx274FrameFormat>>(m, "Imx274FrameFormat")
        .def(py::init<CameraMode, const std::string&, uint32_t, uint32_t, uint32_t, csi::PixelFormat>())
        .def_property_readonly("mode_id", &Imx274FrameFormat::mode_id)
        .def_property_readonly("mode_name", &Imx274FrameFormat::mode_name)
        .def_property_readonly("width", &Imx274FrameFormat::width)
        .def_property_readonly("height", &Imx274FrameFormat::height)
        .def_property_readonly("frame_rate", &Imx274FrameFormat::frame_rate)
        .def_property_readonly("pixel_format", &Imx274FrameFormat::pixel_format);
}

} // namespace hololink::sensors
