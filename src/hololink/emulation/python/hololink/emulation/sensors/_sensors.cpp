/**
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

#include "hololink/emulation/data_plane.hpp"
#include "hololink/emulation/hsb_emulator.hpp"
#include "hololink/emulation/i2c_interface.hpp"
#include "hololink/emulation/sensors/vb1940_emulator.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace hololink::emulation {

PYBIND11_MODULE(_emulation_sensors, m)
{

    py::class_<sensors::Vb1940Emulator, I2CPeripheral>(m, "Vb1940Emulator")
        .def(py::init<>())
        .def("reset", &sensors::Vb1940Emulator::reset, "reset the Vb1940Emulator")
        .def("attach_to_i2c", &sensors::Vb1940Emulator::attach_to_i2c, "attach the Vb1940Emulator to an I2C controller")
        .def("i2c_transaction", &sensors::Vb1940Emulator::i2c_transaction, "perform an I2C transaction")
        .def("is_streaming", &sensors::Vb1940Emulator::is_streaming, "check if Vb1940Emulator is streaming")
        .def("get_pixel_width", &sensors::Vb1940Emulator::get_pixel_width, "get the pixel width")
        .def("get_pixel_height", &sensors::Vb1940Emulator::get_pixel_height, "get the pixel height")
        .def("get_bytes_per_line", &sensors::Vb1940Emulator::get_bytes_per_line, "get the bytes per line")
        .def("get_image_start_byte", &sensors::Vb1940Emulator::get_image_start_byte, "get the image start byte")
        .def("get_pixel_bits", &sensors::Vb1940Emulator::get_pixel_bits, "get the pixel bits")
        .def("get_csi_length", &sensors::Vb1940Emulator::get_csi_length, "get the CSI length");

} // PYBIND11_MODULE

} // namespace hololink::emulation
