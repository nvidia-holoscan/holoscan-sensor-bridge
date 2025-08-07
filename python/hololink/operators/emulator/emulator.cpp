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

#include <hololink/operators/emulator/linux_data_plane_op.hpp>
#include <pybind11/pybind11.h>

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace hololink::operators {

/* Trampoline classes for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the operator.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the operator's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_operator<OperatorT>
 */
class PyLinuxDataPlaneOp : public LinuxDataPlaneOp {
public:
    using LinuxDataPlaneOp::LinuxDataPlaneOp;

    // Define a constructor that fully initializes the object.
    PyLinuxDataPlaneOp(holoscan::Fragment* fragment, const py::args& args, std::shared_ptr<hololink::emulation::HSBEmulator> hsb_emulator,
        const std::string& source_ip_address, uint8_t subnet_bits, uint16_t source_port, hololink::emulation::DataPlaneID data_plane_id, hololink::emulation::SensorID sensor_id, const std::string& name)
        : LinuxDataPlaneOp(holoscan::ArgList {
            holoscan::Arg { "hsb_emulator", hsb_emulator },
            holoscan::Arg { "source_ip_address", source_ip_address },
            holoscan::Arg { "subnet_bits", subnet_bits },
            holoscan::Arg { "source_port", source_port },
            holoscan::Arg { "data_plane_id", data_plane_id },
            holoscan::Arg { "sensor_id", sensor_id } })
    {
        add_positional_condition_and_resource_args(this, args);
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

PYBIND11_MODULE(_emulator, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    py::class_<LinuxDataPlaneOp, PyLinuxDataPlaneOp, holoscan::Operator,
        std::shared_ptr<LinuxDataPlaneOp>>(m, "LinuxDataPlaneOp")
        .def(py::init<holoscan::Fragment*, const py::args&, std::shared_ptr<hololink::emulation::HSBEmulator>, const std::string&, uint8_t, uint16_t, hololink::emulation::DataPlaneID, hololink::emulation::SensorID, const std::string&>(),
            py::arg("fragment"),
            py::arg("args"),
            py::arg("hsb_emulator") = nullptr,
            py::arg("source_ip_address") = "192.168.0.2",
            py::arg("subnet_bits") = 24,
            py::arg("source_port") = 12888,
            py::arg("data_plane_id") = hololink::emulation::DataPlaneID::DATA_PLANE_0,
            py::arg("sensor_id") = hololink::emulation::SensorID::SENSOR_0,
            py::arg("name") = "");

} // PYBIND11_MODULE

} // namespace hololink::operators
