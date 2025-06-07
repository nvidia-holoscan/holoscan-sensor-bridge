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

#include <hololink/core/csi_formats.hpp>
#include <hololink/core/logging.hpp>
#include <hololink/sensors/li_i2c_expander.hpp>
#include <hololink/sensors/sensor.hpp>

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>
#include <type_traits>
#include <variant>

using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace hololink::sensors {

// Trampoline class for Sensor
class PySensor : public Sensor {
public:
    using Sensor::Sensor;

    void start() override { PYBIND11_OVERLOAD_PURE(void, Sensor, start, ); }

    void stop() override { PYBIND11_OVERLOAD_PURE(void, Sensor, stop, ); }
};

template <typename T, typename... ArgsT>
inline constexpr bool is_one_of_v = ((std::is_same_v<T, ArgsT> || ...));

PYBIND11_MODULE(_hololink_sensor, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    // Bind Sensor class
    py::class_<Sensor, PySensor, std::shared_ptr<Sensor>>(m, "Sensor")
        .def(py::init<>())
        .def("start", &Sensor::start)
        .def("stop", &Sensor::stop);

    // Bind I2CExpanderOutputEN enum
    py::enum_<I2CExpanderOutputEN>(m, "I2CExpanderOutputEN")
        .value("OUTPUT_1", I2CExpanderOutputEN::OUTPUT_1)
        .value("OUTPUT_2", I2CExpanderOutputEN::OUTPUT_2)
        .value("OUTPUT_3", I2CExpanderOutputEN::OUTPUT_3)
        .value("OUTPUT_4", I2CExpanderOutputEN::OUTPUT_4)
        .value("DEFAULT", I2CExpanderOutputEN::DEFAULT);

    // Bind LII2CExpander class
    py::class_<LII2CExpander, std::shared_ptr<LII2CExpander>>(m, "LII2CExpander")
        .def(py::init<Hololink&, uint32_t>())
        .def("configure", static_cast<void (LII2CExpander::*)(I2CExpanderOutputEN)>(&LII2CExpander::configure),
            py::arg("output_en") = I2CExpanderOutputEN::DEFAULT)
        .def_property_readonly_static("I2C_EXPANDER_ADDRESS",
            [](py::object) { return LII2CExpander::I2C_EXPANDER_ADDRESS; });

} // PYBIND11_MODULE

} // namespace hololink::sensors
