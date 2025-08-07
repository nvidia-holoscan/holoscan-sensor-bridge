/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <hololink/operators/roce_transmitter/roce_transmitter_op.hpp>

#include "../operator_util.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

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
class PyRoceTransmitterOp : public RoceTransmitterOp {
public:
    /* Inherit the constructors */
    using RoceTransmitterOp::RoceTransmitterOp;

    // Define a constructor that fully initializes the object.
    PyRoceTransmitterOp(holoscan::Fragment* fragment, const py::args& args,
        std::string ibv_name, uint32_t ibv_port,
        std::string hololink_ip, uint32_t ibv_qp,
        uint64_t buffer_size, uint64_t queue_size,
        py::object on_start, py::object on_stop,
        const std::string& name)
        : RoceTransmitterOp(holoscan::ArgList {
            holoscan::Arg { "ibv_name", ibv_name },
            holoscan::Arg { "ibv_port", ibv_port },
            holoscan::Arg { "hololink_ip", hololink_ip },
            holoscan::Arg { "ibv_qp", ibv_qp },
            holoscan::Arg { "buffer_size", buffer_size },
            holoscan::Arg { "queue_size", queue_size },
            holoscan::Arg { "on_start", RoceTransmitterOp::OnStartCallback([on_start](const ConnectionInfo& info) {
                               if (!on_start.is_none()) {
                                   py::gil_scoped_acquire guard;
                                   on_start(info);
                               }
                           }) },
            holoscan::Arg { "on_stop", RoceTransmitterOp::OnStopCallback([on_stop](const ConnectionInfo& info) {
                               if (!on_stop.is_none()) {
                                   py::gil_scoped_acquire guard;
                                   on_stop(info);
                               }
                           }) } })
    {
        add_positional_condition_and_resource_args(this, args);
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

PYBIND11_MODULE(_roce_transmitter, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    py::class_<RoceTransmitterOp::ConnectionInfo>(m, "ConnectionInfo")
        .def_readonly("qp_num", &RoceTransmitterOp::ConnectionInfo::qp_num);

    py::class_<RoceTransmitterOp, PyRoceTransmitterOp, holoscan::Operator,
        std::shared_ptr<RoceTransmitterOp>>(m, "RoceTransmitterOp")
        .def(py::init<holoscan::Fragment*, const py::args&, const std::string&, uint32_t, const std::string&, uint32_t, uint64_t, uint64_t, py::object, py::object, const std::string&>(),
            "fragment"_a,
            "ibv_name"_a,
            "ibv_port"_a,
            "hololink_ip"_a,
            "ibv_qp"_a,
            "buffer_size"_a,
            "queue_size"_a,
            "on_start"_a = py::none(),
            "on_stop"_a = py::none(),
            "name"_a);

} // PYBIND11_MODULE

} // namespace hololink::operators
