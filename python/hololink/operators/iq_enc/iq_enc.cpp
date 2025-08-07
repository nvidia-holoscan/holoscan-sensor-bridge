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

#include <hololink/operators/iq_enc/iq_enc_op.hpp>

#include "../operator_util.hpp"

#include <pybind11/pybind11.h>

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
class PyIQEncoderOp : public IQEncoderOp {
public:
    /* Inherit the constructors */
    using IQEncoderOp::IQEncoderOp;

    // Define a constructor that fully initializes the object.
    PyIQEncoderOp(holoscan::Fragment* fragment, const py::args& args, py::object renderer,
        float scale, const std::string& name)
        : IQEncoderOp(holoscan::ArgList {
            holoscan::Arg { "renderer", py::cast<ImGuiRenderer*>(renderer) },
            holoscan::Arg { "scale", scale } })
    {
        add_positional_condition_and_resource_args(this, args);
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

PYBIND11_MODULE(_iq_enc, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    py::class_<IQEncoderOp, PyIQEncoderOp, holoscan::Operator,
        std::shared_ptr<IQEncoderOp>>(m, "IQEncoderOp")
        .def(py::init<holoscan::Fragment*, const py::args&, py::object, float, const std::string&>(),
            "fragment"_a,
            "renderer"_a = nullptr,
            "scale"_a = 1,
            "name"_a = "iq_encoder"s);

} // PYBIND11_MODULE

} // namespace hololink::operators
