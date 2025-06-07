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

#include <hololink/operators/sig_gen/sig_gen_op.hpp>

#include "../operator_util.hpp"

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include <hololink/common/gui_renderer.hpp>

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
class PySignalGeneratorOp : public SignalGeneratorOp {
public:
    /* Inherit the constructors */
    using SignalGeneratorOp::SignalGeneratorOp;

    // Define a constructor that fully initializes the object.
    PySignalGeneratorOp(holoscan::Fragment* fragment, const py::args& args, py::object renderer,
        unsigned samples_count, py::object sampling_interval, const std::string& in_phase,
        const std::string& quadrature, const std::string& cuda_toolkit_include_path, const std::string& name)
        : SignalGeneratorOp(holoscan::ArgList {
            holoscan::Arg { "renderer", py::cast<ImGuiRenderer*>(renderer) },
            holoscan::Arg { "samples_count", samples_count },
            holoscan::Arg { "sampling_interval", py::cast<Rational>(sampling_interval) },
            holoscan::Arg { "in_phase", in_phase },
            holoscan::Arg { "quadrature", quadrature },
            holoscan::Arg { "cuda_toolkit_include_path", cuda_toolkit_include_path } })
    {
        add_positional_condition_and_resource_args(this, args);
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

PYBIND11_MODULE(_sig_gen, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
    py::class_<Rational>(m, "Rational")
        .def(py::init<>())
        .def(py::init<std::string>())
        .def_readwrite("num", &Rational::num_)
        .def_readwrite("den", &Rational::den_)
        .def("__str__", [](const Rational& rational) {
            std::stringstream ss;
            ss << rational;
            return ss.str();
        });
    py::class_<SignalGeneratorOp, PySignalGeneratorOp, holoscan::Operator,
        std::shared_ptr<SignalGeneratorOp>>(m, "SignalGeneratorOp")
        .def(py::init<holoscan::Fragment*, const py::args&, py::object, int32_t, py::object, const std::string&, const std::string&, const std::string&, const std::string&>(),
            "fragment"_a,
            "renderer"_a = nullptr,
            "samples_count"_a,
            "sampling_interval"_a,
            "in_phase"_a,
            "quadrature"_a,
            "cuda_toolkit_include_path"_a = CUDA_TOOLKIT_INCLUDE_PATH,
            "name"_a);

} // PYBIND11_MODULE

} // namespace hololink::operators
