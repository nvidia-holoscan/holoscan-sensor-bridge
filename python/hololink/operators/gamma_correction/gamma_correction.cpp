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

#include <hololink/operators/gamma_correction/gamma_correction.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for unordered_map -> dict, etc.

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>

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
class PyGammaCorrectionOp : public GammaCorrectionOp {
public:
    /* Inherit the constructors */
    using GammaCorrectionOp::GammaCorrectionOp;

    // Define a constructor that fully initializes the object.
    PyGammaCorrectionOp(holoscan::Fragment* fragment, float gamma, int cuda_device_ordinal, const std::string& name = "gamma_correction")
        : GammaCorrectionOp(holoscan::ArgList { holoscan::Arg { "gamma", gamma }, holoscan::Arg { "cuda_device_ordinal", cuda_device_ordinal } })
    {
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

PYBIND11_MODULE(_gamma_correction, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    py::class_<GammaCorrectionOp, PyGammaCorrectionOp, holoscan::Operator, std::shared_ptr<GammaCorrectionOp>>(m, "GammaCorrectionOp")
        .def(py::init<holoscan::Fragment*, float, int, const std::string&>(),
            "fragment"_a,
            "gamma"_a = 2.2f,
            "cuda_device_ordinal"_a = 0,
            "name"_a = "gamma_correction"s)
        .def("setup", &GammaCorrectionOp::setup, "spec"_a);

} // PYBIND11_MODULE

} // namespace hololink::operators
