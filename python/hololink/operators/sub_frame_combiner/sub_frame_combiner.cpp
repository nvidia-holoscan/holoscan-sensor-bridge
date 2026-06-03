/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <hololink/operators/sub_frame_combiner/sub_frame_combiner.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for unordered_map -> dict, etc.

#include <cstdint>
#include <memory>
#include <string>

#include "../operator_util.hpp"

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
class PySubFrameCombinerOp : public SubFrameCombinerOp {
public:
    /* Inherit the constructors */
    using SubFrameCombinerOp::SubFrameCombinerOp;

    // Define a constructor that fully initializes the object.
    PySubFrameCombinerOp(holoscan::Fragment* fragment, const py::args& args, const std::shared_ptr<holoscan::Allocator>& allocator,
        const std::string& name = "sub_frame_combiner", int cuda_device_ordinal = 0,
        const std::string& out_tensor_name = "", uint32_t height = 0)
        : SubFrameCombinerOp(holoscan::ArgList { holoscan::Arg { "allocator", allocator },
            holoscan::Arg { "cuda_device_ordinal", cuda_device_ordinal },
            holoscan::Arg { "out_tensor_name", out_tensor_name },
            holoscan::Arg { "height", height } })
    {
        add_positional_condition_and_resource_args(this, args);
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

PYBIND11_MODULE(_sub_frame_combiner, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    auto op = py::class_<SubFrameCombinerOp, PySubFrameCombinerOp, holoscan::Operator, std::shared_ptr<SubFrameCombinerOp>>(m,
        "SubFrameCombinerOp")
                  .def(py::init<holoscan::Fragment*, const py::args&, const std::shared_ptr<holoscan::Allocator>&, const std::string&,
                           int, const std::string&, uint32_t>(),
                      "fragment"_a,
                      "allocator"_a,
                      "name"_a = "sub_frame_combiner"s,
                      "cuda_device_ordinal"_a = 0,
                      "out_tensor_name"_a = ""s,
                      "height"_a = 0U)
                  .def("setup", &SubFrameCombinerOp::setup, "spec"_a);
} // PYBIND11_MODULE

} // namespace hololink::operators
