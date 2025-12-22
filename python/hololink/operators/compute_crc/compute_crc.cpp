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

#include <hololink/operators/compute_crc/compute_crc.hpp>

#include <cuda.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for unordered_map -> dict, etc.

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>

#include <hololink/core/logging_internal.hpp>

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace hololink::operators {

class PyComputeCrcOp : public ComputeCrcOp {
public:
    /* Inherit the constructors */
    using ComputeCrcOp::ComputeCrcOp;

    // Define a constructor that fully initializes the object.
    PyComputeCrcOp(holoscan::Fragment* fragment,
        int cuda_device_ordinal,
        const std::string& name = "compute_crc",
        uint64_t frame_size = 0)
        : ComputeCrcOp(holoscan::ArgList {
            holoscan::Arg { "cuda_device_ordinal", cuda_device_ordinal },
            holoscan::Arg { "frame_size", frame_size } })
    {
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

class PyCheckCrcOp : public CheckCrcOp {
public:
    /* Inherit the constructors */
    using CheckCrcOp::CheckCrcOp;

    // Define a constructor that fully initializes the object.
    PyCheckCrcOp(holoscan::Fragment* fragment,
        std::shared_ptr<ComputeCrcOp> compute_crc_op,
        const std::string& name = "check_crc",
        const std::string& computed_crc_metadata_name = "computed_crc")
        : CheckCrcOp(holoscan::ArgList {
            holoscan::Arg { "compute_crc_op", compute_crc_op },
            holoscan::Arg { "computed_crc_metadata_name", computed_crc_metadata_name } })
    {
        name_ = name;
        fragment_ = fragment;
        compute_crc_op_ = compute_crc_op;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());
    }

    void check_crc(uint32_t computed_crc) override
    {
        PYBIND11_OVERRIDE(
            void, /* Return type */
            CheckCrcOp, /* Parent class */
            check_crc, /* Name of function in C++ (must match Python name) */
            computed_crc /* Argument(s) */
        );
    }
};

PYBIND11_MODULE(_compute_crc, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    py::class_<ComputeCrcOp, PyComputeCrcOp, holoscan::Operator,
        std::shared_ptr<ComputeCrcOp>>(m, "ComputeCrcOp")
        .def(py::init<holoscan::Fragment*, int, const std::string&, uint64_t>(),
            "fragment"_a, "cuda_device_ordinal"_a = 0,
            "name"_a = "compute_crc"s, "frame_size"_a = 0)
        .def("setup", &ComputeCrcOp::setup, "spec"_a);

    py::class_<CheckCrcOp, PyCheckCrcOp, holoscan::Operator,
        std::shared_ptr<CheckCrcOp>>(m, "CheckCrcOp")
        .def(py::init<holoscan::Fragment*,
                 std::shared_ptr<ComputeCrcOp>,
                 const std::string&,
                 const std::string&>(),
            "fragment"_a,
            "compute_crc_op"_a,
            "name"_a,
            "computed_crc_metadata_name"_a = "computed_crc"s)
        .def("setup", &CheckCrcOp::setup, "spec"_a);
} // PYBIND11_MODULE

} // namespace hololink::operators
