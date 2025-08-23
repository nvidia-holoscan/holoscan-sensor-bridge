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

#include <hololink/operators/packed_format_converter/packed_format_converter.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for unordered_map -> dict, etc.

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>

#include <hololink/core/data_channel.hpp>

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
class PyPackedFormatConverterOp : public PackedFormatConverterOp {
public:
    /* Inherit the constructors */
    using PackedFormatConverterOp::PackedFormatConverterOp;

    // Define a constructor that fully initializes the object.
    PyPackedFormatConverterOp(holoscan::Fragment* fragment,
        const std::shared_ptr<holoscan::Allocator>& allocator,
        py::object hololink_channel,
        int cuda_device_ordinal,
        const std::string& name = "packed_format_converter",
        const std::string& in_tensor_name = "",
        const std::string& out_tensor_name = "")
        : PackedFormatConverterOp(holoscan::ArgList { holoscan::Arg { "allocator", allocator },
            holoscan::Arg { "cuda_device_ordinal", cuda_device_ordinal },
            holoscan::Arg { "hololink_channel", py::cast<DataChannel*>(hololink_channel) },
            holoscan::Arg { "in_tensor_name", in_tensor_name },
            holoscan::Arg { "out_tensor_name", out_tensor_name } })
    {
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

PYBIND11_MODULE(_packed_format_converter, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    py::module_ hololink_module = py::module_::import("hololink");

    auto op = py::class_<PackedFormatConverterOp, PyPackedFormatConverterOp, holoscan::Operator, hololink::csi::CsiConverter,
        std::shared_ptr<PackedFormatConverterOp>>(m, "PackedFormatConverterOp")
                  .def(py::init<holoscan::Fragment*, const std::shared_ptr<holoscan::Allocator>&,
                           py::object, int, const std::string&, const std::string&, const std::string&>(),
                      "fragment"_a, "allocator"_a, "hololink_channel"_a = nullptr, "cuda_device_ordinal"_a = 0,
                      "name"_a = "packed_format_converter"s, "in_tensor_name"_a = ""s, "out_tensor_name"_a = ""s)
                  .def("setup", &PackedFormatConverterOp::setup, "spec"_a)
                  .def("configure", &PackedFormatConverterOp::configure, "start_byte"_a, "bytes_per_line"_a, "pixel_width"_a, "pixel_height"_a,
                      "pixel_format"_a, "trailing_bytes"_a = 0)
                  .def("get_frame_size", &PackedFormatConverterOp::get_frame_size);
} // PYBIND11_MODULE

} // namespace hololink::operators
