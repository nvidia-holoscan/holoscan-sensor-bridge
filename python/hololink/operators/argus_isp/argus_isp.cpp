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

#include <hololink/operators/argus_isp/argus_isp.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for unordered_map -> dict, etc.

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>

// Complete type inclusion
// ensure this includes the full definition of ArgusImpl
#include <hololink/operators/argus_isp/argus_impl.hpp>
// ensure this includes the full definition of CameraProvider
#include <hololink/operators/argus_isp/camera_provider.hpp>

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
class PyArgusIspOp : public ArgusIspOp {
public:
    /* Inherit the constructors */
    using ArgusIspOp::ArgusIspOp;

    // Define a constructor that fully initializes the object.
    PyArgusIspOp(holoscan::Fragment* fragment,
        int bayer_format,
        float exposure_time_ms,
        float analog_gain,
        int pixel_bit_depth,
        std::shared_ptr<holoscan::Allocator> pool,
        const std::string& name = "argus_isp",
        const std::string& out_tensor_name = "output",
        uint32_t camera_index = 0)
        : ArgusIspOp(holoscan::ArgList { holoscan::Arg { "bayer_format", bayer_format },
            holoscan::Arg { "exposure_time_ms", exposure_time_ms },
            holoscan::Arg { "analog_gain", analog_gain },
            holoscan::Arg { "pixel_bit_depth", pixel_bit_depth },
            holoscan::Arg { "pool", pool },
            holoscan::Arg { "out_tensor_name", out_tensor_name },
            holoscan::Arg { "camera_index", camera_index } })
    {
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

PYBIND11_MODULE(_argus_isp, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    auto op = py::class_<ArgusIspOp, PyArgusIspOp, holoscan::Operator, std::shared_ptr<ArgusIspOp>>(m,
        "ArgusIspOp")
                  .def(py::init<holoscan::Fragment*,
                           int,
                           float,
                           float,
                           int,
                           std::shared_ptr<holoscan::Allocator>,
                           const std::string&,
                           const std::string&,
                           uint32_t>(),
                      "fragment"_a,
                      "bayer_format"_a,
                      "exposure_time_ms"_a,
                      "analog_gain"_a,
                      "pixel_bit_depth"_a,
                      "pool"_a,
                      "name"_a = "argus_isp"s,
                      "out_tensor_name"_a = "output"s,
                      "camera_index"_a = 0)
                  .def("setup", &ArgusIspOp::setup, "spec"_a);

    py::enum_<ArgusIspOp::BayerFormat>(op, "BayerFormat")
        .value("RGGB", ArgusIspOp::BayerFormat::RGGB,
            R"pbdoc(RGGB)pbdoc")
        .value("GBRG", ArgusIspOp::BayerFormat::GBRG,
            R"pbdoc(GBRG)pbdoc")
        .export_values();

} // PYBIND11_MODULE

} // namespace holoscan::operators
