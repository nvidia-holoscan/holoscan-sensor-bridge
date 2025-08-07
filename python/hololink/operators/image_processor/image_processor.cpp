/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <hololink/operators/image_processor/image_processor.hpp>

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
class PyImageProcessorOp : public ImageProcessorOp {
public:
    /* Inherit the constructors */
    using ImageProcessorOp::ImageProcessorOp;

    // Define a constructor that fully initializes the object.
    PyImageProcessorOp(holoscan::Fragment* fragment, int pixel_format, int bayer_format, int32_t optical_black, int cuda_device_ordinal, const std::string& name = "image_processor")
        : ImageProcessorOp(holoscan::ArgList { holoscan::Arg { "pixel_format", pixel_format },
            holoscan::Arg { "bayer_format", bayer_format },
            holoscan::Arg { "optical_black", optical_black },
            holoscan::Arg { "cuda_device_ordinal", cuda_device_ordinal } })
    {
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

PYBIND11_MODULE(_image_processor, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    auto op = py::class_<ImageProcessorOp, PyImageProcessorOp, holoscan::Operator, std::shared_ptr<ImageProcessorOp>>(m,
        "ImageProcessorOp")
                  .def(py::init<holoscan::Fragment*, int, int, int32_t, int, const std::string&>(),
                      "fragment"_a,
                      "pixel_format"_a,
                      "bayer_format"_a,
                      "optical_black"_a = 0,
                      "cuda_device_ordinal"_a = 0,
                      "name"_a = "image_processor"s)
                  .def("setup", &ImageProcessorOp::setup, "spec"_a);
} // PYBIND11_MODULE

} // namespace hololink::operators
