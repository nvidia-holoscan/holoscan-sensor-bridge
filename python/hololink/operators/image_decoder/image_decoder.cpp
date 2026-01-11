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

#include <hololink/operators/image_decoder/image_decoder.hpp>

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
class PyImageDecoder : public ImageDecoder {
public:
    /* Inherit the constructors */
    using ImageDecoder::ImageDecoder;

    // Define a constructor that fully initializes the object.
    PyImageDecoder(holoscan::Fragment* fragment,
        const std::shared_ptr<holoscan::Allocator>& allocator, int cuda_device_ordinal,
        const std::string& name = "image_decoder",
        const std::string& out_tensor_name = "",
        const bool align_depth_to_rgb = false)
        : ImageDecoder(holoscan::ArgList { holoscan::Arg { "allocator", allocator },
            holoscan::Arg { "cuda_device_ordinal", cuda_device_ordinal },
            holoscan::Arg { "out_tensor_name", out_tensor_name }, holoscan::Arg { "align_depth_to_rgb", align_depth_to_rgb } })
    {
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

PYBIND11_MODULE(_image_decoder, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    auto op = py::class_<ImageDecoder, PyImageDecoder, holoscan::Operator,
        std::shared_ptr<ImageDecoder>>(m, "ImageDecoderOp")
                  .def(py::init<holoscan::Fragment*, const std::shared_ptr<holoscan::Allocator>&,
                           int, const std::string&, const std::string&, const bool>(),
                      "fragment"_a, "allocator"_a, "cuda_device_ordinal"_a = 0,
                      "name"_a = "image_decoder"s, "out_tensor_name"_a = ""s,
                      "align_depth_to_rgb"_a = false)
                  .def("setup", &ImageDecoder::setup, "spec"_a)
                  .def("configure", &ImageDecoder::configure, "width"_a, "height"_a,
                      "pixel_format"_a, "frame_start_size"_a, "frame_end_size"_a,
                      "line_start_size"_a, "line_end_size"_a, "margin_left"_a = 0,
                      "margin_top"_a = 0, "margin_right"_a = 0, "margin_bottom"_a = 0)
                  .def("get_csi_length", &ImageDecoder::get_csi_length);

    py::enum_<ImageDecoder::PixelFormat>(op, "PixelFormat")
        .value("Z16", ImageDecoder::PixelFormat::Z16, R"pbdoc(Z16-bit)pbdoc")
        .value("YUYV", ImageDecoder::PixelFormat::YUYV, R"pbdoc(YUYV-bit)pbdoc")
        .export_values();

} // PYBIND11_MODULE

} // namespace hololink::operators
