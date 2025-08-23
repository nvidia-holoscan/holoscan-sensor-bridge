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

#include <hololink/operators/sipl_capture/sipl_capture.hpp>

#include "../operator_util.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
class PySIPLCaptureOp : public SIPLCaptureOp {
public:
    /* Inherit the constructors */
    using SIPLCaptureOp::SIPLCaptureOp;

    // Define a constructor that fully initializes the object.
    PySIPLCaptureOp(holoscan::Fragment* fragment, const py::args& args,
        const std::string& camera_config,
        const std::string& json_config,
        bool raw_output,
        uint32_t capture_queue_depth,
        const std::string& nito_base_path,
        uint32_t timeout,
        const std::string& name = "sipl_capture")
        : SIPLCaptureOp(camera_config, json_config, raw_output,
            holoscan::ArgList {
                holoscan::Arg { "capture_queue_depth", capture_queue_depth },
                holoscan::Arg { "nito_base_path", nito_base_path },
                holoscan::Arg { "timeout", timeout } })
    {
        add_positional_condition_and_resource_args(this, args);
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

PYBIND11_MODULE(_sipl_capture, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    py::class_<SIPLCaptureOp::CameraInfo>(m, "CameraInfo")
        .def_readwrite("output_name", &SIPLCaptureOp::CameraInfo::output_name)
        .def_readwrite("offset", &SIPLCaptureOp::CameraInfo::offset)
        .def_readwrite("width", &SIPLCaptureOp::CameraInfo::width)
        .def_readwrite("height", &SIPLCaptureOp::CameraInfo::height)
        .def_readwrite("bytes_per_line", &SIPLCaptureOp::CameraInfo::bytes_per_line)
        .def_readwrite("pixel_format", &SIPLCaptureOp::CameraInfo::pixel_format)
        .def_readwrite("bayer_format", &SIPLCaptureOp::CameraInfo::bayer_format);

    py::class_<SIPLCaptureOp, PySIPLCaptureOp, holoscan::Operator,
        std::shared_ptr<SIPLCaptureOp>>(m, "SIPLCaptureOp")
        .def(py::init<holoscan::Fragment*, const py::args&,
                 const std::string&,
                 const std::string&,
                 bool,
                 uint32_t,
                 const std::string&,
                 uint32_t,
                 const std::string&>(),
            "fragment"_a,
            "camera_config"_a = "",
            "json_config"_a = "",
            "raw_output"_a = false,
            "capture_queue_depth"_a = 4u,
            "nito_base_path"_a = "/var/nvidia/nvcam/settings/sipl",
            "timeout"_a = 1000000u,
            "name"_a = "sipl_capture"s)
        .def("list_available_configs", &SIPLCaptureOp::list_available_configs)
        .def("get_camera_info", &SIPLCaptureOp::get_camera_info);
} // PYBIND11_MODULE

} // namespace hololink::operators
