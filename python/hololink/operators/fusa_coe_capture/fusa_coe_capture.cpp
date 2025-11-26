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

#include <hololink/operators/fusa_coe_capture/fusa_coe_capture.hpp>

#include "../operator_util.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <hololink/core/data_channel.hpp>

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace hololink::operators {

class PyFusaCoeCaptureOp : public FusaCoeCaptureOp {
public:
    /* Inherit the constructors */
    using FusaCoeCaptureOp::FusaCoeCaptureOp;

    // Define a constructor that fully initializes the object.
    PyFusaCoeCaptureOp(holoscan::Fragment* fragment, const py::args& args,
        const std::string& interface,
        const std::vector<uint8_t>& mac_addr,
        uint32_t timeout,
        py::object hololink_channel,
        py::object device,
        const std::string& out_tensor_name = "",
        const std::string& name = "fusa_coe_capture")
        : device_(device)
        , FusaCoeCaptureOp(holoscan::ArgList {
              holoscan::Arg { "interface", interface },
              holoscan::Arg { "mac_addr", mac_addr },
              holoscan::Arg { "timeout", timeout },
              holoscan::Arg { "hololink_channel", py::cast<DataChannel*>(hololink_channel) },
              holoscan::Arg { "out_tensor_name", out_tensor_name },
              holoscan::Arg { "device_start", std::function<void()>([this]() {
                                 py::gil_scoped_acquire guard;
                                 device_.attr("start")();
                             }) },
              holoscan::Arg { "device_stop", std::function<void()>([this]() {
                                 py::gil_scoped_acquire guard;
                                 device_.attr("stop")();
                             }) } })
    {
        add_positional_condition_and_resource_args(this, args);
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());
    }

private:
    py::object device_;
};

PYBIND11_MODULE(_fusa_coe_capture, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    auto op = py::class_<FusaCoeCaptureOp, PyFusaCoeCaptureOp, holoscan::Operator, std::shared_ptr<FusaCoeCaptureOp>>(m,
        "FusaCoeCaptureOp")
                  .def(py::init<holoscan::Fragment*,
                           const py::args&,
                           const std::string&,
                           const std::vector<uint8_t>&,
                           uint32_t,
                           py::object,
                           py::object,
                           const std::string&,
                           const std::string&>(),
                      "fragment"_a,
                      "interface"_a,
                      "mac_addr"_a,
                      "timeout"_a,
                      "hololink_channel"_a,
                      "device"_a,
                      "out_tensor_name"_a = ""s,
                      "name"_a = "fusa_coe_capture"s)
                  .def("setup", &FusaCoeCaptureOp::setup, "spec"_a)
                  .def("start", &FusaCoeCaptureOp::start)
                  .def("stop", &FusaCoeCaptureOp::stop)
                  .def("configure", &FusaCoeCaptureOp::configure,
                      "start_byte"_a, "received_bytes_per_line"_a,
                      "pixel_width"_a, "pixel_height"_a, "pixel_format"_a,
                      "trailing_bytes"_a = 0)
                  .def("receiver_start_byte", &FusaCoeCaptureOp::receiver_start_byte)
                  .def("received_line_bytes", &FusaCoeCaptureOp::received_line_bytes,
                      "line_bytes"_a)
                  .def("transmitted_line_bytes", &FusaCoeCaptureOp::transmitted_line_bytes,
                      "pixel_format"_a, "pixel_width"_a)
                  .def("configure_converter", &FusaCoeCaptureOp::configure_converter,
                      "converter"_a);

} // PYBIND11_MODULE

} // namespace holoscan::operators
