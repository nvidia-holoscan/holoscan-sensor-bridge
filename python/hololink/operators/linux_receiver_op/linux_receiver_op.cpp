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

#include <hololink/operators/linux_receiver/linux_receiver_op.hpp>

#include "../operator_util.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <set>
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
class PyLinuxReceiverOp : public LinuxReceiverOp {
public:
    /* Inherit the constructors */
    using LinuxReceiverOp::LinuxReceiverOp;

    // Define a constructor that fully initializes the object.
    PyLinuxReceiverOp(holoscan::Fragment* fragment, const py::args& args,
        py::object hololink_channel, py::object device, py::object frame_context, size_t frame_size,
        py::object receiver_affinity, py::object rename_metadata, const std::string& name, bool trim)
        : LinuxReceiverOp(holoscan::ArgList {
            holoscan::Arg { "hololink_channel", py::cast<DataChannel*>(hololink_channel) },
            holoscan::Arg { "device_start", std::function<void()>([this]() {
                               py::gil_scoped_acquire guard;
                               device_.attr("start")();
                           }) },
            holoscan::Arg { "device_stop", std::function<void()>([this]() {
                               py::gil_scoped_acquire guard;
                               device_.attr("stop")();
                           }) },
            holoscan::Arg {
                "frame_context", reinterpret_cast<CUcontext>(frame_context.cast<int64_t>()) },
            holoscan::Arg { "frame_size", frame_size },
            holoscan::Arg { "trim", trim } })
        , device_(device)
    {
        add_positional_condition_and_resource_args(this, args);
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());

        // Set the receiver_affinity if provided
        if (!receiver_affinity.is_none()) {
            if (py::isinstance<py::set>(receiver_affinity)) {
                auto affinity_set = receiver_affinity.cast<std::vector<int>>();
                set_receiver_affinity(affinity_set);
            } else if (py::isinstance<py::int_>(receiver_affinity)) {
                int affinity_value = receiver_affinity.cast<int>();
                set_receiver_affinity({ affinity_value });
            }
        }

        // Set the rename_metadata function if provided
        if (!rename_metadata.is_none()) {
            auto rename_fn = std::function<std::string(const std::string&)>([rename_metadata](const std::string& name) {
                py::gil_scoped_acquire guard;
                return rename_metadata(name).cast<std::string>();
            });
            set_rename_metadata(rename_fn);
        }
    }

    // the `start`, `stop` and `get_next_frame` functions are overwritten by the
    // `InstrumentedReceiverOperator` to measure performance

    void start() override
    {
        PYBIND11_OVERRIDE(void, /* Return type */
            LinuxReceiverOp, /* Parent class */
            start, /* Name of function in C++ (must match Python name) */
        );
    }

    void stop() override
    {
        PYBIND11_OVERRIDE(void, /* Return type */
            LinuxReceiverOp, /* Parent class */
            stop, /* Name of function in C++ (must match Python name) */
        );
    }

    std::tuple<CUdeviceptr, std::shared_ptr<Metadata>> get_next_frame(double timeout_ms) override
    {
        typedef std::tuple<CUdeviceptr, std::shared_ptr<Metadata>> FrameData;
        PYBIND11_OVERRIDE(FrameData, /* Return type */
            LinuxReceiverOp, /* Parent class */
            get_next_frame, /* Name of function in C++ (must match Python name) */
            timeout_ms);
    }

private:
    py::object device_;
};

PYBIND11_MODULE(_linux_receiver_op, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    py::class_<LinuxReceiverOp, PyLinuxReceiverOp, holoscan::Operator,
        std::shared_ptr<LinuxReceiverOp>>(m, "LinuxReceiverOp")
        .def(py::init<holoscan::Fragment*, const py::args&, py::object, py::object, py::object,
                 size_t, py::object, py::object, const std::string&, bool>(),
            "fragment"_a, "hololink_channel"_a, "device"_a, "frame_context"_a, "frame_size"_a,
            "receiver_affinity"_a = py::none(), "rename_metadata"_a = py::none(), "name"_a = "linux_receiver"s, "trim"_a = false)
        .def("get_next_frame", &LinuxReceiverOp::get_next_frame, "timeout_ms"_a)
        .def("setup", &LinuxReceiverOp::setup, "spec"_a)
        .def("start", &LinuxReceiverOp::start)
        .def("stop", &LinuxReceiverOp::stop)
        .def("set_rename_metadata", &LinuxReceiverOp::set_rename_metadata, "rename_fn"_a)
        .def("set_receiver_affinity", &LinuxReceiverOp::set_receiver_affinity, "affinity"_a);

} // PYBIND11_MODULE

} // namespace hololink::operators
