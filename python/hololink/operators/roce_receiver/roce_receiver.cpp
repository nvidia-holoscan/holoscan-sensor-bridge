/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <hololink/operators/roce_receiver/roce_receiver.hpp>
#include <hololink/operators/roce_receiver/roce_receiver_op.hpp>

#include "../operator_util.hpp"
// NOLINTNEXTLINE - Provides CUstream type caster for pybind11 used in get_next_frame
#include "../type_caster.hpp" // IWYU pragma: keep

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

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

/* Trampoline class for RoceReceiver to allow Python overrides of virtual methods */
class PyRoceReceiver : public RoceReceiver {
public:
    using RoceReceiver::RoceReceiver;

    void copy_metadata_to_host(uint32_t page, void* metadata_buffer, CUevent metadata_event) override
    {
        PYBIND11_OVERRIDE(void, RoceReceiver, copy_metadata_to_host, page, metadata_buffer, metadata_event);
    }

    const Hololink::FrameMetadata get_frame_metadata(void* metadata_buffer, CUevent metadata_event) override
    {
        PYBIND11_OVERRIDE(Hololink::FrameMetadata, RoceReceiver, get_frame_metadata, metadata_buffer, metadata_event);
    }
};

/* Trampoline classes for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the operator.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the operator's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_operator<OperatorT>
 */
class PyRoceReceiverOp : public RoceReceiverOp {
public:
    /* Inherit the constructors */
    using RoceReceiverOp::RoceReceiverOp;

    // Define a constructor that fully initializes the object.
    PyRoceReceiverOp(holoscan::Fragment* fragment, const py::args& args,
        py::object hololink_channel, py::object device, py::object frame_context, size_t frame_size,
        const std::string& ibv_name, uint32_t ibv_port, py::object rename_metadata, bool trim,
        bool use_frame_ready_condition, uint32_t pages, uint32_t queue_size, const std::string& name, size_t metadata_offset)
        : RoceReceiverOp(holoscan::ArgList {
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
            holoscan::Arg { "ibv_name", ibv_name },
            holoscan::Arg { "ibv_port", ibv_port },
            holoscan::Arg { "trim", trim },
            holoscan::Arg { "use_frame_ready_condition", use_frame_ready_condition },
            holoscan::Arg { "pages", pages },
            holoscan::Arg { "queue_size", queue_size },
            holoscan::Arg { "metadata_offset", metadata_offset } })
        , device_(device)
    {
        add_positional_condition_and_resource_args(this, args);
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());

        // Set the rename_metadata function if provided
        if (!rename_metadata.is_none()) {
            auto rename_fn = std::function<std::string(const std::string&)>([rename_metadata](const std::string& name) {
                py::gil_scoped_acquire guard;
                return rename_metadata(name).cast<std::string>();
            });
            set_rename_metadata(rename_fn);
        }
    }

    // the `'start`, 'stop` and `get_next_frame` functions are overwritten by the
    // `InstrumentedReceiverOperator` to measure performance

    void start() override
    {
        PYBIND11_OVERRIDE(void, /* Return type */
            RoceReceiverOp, /* Parent class */
            start, /* Name of function in C++ (must match Python name) */
        );
    }

    void stop() override
    {
        PYBIND11_OVERRIDE(void, /* Return type */
            RoceReceiverOp, /* Parent class */
            stop, /* Name of function in C++ (must match Python name) */
        );
    }

    std::tuple<CUdeviceptr, std::shared_ptr<Metadata>> get_next_frame(double timeout_ms, CUstream cuda_stream) override
    {
        typedef std::tuple<CUdeviceptr, std::shared_ptr<Metadata>> FrameData;
        PYBIND11_OVERRIDE(FrameData, /* Return type */
            RoceReceiverOp, /* Parent class */
            get_next_frame, /* Name of function in C++ (must match Python name) */
            timeout_ms, cuda_stream);
    }

    bool frames_ready() override
    {
        PYBIND11_OVERRIDE(bool, /* Return type */
            RoceReceiverOp, /* Parent class */
            frames_ready /* Name of function in C++ (must match Python name) */
        );
    }

private:
    py::object device_;
};

PYBIND11_MODULE(_roce_receiver, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    // Bind RoceReceiverMetadata class
    py::class_<RoceReceiverMetadata>(m, "RoceReceiverMetadata")
        .def(py::init<>())
        .def_readwrite("rx_write_requests", &RoceReceiverMetadata::rx_write_requests)
        .def_readwrite("received_frame_number", &RoceReceiverMetadata::received_frame_number)
        .def_readwrite("imm_data", &RoceReceiverMetadata::imm_data)
        .def_readwrite("received_s", &RoceReceiverMetadata::received_s)
        .def_readwrite("received_ns", &RoceReceiverMetadata::received_ns)
        .def_readwrite("frame_memory", &RoceReceiverMetadata::frame_memory)
        .def_readwrite("metadata_memory", &RoceReceiverMetadata::metadata_memory)
        .def_readwrite("dropped", &RoceReceiverMetadata::dropped)
        .def_readwrite("frame_metadata", &RoceReceiverMetadata::frame_metadata)
        .def_readwrite("frame_number", &RoceReceiverMetadata::frame_number);

    // Bind RoceReceiver class
    py::class_<RoceReceiver, PyRoceReceiver, std::shared_ptr<RoceReceiver>>(m, "RoceReceiver")
        .def(py::init<const char*, unsigned, CUdeviceptr, size_t, size_t, size_t, unsigned, size_t, const char*, unsigned>(),
            "ibv_name"_a, "ibv_port"_a, "cu_buffer"_a, "cu_buffer_size"_a, "cu_frame_size"_a,
            "cu_page_size"_a, "pages"_a, "metadata_offset"_a, "peer_ip"_a, "queue_size"_a = 1)
        // Release GIL when calling blocking_monitor.
        .def("blocking_monitor", &RoceReceiver::blocking_monitor, py::call_guard<py::gil_scoped_release>())
        .def("copy_metadata_to_host", &RoceReceiver::copy_metadata_to_host, "page"_a, "metadata_buffer"_a, "metadata_event"_a)
        .def("start", &RoceReceiver::start)
        .def("close", &RoceReceiver::close)
        .def(
            "get_next_frame", [](RoceReceiver& self, unsigned timeout_ms) {
                RoceReceiverMetadata metadata;
                bool success = self.get_next_frame(timeout_ms, metadata);
                return std::make_tuple(success, metadata);
            },
            py::call_guard<py::gil_scoped_release>(), "timeout_ms"_a)
        .def("frames_ready", &RoceReceiver::frames_ready)
        .def("get_frame_metadata", &RoceReceiver::get_frame_metadata, "metadata_buffer"_a, "metadata_event"_a)
        .def("get_qp_number", &RoceReceiver::get_qp_number)
        .def("get_rkey", &RoceReceiver::get_rkey)
        .def("external_frame_memory", &RoceReceiver::external_frame_memory)
        .def("set_frame_ready", &RoceReceiver::set_frame_ready, "frame_ready"_a);

    py::class_<RoceReceiverOp, PyRoceReceiverOp, holoscan::Operator,
        std::shared_ptr<RoceReceiverOp>>(m, "RoceReceiverOp")
        .def(py::init<holoscan::Fragment*, const py::args&, py::object, py::object, py::object,
                 size_t, const std::string&, uint32_t, py::object, bool, bool, uint32_t, uint32_t, const std::string&, size_t>(),
            "fragment"_a, "hololink_channel"_a, "device"_a, "frame_context"_a, "frame_size"_a,
            "ibv_name"_a = "roceP5p3s0f0", "ibv_port"_a = 1, "rename_metadata"_a = py::none(),
            "trim"_a = true, "use_frame_ready_condition"_a = true, "pages"_a = 2, "queue_size"_a = 1,
            "name"_a = "roce_receiver"s, "metadata_offset"_a = 0)
        .def("get_next_frame", &RoceReceiverOp::get_next_frame, "timeout_ms"_a, "cuda_stream"_a, py::call_guard<py::gil_scoped_release>())
        .def("frames_ready", &RoceReceiverOp::frames_ready, py::call_guard<py::gil_scoped_release>())
        .def("setup", &RoceReceiverOp::setup, "spec"_a)
        .def("start", &RoceReceiverOp::start)
        .def("stop", &RoceReceiverOp::stop)
        .def("set_rename_metadata", &RoceReceiverOp::set_rename_metadata, "rename_fn"_a);

} // PYBIND11_MODULE

} // namespace hololink::operators
