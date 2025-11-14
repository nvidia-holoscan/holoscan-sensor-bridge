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

#include <hololink/operators/gpu_roce_transceiver/gpu_roce_transceiver_op.hpp>

#include "../operator_util.hpp"

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/condition.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resource.hpp>

#include <hololink/core/data_channel.hpp>

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace hololink::operators {

class PyGpuRoceTransceiverOp : public GpuRoceTransceiverOp {
public:
    using GpuRoceTransceiverOp::GpuRoceTransceiverOp;

    PyGpuRoceTransceiverOp(holoscan::Fragment* fragment, const py::args& args,
        py::object hololink_channel, py::object device, py::object frame_context, size_t frame_size,
        const std::string& ibv_name, uint32_t ibv_port, uint32_t gpu_id, py::object rename_metadata, const std::string& name, bool trim)
        : GpuRoceTransceiverOp(holoscan::ArgList {
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
            holoscan::Arg { "gpu_id", gpu_id },
            holoscan::Arg { "trim", trim } })
        , device_(device)
    {
        // Block scheduling Conditions passed positionally; allow Resources
        for (auto it = args.begin(); it != args.end(); ++it) {
            if (py::isinstance<holoscan::Condition>(*it)) {
                HSB_LOG_WARN("Condition passed to GpuRoceTransceiverOp is ignored; the operator runs autonomously without ticks.");
                continue;
            }
            if (py::isinstance<holoscan::Resource>(*it)) {
                add_arg(it->cast<std::shared_ptr<holoscan::Resource>>());
                continue;
            }
            HSB_LOG_WARN("Unhandled positional argument detected (only Resource objects are accepted; Conditions are ignored)");
        }
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<holoscan::OperatorSpec>(fragment);
        setup(*spec_.get());

        if (!rename_metadata.is_none()) {
            auto rename_fn = std::function<std::string(const std::string&)>([rename_metadata](const std::string& name) {
                py::gil_scoped_acquire guard;
                return rename_metadata(name).cast<std::string>();
            });
            set_rename_metadata(rename_fn);
        }
    }

    // The `'start`, 'stop` and `get_next_frame` functions are overwritten by the
    // `InstrumentedReceiverOperator` to measure performance

    void start() override
    {
        PYBIND11_OVERRIDE(void, /* Return type */
            GpuRoceTransceiverOp, /* Parent class */
            start, /* Name of function in C++ (must match Python name) */
        );
    }

    void stop() override
    {
        PYBIND11_OVERRIDE(void, /* Return type */
            GpuRoceTransceiverOp, /* Parent class */
            stop, /* Name of function in C++ (must match Python name) */
        );
    }

    std::tuple<CUdeviceptr, std::shared_ptr<Metadata>> get_next_frame(double timeout_ms) override
    {
        typedef std::tuple<CUdeviceptr, std::shared_ptr<Metadata>> FrameData;
        PYBIND11_OVERRIDE(FrameData, /* Return type */
            GpuRoceTransceiverOp, /* Parent class */
            get_next_frame, /* Name of function in C++ (must match Python name) */
            timeout_ms);
    }

private:
    py::object device_;
};

PYBIND11_MODULE(_gpu_roce_transceiver, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    py::class_<GpuRoceTransceiverOp, PyGpuRoceTransceiverOp, holoscan::Operator,
        std::shared_ptr<GpuRoceTransceiverOp>>(m, "GpuRoceTransceiverOp")
        .def(py::init<holoscan::Fragment*, const py::args&, py::object, py::object, py::object,
                 size_t, const std::string&, uint32_t, uint32_t, py::object, const std::string&, bool>(),
            "fragment"_a, "hololink_channel"_a, "device"_a, "frame_context"_a, "frame_size"_a,
            "ibv_name"_a = "roceP5p3s0f0", "ibv_port"_a = 1, "gpu_id"_a = 0, "rename_metadata"_a = py::none(), "name"_a = "gpu_roce_transceiver"s, "trim"_a = false)
        .def("get_next_frame", &GpuRoceTransceiverOp::get_next_frame, "timeout_ms"_a)
        .def("setup", &GpuRoceTransceiverOp::setup, "spec"_a)
        .def("start", &GpuRoceTransceiverOp::start)
        .def("stop", &GpuRoceTransceiverOp::stop)
        .def("set_rename_metadata", &GpuRoceTransceiverOp::set_rename_metadata, "rename_fn"_a);
} // PYBIND11_MODULE

} // namespace hololink::operators
