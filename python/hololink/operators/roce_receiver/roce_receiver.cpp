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

#include <hololink/operators/roce_receiver/roce_receiver.hpp>

#include <pybind11/pybind11.h>

using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace hololink::operators {

PYBIND11_MODULE(_roce_receiver, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    // NOTE: pybind11 never implicitly release the GIL (see https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil),
    //       therefore for blocking function explicitly release the GIL using `py::call_guard<py::gil_scoped_release>()`.
    py::class_<RoceReceiver>(m, "RoceReceiver")
        .def(py::init<const char*, unsigned, CUdeviceptr, size_t, const char*>(), "ibv_name"_a, "ibv_port"_a, "cu_buffer"_a, "cu_buffer_size"_a, "peer_ip"_a)
        .def("blocking_monitor", &RoceReceiver::blocking_monitor, py::call_guard<py::gil_scoped_release>())
        .def("start", &RoceReceiver::start)
        .def("close", &RoceReceiver::close)
        .def(
            "get_next_frame", [](RoceReceiver& self, unsigned timeout_ms) {
                RoceReceiverMetadata metadata;
                bool success = self.get_next_frame(timeout_ms, metadata);
                return std::make_tuple(success, metadata);
            },
            py::call_guard<py::gil_scoped_release>(), "timeout_ms"_a)
        .def("get_qp_number", &RoceReceiver::get_qp_number)
        .def("get_rkey", &RoceReceiver::get_rkey);

    py::class_<RoceReceiverMetadata>(m, "RoceReceiverMetadata")
        .def_readonly("rx_write_requests", &RoceReceiverMetadata::rx_write_requests)
        .def_readonly("frame_number", &RoceReceiverMetadata::frame_number)
        .def_readonly("frame_end_s", &RoceReceiverMetadata::frame_end_s)
        .def_readonly("frame_end_ns", &RoceReceiverMetadata::frame_end_ns);

} // PYBIND11_MODULE

} // namespace hololink::operators
