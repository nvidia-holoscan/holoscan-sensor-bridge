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

#include <hololink/operators/linux_coe_receiver/linux_coe_receiver.hpp>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace hololink::operators {

PYBIND11_MODULE(_linux_coe_receiver, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    // NOTE: pybind11 never implicitly release the GIL (see https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil),
    //       therefore for blocking function explicitly release the GIL using `py::call_guard<py::gil_scoped_release>()`.
    py::class_<LinuxCoeReceiver>(m, "LinuxCoeReceiver")
        .def(py::init<CUdeviceptr, size_t, int, uint16_t>(), "cu_buffer"_a, "cu_buffer_size"_a, "socket"_a, "channel"_a)
        .def("run", &LinuxCoeReceiver::run, py::call_guard<py::gil_scoped_release>())
        .def("close", &LinuxCoeReceiver::close)
        .def(
            "get_next_frame", [](LinuxCoeReceiver& self, unsigned timeout_ms) {
                LinuxCoeReceiverMetadata metadata;
                bool success = self.get_next_frame(timeout_ms, metadata);
                return std::make_tuple(success, metadata);
            },
            py::call_guard<py::gil_scoped_release>(), "timeout_ms"_a)
        .def("set_frame_ready", &LinuxCoeReceiver::set_frame_ready, "frame_ready"_a);

    py::class_<LinuxCoeReceiverMetadata>(m, "LinuxCoeReceiverMetadata")
        .def_readonly("frame_packets_received", &LinuxCoeReceiverMetadata::frame_packets_received)
        .def_readonly("frame_bytes_received", &LinuxCoeReceiverMetadata::frame_bytes_received)
        .def_readonly("received_frame_number", &LinuxCoeReceiverMetadata::received_frame_number)
        .def_readonly("frame_start_s", &LinuxCoeReceiverMetadata::frame_start_s)
        .def_readonly("frame_start_ns", &LinuxCoeReceiverMetadata::frame_start_ns)
        .def_readonly("frame_end_s", &LinuxCoeReceiverMetadata::frame_end_s)
        .def_readonly("frame_end_ns", &LinuxCoeReceiverMetadata::frame_end_ns)
        .def_readonly("received_s", &LinuxCoeReceiverMetadata::received_s)
        .def_readonly("received_ns", &LinuxCoeReceiverMetadata::received_ns)
        .def_property_readonly("timestamp_s", [](LinuxCoeReceiverMetadata& me) {
            return me.frame_metadata.timestamp_s;
        })
        .def_property_readonly("timestamp_ns", [](LinuxCoeReceiverMetadata& me) {
            return me.frame_metadata.timestamp_ns;
        })
        .def_property_readonly("metadata_s", [](LinuxCoeReceiverMetadata& me) {
            return me.frame_metadata.metadata_s;
        })
        .def_property_readonly("metadata_ns", [](LinuxCoeReceiverMetadata& me) {
            return me.frame_metadata.metadata_ns;
        })
        .def_property_readonly("crc", [](LinuxCoeReceiverMetadata& me) {
            return me.frame_metadata.crc;
        })
        .def_property_readonly("psn", [](LinuxCoeReceiverMetadata& me) {
            return me.frame_metadata.psn;
        })
        .def_property_readonly("frame_number", [](LinuxCoeReceiverMetadata& me) {
            return me.frame_number;
        })
        .def_property_readonly("bytes_written", [](LinuxCoeReceiverMetadata& me) {
            return me.frame_metadata.bytes_written;
        });

} // PYBIND11_MODULE

} // namespace hololink::operators
