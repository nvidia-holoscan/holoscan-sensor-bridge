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

#include <hololink/operators/linux_receiver/linux_receiver.hpp>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace hololink::operators {

PYBIND11_MODULE(_linux_receiver, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    // NOTE: pybind11 never implicitly release the GIL (see https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil),
    //       therefore for blocking function explicitly release the GIL using `py::call_guard<py::gil_scoped_release>()`.
    py::class_<LinuxReceiver>(m, "LinuxReceiver")
        .def(py::init<CUdeviceptr, size_t, int, uint64_t>(), "cu_buffer"_a, "cu_buffer_size"_a, "socket"_a, "received_address_offset"_a)
        .def("run", &LinuxReceiver::run, py::call_guard<py::gil_scoped_release>())
        .def("close", &LinuxReceiver::close)
        .def(
            "get_next_frame", [](LinuxReceiver& self, unsigned timeout_ms) {
                LinuxReceiverMetadata metadata;
                bool success = self.get_next_frame(timeout_ms, metadata);
                return std::make_tuple(success, metadata);
            },
            py::call_guard<py::gil_scoped_release>(), "timeout_ms"_a)
        .def("get_qp_number", &LinuxReceiver::get_qp_number)
        .def("get_rkey", &LinuxReceiver::get_rkey)
        .def("set_frame_ready", &LinuxReceiver::set_frame_ready, "frame_ready"_a);

    py::class_<LinuxReceiverMetadata>(m, "LinuxReceiverMetadata")
        .def_readonly("frame_packets_received", &LinuxReceiverMetadata::frame_packets_received)
        .def_readonly("frame_bytes_received", &LinuxReceiverMetadata::frame_bytes_received)
        .def_readonly("received_frame_number", &LinuxReceiverMetadata::received_frame_number)
        .def_readonly("frame_start_s", &LinuxReceiverMetadata::frame_start_s)
        .def_readonly("frame_start_ns", &LinuxReceiverMetadata::frame_start_ns)
        .def_readonly("frame_end_s", &LinuxReceiverMetadata::frame_end_s)
        .def_readonly("frame_end_ns", &LinuxReceiverMetadata::frame_end_ns)
        .def_readonly("imm_data", &LinuxReceiverMetadata::imm_data)
        .def_readonly("packets_dropped", &LinuxReceiverMetadata::packets_dropped)
        .def_readonly("received_s", &LinuxReceiverMetadata::received_s)
        .def_readonly("received_ns", &LinuxReceiverMetadata::received_ns)
        .def_property_readonly("timestamp_s", [](LinuxReceiverMetadata& me) {
            return me.frame_metadata.timestamp_s;
        })
        .def_property_readonly("timestamp_ns", [](LinuxReceiverMetadata& me) {
            return me.frame_metadata.timestamp_ns;
        })
        .def_property_readonly("metadata_s", [](LinuxReceiverMetadata& me) {
            return me.frame_metadata.metadata_s;
        })
        .def_property_readonly("metadata_ns", [](LinuxReceiverMetadata& me) {
            return me.frame_metadata.metadata_ns;
        })
        .def_property_readonly("crc", [](LinuxReceiverMetadata& me) {
            return me.frame_metadata.crc;
        })
        .def_property_readonly("psn", [](LinuxReceiverMetadata& me) {
            return me.frame_metadata.psn;
        })
        .def_property_readonly("frame_number", [](LinuxReceiverMetadata& me) {
            return me.frame_number;
        })
        .def_property_readonly("bytes_written", [](LinuxReceiverMetadata& me) {
            return me.frame_metadata.bytes_written;
        });

} // PYBIND11_MODULE

} // namespace hololink::operators
