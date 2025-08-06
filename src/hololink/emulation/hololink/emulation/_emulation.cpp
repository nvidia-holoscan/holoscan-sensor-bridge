/**
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

#include "dlpack/dlpack.h"
#include "hololink/emulation/data_plane.hpp"
#include "hololink/emulation/hsb_emulator.hpp"
#include "hololink/emulation/linux_data_plane.hpp"
#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace hololink::emulation {

int64_t py_send_dltensor(DataPlane* self, py::object py_tensor)
{
    py::object capsule = py_tensor.attr("__dlpack__")();
    if (!PyCapsule_CheckExact(capsule.ptr())) {
        throw std::runtime_error("object does not satisfy Python array API standard - __dlpack__() returned non-capsule object");
    }

    DLTensor* tensor = (DLTensor*)PyCapsule_GetPointer(capsule.ptr(), "dltensor");
    if (tensor == NULL) {
        throw std::runtime_error("could not extract DLTensor from PyCapsule");
    }
    // Not renaming the PyCapsule object so that PyCapsule_Destructor gets
    // called and decrefs the DLManagedTensor and releases the memory for
    // this to work, send must not propagate the DLManagedTensor to other
    // objects
    return self->send(*tensor);
}

// DataPlane tranmpoline
class PyDataPlane : public DataPlane {
public:
    /* Inherit the constructors */
    using DataPlane::DataPlane;

    /* Trampoline (need one for each virtual function) */
    void update_metadata() override
    {
        PYBIND11_OVERRIDE_PURE(void, DataPlane, update_metadata, /* comma for no arguments */);
    }
};

class DataPlanePublicist : public DataPlane {
public:
    using DataPlane::update_metadata;
};

PYBIND11_MODULE(_emulation, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    py::enum_<DataPlaneID>(m, "DataPlaneID")
        .value("DATA_PLANE_0", DATA_PLANE_0)
        .value("DATA_PLANE_1", DATA_PLANE_1)
        .export_values();

    py::enum_<SensorID>(m, "SensorID")
        .value("SENSOR_0", SENSOR_0)
        .value("SENSOR_1", SENSOR_1)
        .export_values();

    py::class_<IPAddress>(m, "IPAddress")
        .def(py::init(&IPAddress_from_string), py::arg("ip_address"), py::arg("subnet_bits") = IP_ADDRESS_DEFAULT_BITS)
        .def("__str__", &IPAddress_to_string)
        .def_readwrite("ip_address", &IPAddress::ip_address)
        .def_readwrite("subnet_mask", &IPAddress::subnet_mask);

    py::class_<HSBEmulator>(m, "HSBEmulator")
        .def(py::init<>())
        .def("start", &HSBEmulator::start, "start HSB emulator")
        .def("stop", &HSBEmulator::stop, "stop HSB emulator")
        .def("is_running", &HSBEmulator::is_running, "check if HSB emulator is running");

    py::class_<DataPlane, PyDataPlane>(m, "DataPlane")
        .def("send", &py_send_dltensor)
        .def("start", &DataPlane::start, "start DataPlane")
        .def("stop", &DataPlane::stop, "stop DataPlane")
        .def("is_running", &DataPlane::is_running, "check if DataPlane is running")
        .def("update_metadata", &DataPlanePublicist::update_metadata, "update DataPlane metadata");

    py::class_<LinuxDataPlane, DataPlane>(m, "LinuxDataPlane")
        .def(py::init<HSBEmulator&, const IPAddress&, uint16_t, DataPlaneID, SensorID>(), py::arg("hsb_emulator"), py::arg("source_ip"), py::arg("source_port"), py::arg("data_plane_id"), py::arg("sensor_id"));
    /* not including update_metadata in a trampoline class for LinuxDataPlanebecause we are not allowing subclassing extensions of DataPlane subclasses yet */
} // PYBIND11_MODULE

} // namespace hololink::emulation
