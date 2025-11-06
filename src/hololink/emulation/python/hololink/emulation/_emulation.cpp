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
#include "hololink/emulation/coe_data_plane.hpp"
#include "hololink/emulation/data_plane.hpp"
#include "hololink/emulation/hsb_config.hpp"
#include "hololink/emulation/hsb_emulator.hpp"
#include "hololink/emulation/i2c_interface.hpp"
#include "hololink/emulation/linux_data_plane.hpp"
#include <cstring>
#include <pybind11/numpy.h>
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

// DataPlane trampoline
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

// I2CPeripheral trampoline
class PyI2CPeripheral : public I2CPeripheral {
public:
    /* Inherit the constructors */
    using I2CPeripheral::I2CPeripheral;

    // Trampoline (need one for each virtual function)
    I2CStatus i2c_transaction(uint8_t peripheral_address, const std::vector<uint8_t>& write_bytes, std::vector<uint8_t>& read_bytes) override
    {
        PYBIND11_OVERRIDE_PURE(I2CStatus, I2CPeripheral, i2c_transaction, peripheral_address, write_bytes, read_bytes);
    }

    void start() override
    {
        PYBIND11_OVERRIDE(void, I2CPeripheral, start, /* comma for no arguments */);
    }

    void stop() override
    {
        PYBIND11_OVERRIDE(void, I2CPeripheral, stop, /* comma for no arguments */);
    }

    void attach_to_i2c(I2CController& i2c_controller, uint8_t i2c_bus_address) override
    {
        PYBIND11_OVERRIDE_PURE(void, I2CPeripheral, attach_to_i2c, i2c_controller, i2c_bus_address);
    }
};

PYBIND11_MODULE(_emulation, m)
{
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    PYBIND11_NUMPY_DTYPE(HSBConfiguration, tag, tag_length, vendor_id, data_plane, enum_version, board_id_lo, board_id_hi, uuid, serial_num, hsb_ip_version, fpga_crc, sensor_count, data_plane_count, sifs_per_sensor);

    py::class_<HSBConfiguration>(m, "HSBConfiguration")
        .def(py::init<>())
        .def_readwrite("tag", &HSBConfiguration::tag)
        .def_readwrite("tag_length", &HSBConfiguration::tag_length)
        .def_property(
            "vendor_id",
            [](const HSBConfiguration& self) -> std::string {
                return std::string(reinterpret_cast<const char*>(self.vendor_id), VENDOR_ID_SIZE);
            },
            [](HSBConfiguration& self, const std::string& vendor_id) {
                if (vendor_id.length() != VENDOR_ID_SIZE) {
                    throw std::runtime_error("Vendor ID must be exactly 4 characters");
                }
                std::memcpy(self.vendor_id, vendor_id.c_str(), VENDOR_ID_SIZE);
            })
        .def_readwrite("data_plane", &HSBConfiguration::data_plane)
        .def_readwrite("enum_version", &HSBConfiguration::enum_version)
        .def_readwrite("board_id_lo", &HSBConfiguration::board_id_lo)
        .def_readwrite("board_id_hi", &HSBConfiguration::board_id_hi)
        .def_property(
            "uuid",
            [](const HSBConfiguration& self) -> std::string {
                return std::string(reinterpret_cast<const char*>(self.uuid), BOARD_VERSION_SIZE);
            },
            [](HSBConfiguration& self, const std::string& uuid_str) {
                if (hsb_config_set_uuid(self, uuid_str.c_str()) != 0) {
                    throw std::runtime_error("Invalid UUID format");
                }
            })
        .def_property(
            "serial_num",
            [](const HSBConfiguration& self) -> std::string {
                return std::string(reinterpret_cast<const char*>(self.serial_num), BOARD_SERIAL_NUM_SIZE);
            },
            [](HSBConfiguration& self, const std::string& serial_num) {
                if (serial_num.length() != BOARD_SERIAL_NUM_SIZE) {
                    throw std::runtime_error("Serial number must be exactly 7 characters");
                }
                std::memcpy(self.serial_num, serial_num.c_str(), BOARD_SERIAL_NUM_SIZE);
            })
        .def_readwrite("hsb_ip_version", &HSBConfiguration::hsb_ip_version)
        .def_readwrite("fpga_crc", &HSBConfiguration::fpga_crc)
        .def_readwrite("sensor_count", &HSBConfiguration::sensor_count)
        .def_readwrite("data_plane_count", &HSBConfiguration::data_plane_count)
        .def_readwrite("sifs_per_sensor", &HSBConfiguration::sifs_per_sensor);

    m.attr("HSB_EMULATOR_CONFIG") = HSB_EMULATOR_CONFIG;
    m.attr("HSB_LEOPARD_EAGLE_CONFIG") = HSB_LEOPARD_EAGLE_CONFIG;

    py::class_<IPAddress>(m, "IPAddress")
        .def(py::init([](const std::string& ip_address) {
            return IPAddress_from_string(ip_address);
        }),
            py::arg("ip_address"))
        .def("__str__", &IPAddress_to_string);

    // Helper function for setting UUID in HSBConfiguration
    m.def("hsb_config_set_uuid", &hsb_config_set_uuid, py::arg("config"), py::arg("uuid_str"), "Set UUID in HSBConfiguration from string");

    py::class_<HSBEmulator>(m, "HSBEmulator")
        .def(py::init<const HSBConfiguration&>(), py::arg("config") = HSB_EMULATOR_CONFIG)
        .def("start", &HSBEmulator::start, "start HSB emulator")
        .def("stop", &HSBEmulator::stop, "stop HSB emulator")
        .def("is_running", &HSBEmulator::is_running, "check if HSB emulator is running")
        .def("get_i2c", &HSBEmulator::get_i2c, "get I2CController from HSB emulator", py::return_value_policy::reference)
        .def("write", &HSBEmulator::write, "write to HSB emulator")
        .def("read", &HSBEmulator::read, "read from HSB emulator");

    py::class_<DataPlane, PyDataPlane>(m, "DataPlane")
        .def("send", &py_send_dltensor)
        .def("start", &DataPlane::start, "start DataPlane")
        .def("stop", &DataPlane::stop, "stop DataPlane")
        .def("stop_bootp", &DataPlane::stop_bootp, "stop bootp on DataPlane")
        .def("is_running", &DataPlane::is_running, "check if DataPlane is running")
        .def("get_sensor_id", &DataPlane::get_sensor_id, "get sensor id from DataPlane")
        .def("update_metadata", &DataPlanePublicist::update_metadata, "update DataPlane metadata")
        .def("packetizer_enabled", &DataPlane::packetizer_enabled, "check if packetizer is enabled");

    py::class_<LinuxDataPlane, DataPlane>(m, "LinuxDataPlane")
        .def(py::init<HSBEmulator&, IPAddress, uint8_t, uint8_t>(), py::arg("hsb_emulator"), py::arg("source_ip"), py::arg("data_plane_id"), py::arg("sensor_id"));
    /* not including update_metadata in a trampoline class for LinuxDataPlanebecause we are not allowing subclassing extensions of DataPlane subclasses yet */

    py::class_<COEDataPlane, DataPlane>(m, "COEDataPlane")
        .def(py::init<HSBEmulator&, IPAddress, uint8_t, uint8_t>(), py::arg("hsb_emulator"), py::arg("source_ip"), py::arg("data_plane_id"), py::arg("sensor_id"));

    // I2CStatus enumeration constants
    py::enum_<I2CStatus>(m, "I2CStatus")
        .value("I2C_STATUS_SUCCESS", I2CStatus::I2C_STATUS_SUCCESS)
        .value("I2C_STATUS_BUSY", I2CStatus::I2C_STATUS_BUSY)
        .value("I2C_STATUS_WRITE_FAILED", I2CStatus::I2C_STATUS_WRITE_FAILED)
        .value("I2C_STATUS_READ_FAILED", I2CStatus::I2C_STATUS_READ_FAILED)
        .value("I2C_STATUS_MESSAGE_NOT_UNDERSTOOD", I2CStatus::I2C_STATUS_MESSAGE_NOT_UNDERSTOOD)
        .value("I2C_STATUS_TIMEOUT", I2CStatus::I2C_STATUS_TIMEOUT)
        .value("I2C_STATUS_NACK", I2CStatus::I2C_STATUS_NACK)
        .value("I2C_STATUS_BUFFER_SIZE_SMALL", I2CStatus::I2C_STATUS_BUFFER_SIZE_SMALL)
        .value("I2C_STATUS_BUFFER_SIZE_LARGE", I2CStatus::I2C_STATUS_BUFFER_SIZE_LARGE)
        .value("I2C_STATUS_INVALID_PERIPHERAL_ADDRESS", I2CStatus::I2C_STATUS_INVALID_PERIPHERAL_ADDRESS)
        .value("I2C_STATUS_INVALID_REGISTER_ADDRESS", I2CStatus::I2C_STATUS_INVALID_REGISTER_ADDRESS)
        .value("I2C_STATUS_FATAL_ERROR", I2CStatus::I2C_STATUS_FATAL_ERROR);

    py::class_<I2CPeripheral, PyI2CPeripheral>(m, "I2CPeripheral")
        .def(py::init<>())
        .def("i2c_transaction", &I2CPeripheral::i2c_transaction, "perform an I2C transaction")
        .def("start", &I2CPeripheral::start, "start I2CPeripheral")
        .def("stop", &I2CPeripheral::stop, "stop I2CPeripheral")
        .def("attach_to_i2c", &I2CPeripheral::attach_to_i2c, "attach I2CPeripheral to I2CController");

    py::class_<I2CController>(m, "I2CController")
        .def("attach_i2c_peripheral", &I2CController::attach_i2c_peripheral, "attach I2CPeripheral to I2CController");
} // PYBIND11_MODULE

} // namespace hololink::emulation
