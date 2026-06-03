/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "traditional_peripherals.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace hololink;

namespace {

// Owning wrappers so the internally-constructed Hololink outlives the
// Spi/I2c handle (TraditionalSpi/TraditionalI2c hold a Hololink& member).
struct TraditionalSpiHandle {
    std::shared_ptr<Hololink> hololink;
    std::shared_ptr<Hololink::Spi> spi;

    std::vector<uint8_t> spi_transaction(
        const std::vector<uint8_t>& write_command_bytes,
        const std::vector<uint8_t>& write_data_bytes,
        uint32_t read_byte_count)
    {
        return spi->spi_transaction(write_command_bytes, write_data_bytes, read_byte_count);
    }
};

struct TraditionalI2cHandle {
    std::shared_ptr<Hololink> hololink;
    std::shared_ptr<Hololink::I2c> i2c;

    bool set_i2c_clock() { return i2c->set_i2c_clock(); }

    std::vector<uint8_t> i2c_transaction(
        uint32_t peripheral_i2c_address,
        const std::vector<uint8_t>& write_bytes,
        uint32_t read_byte_count,
        bool ignore_nak)
    {
        return i2c->i2c_transaction(peripheral_i2c_address, write_bytes, read_byte_count,
            std::shared_ptr<Timeout> {}, ignore_nak);
    }
};

} // namespace

PYBIND11_MODULE(traditional_peripherals_py, m)
{
    m.doc() = "Legacy traditional SPI/I2C peripheral interfaces for older HSB FPGAs. "
              "Takes connection primitives so no hololink types cross the Python boundary.";

    py::class_<TraditionalSpiHandle, std::shared_ptr<TraditionalSpiHandle>>(m, "Spi")
        .def("spi_transaction", &TraditionalSpiHandle::spi_transaction,
            "write_command_bytes"_a, "write_data_bytes"_a, "read_byte_count"_a);

    py::class_<TraditionalI2cHandle, std::shared_ptr<TraditionalI2cHandle>>(m, "I2c")
        .def("set_i2c_clock", &TraditionalI2cHandle::set_i2c_clock)
        .def("i2c_transaction", &TraditionalI2cHandle::i2c_transaction,
            "peripheral_i2c_address"_a, "write_bytes"_a, "read_byte_count"_a,
            "ignore_nak"_a = false);

    m.def(
        "get_traditional_spi",
        [](const std::string& peer_ip, uint32_t control_port, const std::string& serial_number,
            uint32_t spi_address, uint32_t chip_select, uint32_t clock_divisor,
            uint32_t cpol, uint32_t cpha, uint32_t width) {
            auto hololink = std::make_shared<Hololink>(
                peer_ip, control_port, serial_number,
                false /*sequence_number_checking*/,
                false /*skip_sequence_initialization*/,
                false /*ptp_enable*/,
                false /*block_enable*/);
            hololink->start();
            auto spi = hololink::get_traditional_spi(
                *hololink, spi_address, chip_select, clock_divisor, cpol, cpha, width);
            return std::make_shared<TraditionalSpiHandle>(
                TraditionalSpiHandle { std::move(hololink), std::move(spi) });
        },
        "peer_ip"_a, "control_port"_a, "serial_number"_a,
        "spi_address"_a, "chip_select"_a, "clock_divisor"_a,
        "cpol"_a, "cpha"_a, "width"_a);

    m.def(
        "get_traditional_i2c",
        [](const std::string& peer_ip, uint32_t control_port, const std::string& serial_number,
            uint32_t i2c_address) {
            auto hololink = std::make_shared<Hololink>(
                peer_ip, control_port, serial_number,
                false /*sequence_number_checking*/,
                false /*skip_sequence_initialization*/,
                false /*ptp_enable*/,
                false /*block_enable*/);
            hololink->start();
            auto i2c = hololink::get_traditional_i2c(*hololink, i2c_address);
            return std::make_shared<TraditionalI2cHandle>(
                TraditionalI2cHandle { std::move(hololink), std::move(i2c) });
        },
        "peer_ip"_a, "control_port"_a, "serial_number"_a,
        "i2c_address"_a);
}
