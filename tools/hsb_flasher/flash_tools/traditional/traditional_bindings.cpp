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

#include <hololink/core/hololink.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace hololink;

std::shared_ptr<Hololink::Spi> get_traditional_spi(Hololink& hololink, uint32_t spi_address, uint32_t chip_select,
    uint32_t clock_divisor, uint32_t cpol, uint32_t cpha, uint32_t width);

std::shared_ptr<Hololink::I2c> get_traditional_i2c(Hololink& hololink, uint32_t i2c_address);

PYBIND11_MODULE(traditional_peripherals_py, m)
{
    m.doc() = "Legacy traditional SPI/I2C peripheral interfaces for older HSB FPGAs";

    m.def("get_traditional_i2c", &get_traditional_i2c, "hololink"_a, "i2c_address"_a);

    m.def("get_traditional_spi", &get_traditional_spi,
        "hololink"_a, "spi_address"_a, "chip_select"_a, "clock_divisor"_a,
        "cpol"_a, "cpha"_a, "width"_a);
}
