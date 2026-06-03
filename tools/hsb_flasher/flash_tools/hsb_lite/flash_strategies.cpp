/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * flash_strategies - pybind11 module exposing C++ flash routines to Python
 *
 * This module provides the low-level flash operations that can be used
 * by firmware_flash_strategies Python scripts.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hsb_lite_flash_2407.hpp"
#include "hsb_lite_flash_2504.hpp"
#include "hsb_lite_flash_2507.hpp"

namespace py = pybind11;

PYBIND11_MODULE(flash_strategies, m)
{
    m.doc() = "Flash strategies - C++ flash routines for HSB devices";

    m.def("hsb_lite_flash_2507", &hololink::hsb_lite_flash_2507,
        "Flash an HSB_Lite device with CLNX and CPNX firmware (modern connection)",
        py::arg("ip_address"),
        py::arg("clnx_path"),
        py::arg("cpnx_path"));

    m.def("hsb_lite_flash_2407", &hololink::hsb_lite_flash_2407,
        "Flash using raw UDP (blind mode for ancient versions - no verification)",
        py::arg("ip_address"),
        py::arg("clnx_path"),
        py::arg("cpnx_path"));

    m.def("hsb_lite_flash_2504", &hololink::hsb_lite_flash_2504,
        "Flash using legacy connection mode (for older versions with workarounds)",
        py::arg("ip_address"),
        py::arg("clnx_path"),
        py::arg("cpnx_path"),
        py::arg("current_version"),
        py::arg("fpga_uuid"),
        py::arg("serial_number"));
}
