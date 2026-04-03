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
 * hsb_leopard_flash_strategies - pybind11 module exposing C++ flash routines to Python
 *
 * This module provides the low-level flash operations that can be used
 * by firmware_flash_strategies Python scripts for HSB Leopard devices.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hsb_leopard_flash_2507.hpp"

namespace py = pybind11;

PYBIND11_MODULE(hsb_leopard_flash_strategies, m)
{
    m.doc() = "Flash strategies - C++ flash routines for HSB Leopard devices";

    m.def("hsb_leopard_flash_2507", &hololink::hsb_leopard_flash_2507,
        "Flash an HSB_Leopard device with CPNX firmware (modern connection)",
        py::arg("ip_address"),
        py::arg("cpnx_path"));
}
