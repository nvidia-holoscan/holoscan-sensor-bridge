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

#include <pybind11/pybind11.h>

#include <memory>

#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/module.hpp"
#include "hololink/module/taurotech_da326/taurotech_da326.hpp"

namespace py = pybind11;
namespace ad = hololink::module;

PYBIND11_MODULE(_hololink_module_taurotech_da326, m)
{
    m.doc() = "hololink_module TauroTech DA326 supplement bindings";

    py::module_::import("hololink_module._hololink_py_module");

    py::class_<ad::taurotech_da326::TauroTechDa326InterfaceV1,
        std::shared_ptr<ad::taurotech_da326::TauroTechDa326InterfaceV1>>(
        m, "TauroTechDa326InterfaceV1")
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, const std::string& instance_id,
                bool allow_null) {
                return ad::taurotech_da326::TauroTechDa326InterfaceV1::get_service(
                    std::move(module), instance_id.c_str(), allow_null);
            },
            py::arg("module"), py::arg("instance_id"),
            py::arg("allow_null") = false)
        .def_static(
            "get_service",
            [](const ad::EnumerationMetadata& metadata, bool allow_null) {
                return ad::taurotech_da326::TauroTechDa326InterfaceV1::get_service(
                    metadata, allow_null);
            },
            py::arg("metadata"), py::arg("allow_null") = false)
        .def("release_reset",
            [](ad::taurotech_da326::TauroTechDa326InterfaceV1& self) {
                const auto s = self.release_reset();
                if (s != HOLOLINK_MODULE_OK)
                    throw std::runtime_error(
                        "TauroTechDa326Interface.release_reset failed: "
                        + std::to_string(s));
            })
        .def("power_cycle",
            [](ad::taurotech_da326::TauroTechDa326InterfaceV1& self) {
                const auto s = self.power_cycle();
                if (s != HOLOLINK_MODULE_OK)
                    throw std::runtime_error(
                        "TauroTechDa326Interface.power_cycle failed: "
                        + std::to_string(s));
            })
        .def("check_power",
            [](ad::taurotech_da326::TauroTechDa326InterfaceV1& self) {
                const auto s = self.check_power();
                if (s != HOLOLINK_MODULE_OK)
                    throw std::runtime_error(
                        "TauroTechDa326Interface.check_power failed: "
                        + std::to_string(s));
            })
        .def("setup_clock",
            [](ad::taurotech_da326::TauroTechDa326InterfaceV1& self) {
                const auto s = self.setup_clock();
                if (s != HOLOLINK_MODULE_OK)
                    throw std::runtime_error(
                        "TauroTechDa326Interface.setup_clock failed: "
                        + std::to_string(s));
            })
        .def("hololink",
            &ad::taurotech_da326::TauroTechDa326InterfaceV1::hololink);
}
