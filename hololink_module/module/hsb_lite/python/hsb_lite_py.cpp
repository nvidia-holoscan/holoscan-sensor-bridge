/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hsb_lite/hsb_lite.hpp"
#include "hololink/module/module.hpp"
#include "hololink/module/status.h"

namespace py = pybind11;
namespace ad = hololink::module;

PYBIND11_MODULE(_hololink_module_hsb_lite, m)
{
    m.doc() = "hololink_module HSB-Lite supplement bindings";

    // The core extension registers the V1 base types HsbLiteInterface
    // depends on (Module, EnumerationMetadata). Importing it here
    // ensures those registrations are present before pybind tries to
    // resolve cross-module references.
    py::module_::import("hololink_module._hololink_py_module");

    py::class_<ad::hsb_lite::HsbLiteInterfaceV1,
        std::shared_ptr<ad::hsb_lite::HsbLiteInterfaceV1>>(m, "HsbLiteInterfaceV1")
        .def_static(
            "get_service",
            [](std::shared_ptr<ad::Module> module, const std::string& instance_id,
                bool allow_null) {
                return ad::hsb_lite::HsbLiteInterfaceV1::get_service(
                    std::move(module), instance_id.c_str(), allow_null);
            },
            py::arg("module"), py::arg("instance_id"),
            py::arg("allow_null") = false)
        .def_static(
            "get_service",
            [](const ad::EnumerationMetadata& metadata, bool allow_null) {
                return ad::hsb_lite::HsbLiteInterfaceV1::get_service(
                    metadata, allow_null);
            },
            py::arg("metadata"), py::arg("allow_null") = false)
        .def(
            "setup_clock",
            [](ad::hsb_lite::HsbLiteInterfaceV1& self,
                py::iterable clock_profile) {
                // Accept the shape returned by the legacy
                // renesas_bajoran_lite_ts1.device_configuration():
                // list[bytes]. pybind11's stl.h caster for
                // vector<uint8_t> doesn't unwrap bytes automatically,
                // so handle both bytes and list[int] here.
                std::vector<std::vector<uint8_t>> profile;
                for (py::handle item : clock_profile) {
                    if (py::isinstance<py::bytes>(item)
                        || py::isinstance<py::bytearray>(item)) {
                        const std::string buffer = item.cast<std::string>();
                        profile.emplace_back(buffer.begin(), buffer.end());
                    } else {
                        profile.push_back(item.cast<std::vector<uint8_t>>());
                    }
                }
                const hololink_module_status_t s = self.setup_clock(profile);
                if (s != HOLOLINK_MODULE_OK) {
                    throw std::runtime_error(
                        "While invoking HsbLiteInterface.setup_clock: status "
                        + std::to_string(s));
                }
            },
            py::arg("clock_profile"))
        .def(
            "trigger_reset",
            [](ad::hsb_lite::HsbLiteInterfaceV1& self) {
                const hololink_module_status_t s = self.trigger_reset();
                if (s != HOLOLINK_MODULE_OK) {
                    throw std::runtime_error(
                        "While invoking HsbLiteInterface.trigger_reset: status "
                        + std::to_string(s));
                }
            },
            "Reset the board and return without waiting for it to re-announce "
            "or reconfiguring HSB; the board reboots and device I/O fails until "
            "it re-enumerates. Recovery is left to the pipeline (unlike the "
            "blocking reset()); models an abrupt loss for reconnection tests.");
}
