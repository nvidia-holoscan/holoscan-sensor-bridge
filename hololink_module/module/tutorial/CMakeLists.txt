# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Tutorial device module. A teaching example: an HSB-Lite specialization
# that adds one bespoke service (the status LED) and a single-data-plane
# sensor layout. Like the canonical hsb_lite module, it omits COMPAT and
# NO_COMPAT_SUFFIX, so add_hololink_module() uses the framework default
# compat-id (hololink_module_core's DEFAULT_COMPAT) and emits
#   hololink_d3061b3b-85b0-4096-ba57-296d2418477f_<compat>.so
# under the build tree's module dir. The host loads it by matching the
# board's advertised FPGA UUID and compat-id (its FPGA IP version).

add_hololink_module(
    NAME tutorial
    UUID d3061b3b-85b0-4096-ba57-296d2418477f
    SOURCES
        module_entry.cpp
)

target_include_directories(tutorial
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../hsb_lite/include
        $<TARGET_PROPERTY:hololink::module_core,INCLUDE_DIRECTORIES>)

target_link_libraries(tutorial
    PRIVATE
        hololink::module_core)

# THIS SECTION BELOW CAN BE OMITTED FOR MODULES THAT DON'T EXPORT
# A DEVICE-SPECIFIC INTERFACE OBJECT.
#
# Public header target for the bespoke service type. Consumed by C++
# applications that fetch TutorialDeviceInterfaceV1 (e.g. the status-LED
# example).
add_library(hololink_tutorial_headers INTERFACE)
add_library(hololink::tutorial::headers ALIAS hololink_tutorial_headers)
target_include_directories(hololink_tutorial_headers
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(hololink_tutorial_headers
    INTERFACE
        hololink::module_headers)
target_compile_features(hololink_tutorial_headers INTERFACE cxx_std_17)
