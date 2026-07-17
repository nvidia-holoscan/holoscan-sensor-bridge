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
# TauroTech DA326 module. The current FPGA firmware does not report a
# compat_id in its BOOTP advertisement, so NO_COMPAT_SUFFIX is used as
# the sole module. A compat-suffixed variant (e.g. COMPAT 2606) can be
# added alongside this one once the firmware begins advertising it.
#
# add_hololink_module() emits
#   hololink_9957a9ac-36b5-4518-83ec-5d514aecb750.so
# under the build tree's module dir.

add_hololink_module(
    NAME taurotech_da326
    UUID 9957a9ac-36b5-4518-83ec-5d514aecb750
    NO_COMPAT_SUFFIX
    SOURCES
        module_entry.cpp
)

target_include_directories(taurotech_da326
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../hsb_lite/include
        $<TARGET_PROPERTY:hololink::module_core,INCLUDE_DIRECTORIES>)

target_link_libraries(taurotech_da326
    PRIVATE
        hololink::module_core)

# Public header target for the supplement type.
add_library(hololink_taurotech_da326_headers INTERFACE)
add_library(hololink::taurotech_da326::headers ALIAS hololink_taurotech_da326_headers)
target_include_directories(hololink_taurotech_da326_headers
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(hololink_taurotech_da326_headers
    INTERFACE
        hololink::module_headers)
target_compile_features(hololink_taurotech_da326_headers INTERFACE cxx_std_17)

if(HOLOLINK_BUILD_PYTHON)
    include(hololink_deps/pybind11)

    pybind11_add_module(_hololink_module_taurotech_da326 MODULE
        python/taurotech_da326_py.cpp)
    target_link_libraries(_hololink_module_taurotech_da326
        PRIVATE
            hololink::taurotech_da326::headers
            hololink::module_core
            holoscan::pybind11)
    target_compile_features(_hololink_module_taurotech_da326 PRIVATE cxx_std_17)
    if(TARGET hololink_module_core)
        target_precompile_headers(_hololink_module_taurotech_da326
            REUSE_FROM hololink_module_core)
    endif()

    set_target_properties(_hololink_module_taurotech_da326 PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
            "${CMAKE_BINARY_DIR}/python/hololink_module/taurotech_da326")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/python/__init__.py"
        "${CMAKE_BINARY_DIR}/python/hololink_module/taurotech_da326/__init__.py"
        COPYONLY)
endif()
