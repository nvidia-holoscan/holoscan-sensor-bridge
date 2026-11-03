# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# HSB-Lite module. Claims the HSB-Lite UUID, publishes the core
# default singletons, contributes its own EnumerationInterfaceV1
# override, and exports the HsbLiteInterfaceV1 supplement type.
#
# add_hololink_module() emits hololink_<UUID>.so under the build
# tree's module dir; partner modules use the same helper.

add_hololink_module(
    NAME hsb_lite
    UUID 889b7ce3-65a5-4247-8b05-4ff1904c3359
    SOURCES
        module_entry.cpp
)

target_include_directories(hsb_lite
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        $<TARGET_PROPERTY:hololink::module_core,INCLUDE_DIRECTORIES>)

target_link_libraries(hsb_lite
    PRIVATE
        hololink::module_core)

# Public header target the supplement type ships through. Out-of-tree
# applications and per-module python sub-packages link this when they
# need to call HsbLiteInterfaceV1::get_service(metadata) or hold a
# typed shared_ptr<HsbLiteInterfaceV1>.
add_library(hololink_hsb_lite_headers INTERFACE)
add_library(hololink::hsb_lite::headers ALIAS hololink_hsb_lite_headers)
target_include_directories(hololink_hsb_lite_headers
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(hololink_hsb_lite_headers
    INTERFACE
        hololink::module_headers)
target_compile_features(hololink_hsb_lite_headers INTERFACE cxx_std_17)

# Python bindings: NOT YET ported to nanobind, unlike its hololink_module Python-extension
# siblings (core, operators, vb1940, taurotech_da326 -- see python/hololink_module/CMakeLists.txt
# and hololink_module/host/CMakeLists.txt's own Python comment). This pybind11_add_module()
# block (holoscan::pybind11, not exported by the installed HSDK5 holoscan-sdk) would otherwise
# break HOLOLINK_BUILD_MODULE's configure step the same way the other 4 did, so it's disabled
# here rather than fixed -- no app in examples/*.py imports hololink_module.hsb_lite, so this
# is deliberately deferred (same treatment as CheckCrcOp), not an oversight. Port
# python/hsb_lite_py.cpp into the consolidated _hololink_module.cpp the same way the other 4
# files were if a caller ever needs hololink_module.hsb_lite from Python.
