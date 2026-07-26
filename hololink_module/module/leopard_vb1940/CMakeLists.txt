# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Leopard VB1940 module. Documented exception to the "modules should
# not use NO_COMPAT_SUFFIX" rule: the Leopard VB1940 implementation
# does not currently publish a compat-id over bootp, so the bare
# `hololink_<UUID>.so` filename is what Adapter::load_module_for
# resolves to for these boards (via its no-compat-id fallback path).
# `module/hsb_lite_2510/` is the other documented exception. Switch
# to the compat-suffixed form once the FPGA reports a compat-id.
#
# Inherits the canonical HSB-shape service chain from
# module_core::HsbLitePublisher via LeopardVb1940Publisher; swaps in
# the Bajoran TS2 oscillator and the LeopardVb1940V1 supplement.

add_hololink_module(
    NAME leopard_vb1940
    UUID f1627640-b4dc-48af-a360-c55b09b3d230
    NO_COMPAT_SUFFIX
    SOURCES
        module_entry.cpp
)

target_include_directories(leopard_vb1940
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        $<TARGET_PROPERTY:hololink::hsb_lite::headers,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:hololink::module_core,INCLUDE_DIRECTORIES>)

target_link_libraries(leopard_vb1940
    PRIVATE
        hololink::module_core)

# Public header target the supplement type ships through. Out-of-tree
# applications and per-module python sub-packages link this when they
# need to call LeopardInterfaceV1::get_service(metadata) or hold a
# typed shared_ptr<LeopardInterfaceV1>.
add_library(hololink_leopard_vb1940_headers INTERFACE)
add_library(hololink::leopard_vb1940::headers ALIAS hololink_leopard_vb1940_headers)
target_include_directories(hololink_leopard_vb1940_headers
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(hololink_leopard_vb1940_headers
    INTERFACE
        hololink::module_headers)
target_compile_features(hololink_leopard_vb1940_headers INTERFACE cxx_std_17)

# Export for external applications: this INTERFACE target is a PUBLIC usage
# requirement of hololink::module::sensors::vb1940, so it must join the same
# HololinkTargets export set for that sensor's imported target to resolve.
include(GNUInstallDirs)
set_target_properties(hololink_leopard_vb1940_headers PROPERTIES
    EXPORT_NAME leopard_vb1940::headers)
install(TARGETS hololink_leopard_vb1940_headers
    EXPORT HololinkTargets
    COMPONENT hololink-module)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/hololink
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT hololink-module)
