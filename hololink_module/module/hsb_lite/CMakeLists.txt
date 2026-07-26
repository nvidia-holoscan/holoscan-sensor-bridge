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

# Per-module Python sub-package. Builds _hololink_module_hsb_lite.so
# under ${BUILD}/python/hololink_module/hsb_lite/ alongside the
# matching __init__.py so PYTHONPATH=${BUILD}/python python3 -c
# "import hololink_module.hsb_lite" resolves directly out of the
# build tree.
if(HOLOLINK_BUILD_PYTHON)
    include(hololink_deps/pybind11)

    pybind11_add_module(_hololink_module_hsb_lite MODULE
        python/hsb_lite_py.cpp)
    # holoscan::pybind11 pins this extension to the same pybind11 ABI
    # the sibling _hololink_py_module and _hololink_module_operators
    # extensions use, so cross-module type lookups (EnumerationMetadata,
    # Module) hit one shared internals capsule. Without it the pybind11
    # internals tag is `__pybind11_internals_v4_gcc_libstdcpp_cxxabi1016__`
    # instead of holoscan's `__pybind11_internals_v4__`, and the
    # registries split.
    target_link_libraries(_hololink_module_hsb_lite
        PRIVATE
            hololink::hsb_lite::headers
            holoscan::pybind11)
    target_compile_features(_hololink_module_hsb_lite PRIVATE cxx_std_17)
    # Reuse hololink::module_core's PCH — the binding TU transitively pulls
    # in the same framework headers. Visibility is compatible
    # (pybind11_add_module sets hidden by default, matching
    # hololink_module_core).
    if(TARGET hololink_module_core)
        target_precompile_headers(_hololink_module_hsb_lite
            REUSE_FROM hololink_module_core)
    endif()

    set_target_properties(_hololink_module_hsb_lite PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
            "${CMAKE_BINARY_DIR}/python/hololink_module/hsb_lite")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/python/__init__.py"
        "${CMAKE_BINARY_DIR}/python/hololink_module/hsb_lite/__init__.py"
        COPYONLY)
endif()
