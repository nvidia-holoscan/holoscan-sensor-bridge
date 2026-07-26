# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Top-level CMake for the hololink_module tree. Configures the
# framework targets (hololink::module_headers, ::module_runtime,
# ::adapter) and registers the add_hololink_module() helper for
# building per-board module .so files.

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(HololinkModule)

# Default ON when Holoscan is available; partner builds without
# Holoscan configure with -DHOLOLINK_MODULE_BUILD_OPERATORS=OFF.
option(HOLOLINK_MODULE_BUILD_OPERATORS
    "Build hololink_module Holoscan-coupled operators"
    ON)

# Order matters: module/* must be processed before host/sensors so
# the per-board supplement-header INTERFACE targets
# (hololink::hsb_lite::headers, hololink::leopard_vb1940::headers) exist
# by the time the per-sensor camera drivers link against them. host/
# itself stays first because it defines hololink::module_headers,
# which everything below depends on.
add_subdirectory(host)
add_subdirectory(module/core)
add_subdirectory(module/hsb_lite)
add_subdirectory(module/hsb_lite_2510)
add_subdirectory(module/leopard_vb1940)
add_subdirectory(module/taurotech_da326)
add_subdirectory(module/tutorial)
add_subdirectory(host/operators)
add_subdirectory(host/sensors)
add_subdirectory(python)
add_subdirectory(tools)

# Install supplement `.so` files (hololink_<UUID>.so) built by
# add_hololink_module() — they're staged under
# ${CMAKE_BINARY_DIR}/lib/hololink/modules/ and need to land at a
# stable path that Adapter::resolved_module_directories()'s baked-in
# defaults point to. Unconditional (no HOLOLINK_BUILD_PYTHON gate):
# the C++ install layout doesn't change when Python is disabled.
include(GNUInstallDirs)
install(
    DIRECTORY "${CMAKE_BINARY_DIR}/lib/hololink/modules/"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/hololink/modules"
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    DIRECTORY_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    COMPONENT "hololink-modules")

# Install the staged Python tree alongside the legacy `hololink` package
# so `pip install ./python/` (cmake_build_extension) picks it up. Each
# pybind extension + per-package __init__.py is staged into
# ${CMAKE_BINARY_DIR}/python/hololink_module/... by the subdirectories
# above; this single install rule copies the whole tree.
#
# HOLOLINK_PYTHON_INSTALL_DIR is defined in python/CMakeLists.txt but
# that subdirectory is processed after this one. We define the cache
# variable here with the same default so an in-tree `cmake --install`
# without -D still works; if it was already set externally (e.g. via
# python/setup.py's -DHOLOLINK_PYTHON_INSTALL_DIR=.) CACHE PATH leaves
# the existing value intact.
if(HOLOLINK_BUILD_PYTHON)
    set(HOLOLINK_PYTHON_INSTALL_DIR "python"
        CACHE PATH "Installation directory for hololink Python packages")
    install(
        DIRECTORY "${CMAKE_BINARY_DIR}/python/hololink_module/"
        DESTINATION "${HOLOLINK_PYTHON_INSTALL_DIR}/hololink_module"
        FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        DIRECTORY_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        COMPONENT "hololink-python_libs"
        PATTERN "__pycache__" EXCLUDE)
endif()
