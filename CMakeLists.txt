# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.24)

# Options
option(HOLOLINK_BUILD_PYTHON "Build Hololink Python Bindings" ON)
option(HOLOLINK_BUILD_TESTS "Build Hololink Tests" ON)
option(HOLOLINK_BUILD_TOOLS "Build Hololink Tools" ON)
option(HOLOLINK_BUILD_EXAMPLES "Build Hololink Examples" ON)
# Declared here (as well as in hololink_module/CMakeLists.txt, where
# option() honors this cache value as a no-op) so the variable is
# defined before add_subdirectory(examples) — the Linux software-receiver
# example players gate on it, and examples/ is processed before the
# adapter subdirectory.
option(HOLOLINK_MODULE_BUILD_OPERATORS "Build hololink_module Holoscan-coupled operators" ON)
option(HOLOLINK_BUILD_ONLY_NATIVE "Build only native (i.e. C++) components" OFF)
option(HOLOLINK_BUILD_ARGUS_ISP "Build Argus ISP support" OFF)
option(HOLOLINK_BUILD_SIPL "Build NvSIPL capture components" OFF)
option(HOLOLINK_BUILD_EMULATOR "Build HSB Emulator components" ON)
option(HOLOLINK_BUILD_FUSA "Build NvFusa capture components" OFF)
option(HOLOLINK_BUILD_PVA "Build PVA CRC components" OFF)
option(HOLOLINK_BUILD_CPACK "Build DEB package" OFF)
option(HOLOLINK_BUILD_EXPORT "Install Hololink targets" ON)
# HOLOLINK_BUILD_ROCE is declared just below, after CMAKE_MODULE_PATH is
# set, because its default is probed via find_package(ibverbs).
option(HOLOLINK_BUILD_GPUNETIO_ROCE "Build DOCA GPUNetIO RoCE transceiver (requires DOCA gpunetio/verbs)" ON)
option(HOLOLINK_BUILD_SOCKET "Build Hololink with Linux Socket support" ON)
option(HOLOLINK_ROCE_USE_GPU_VRAM "Prefer GPU VRAM for RoCE receiver frame buffers (requires GPUDirect RDMA). When OFF, always use pinned host memory, which avoids IOMMU write-access faults at the cost of an extra PCIe hop for GPU access." ON)

if (HOLOLINK_BUILD_PYTHON AND NOT HOLOLINK_BUILD_EMULATOR)
  message(FATAL_ERROR "If building Python bindings, HOLOLINK_BUILD_EMULATOR must be ON")
endif()

file(READ "VERSION" HOLOLINK_VERSION)
string(STRIP ${HOLOLINK_VERSION} HOLOLINK_VERSION)
string(REGEX MATCH "^([0-9]+\.[0-9]+\.[0-9]+)" HOLOLINK_VERSION_NUM ${HOLOLINK_VERSION})

project(hololink
  LANGUAGES CXX
  VERSION ${HOLOLINK_VERSION_NUM})

# To discover our custom CMake modules
list(APPEND CMAKE_MODULE_PATH ${hololink_SOURCE_DIR}/cmake)

# Default RoCE support on only when ibverbs is actually available to link
# against. python/setup.py and docker/build.sh pass -DHOLOLINK_BUILD_ROCE
# explicitly (which overrides this default); probing the package here lets
# a plain `cmake` configure on a host without libibverbs-dev (e.g. AGX
# Thor) Do The Right Thing instead of failing the later
# find_package(ibverbs REQUIRED). Needs CMAKE_MODULE_PATH (set just above)
# so Findibverbs.cmake resolves.
find_package(ibverbs QUIET)
option(HOLOLINK_BUILD_ROCE "Build Hololink with RoCE support" ${ibverbs_FOUND})

# Find globally required dependencies
if (NOT HOLOLINK_BUILD_ONLY_NATIVE)
  # Initialize CUDA architecture target list by precedence:
  #     -DCMAKE_CUDA_ARCHITECTURES > $ENV{CUDAARCHS} > `native`
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES "native")
  endif()
  enable_language(CUDA)
  message(STATUS "hololink: initialized CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

  # Find Holoscan if it is not already defined for source build with FetchContent
  if(NOT holoscan_SOURCE_DIR)
    find_package(holoscan 4.0 REQUIRED CONFIG PATHS "/opt/nvidia/holoscan")
  endif()

  # enabled trace and debug log messages in debug build only
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_definitions(HOLOSCAN_LOG_ACTIVE_LEVEL=2)
  else()
    add_compile_definitions(HOLOSCAN_LOG_ACTIVE_LEVEL=0)
  endif()
endif()

# if HSDK is present/used, fmt should discover HSDK's version, which is cleaner
# if fmt discovery is after HSDK; especially after HSDK 4.0 provides fmt targets downstream.
include(hololink_deps/fmt)

if(NOT CMAKE_INSTALL_LIBDIR)
  set(CMAKE_INSTALL_LIBDIR lib)
endif()
if(NOT CMAKE_INSTALL_BINDIR)
  set(CMAKE_INSTALL_BINDIR bin)
endif()
if(NOT CMAKE_INSTALL_INCLUDEDIR)
  set(CMAKE_INSTALL_INCLUDEDIR include)
endif()

# Add subdirectories based on the build options
add_subdirectory(src)
if(HOLOLINK_BUILD_TESTS)
  enable_testing()
endif()
if(HOLOLINK_BUILD_TOOLS)
  add_subdirectory(tools)
endif()
if(HOLOLINK_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
if(NOT HOLOLINK_BUILD_ONLY_NATIVE)
  # The adapter wraps the legacy `src/hololink/operators/` tree
  # (currently for RoceReceiver, via module/core's RoceReceiverImplV1)
  # which is itself gated on the native-only flag, so the adapter is
  # also off in native-only builds. Container builds that produce CLI
  # tools (docker/Dockerfile.demo) set HOLOLINK_BUILD_ONLY_NATIVE=ON
  # and install nothing from the adapter.
  add_subdirectory(hololink_module)
  if(HOLOLINK_BUILD_TESTS)
    add_subdirectory(tests)
  endif()
  if(HOLOLINK_BUILD_PYTHON)
    add_subdirectory(python)
  endif()
add_subdirectory(scripts)
endif()

# Add packaging
if(HOLOLINK_BUILD_CPACK)
  include(cmake/HololinkCPack.cmake)
endif()

# Add export
if(HOLOLINK_BUILD_EXPORT)
  include(cmake/HololinkExport.cmake)
endif()
