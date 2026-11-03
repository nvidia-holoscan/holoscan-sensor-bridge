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
# The Holoscan operators under src/hololink/operators. Every example links some of them, so
# turning them off also requires HOLOLINK_BUILD_EXAMPLES=OFF (the tools-only build in
# docker/Dockerfile.demo does exactly that; the wheel build ships the examples).
option(HOLOLINK_BUILD_OPERATORS "Build Hololink Holoscan operators (required by the examples)" ON)
if(HOLOLINK_BUILD_EXAMPLES AND NOT HOLOLINK_BUILD_OPERATORS)
  message(FATAL_ERROR "HOLOLINK_BUILD_EXAMPLES requires HOLOLINK_BUILD_OPERATORS; pass -DHOLOLINK_BUILD_EXAMPLES=OFF as well")
endif()
# Declared here (as well as in hololink_module/CMakeLists.txt, where
# option() honors this cache value as a no-op) so the variable is
# defined before add_subdirectory(examples) — the Linux software-receiver
# example players gate on it, and examples/ is processed before the
# adapter subdirectory.
option(HOLOLINK_MODULE_BUILD_OPERATORS "Build hololink_module Holoscan-coupled operators" ON)
option(HOLOLINK_BUILD_ONLY_NATIVE "Build only native (i.e. C++) components" OFF)
option(HOLOLINK_BUILD_ARGUS_ISP "Build Argus ISP support" OFF)
option(HOLOLINK_BUILD_EMULATOR "Build the HSB emulator (C++ libraries and examples; with HOLOLINK_BUILD_PYTHON also the hololink.emulation module)" OFF)
option(HOLOLINK_BUILD_UNPORTED_OPERATORS "Build operators not yet ported to HSDK5: AudioPacketizerOp (legacy holoscan::gxf::Entity API)." OFF)
option(HOLOLINK_BUILD_MODULE "Build hololink_module" ON)
option(HOLOLINK_BUILD_FUSA "Build NvFusa capture components" OFF)
option(HOLOLINK_BUILD_PVA "Build PVA CRC components" OFF)
option(HOLOLINK_BUILD_CPACK "Build DEB package" OFF)
option(HOLOLINK_BUILD_EXPORT "Install Hololink targets" ON)
# HOLOLINK_BUILD_ROCE is declared just below, after CMAKE_MODULE_PATH is
# set, because its default is probed via find_package(ibverbs).
option(HOLOLINK_BUILD_GPUNETIO_ROCE "Build DOCA GPUNetIO RoCE transceiver (requires DOCA gpunetio/verbs)" ON)
option(HOLOLINK_BUILD_SOCKET "Build Hololink with Linux Socket support" ON)
# Everything PCIe: the hsb_pcie module, the module-native PcieReceiver, the
# hololink_pcie_identity tool, the IMX258 driver (the sensor on the
# CertusPro-NX Versa PCIe board), the hololink_pcie kernel module and their
# tests. PcieReceiverOp and module_pcie_imx258_player are not yet ported to
# HSDK5 and are not built. Off by default: it needs a bound PCIe endpoint (or
# its kernel headers) to be of any use.
option(HOLOLINK_BUILD_PCIE "Build the PCIe transport (hsb_pcie module, PcieReceiver, tools, tests, examples; the hololink_pcie kernel driver is built from its own tree)" OFF)
option(HOLOLINK_ROCE_USE_GPU_VRAM "Prefer GPU VRAM for RoCE receiver frame buffers (requires GPUDirect RDMA). When OFF, always use pinned host memory, which avoids IOMMU write-access faults at the cost of an extra PCIe hop for GPU access." ON)

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
    find_package(holoscan 5.0 REQUIRED CONFIG PATHS "/opt/nvidia/holoscan")
  endif()

  # The Holoscan Vision module (BayerDemosaicOp et al.) ships as its own standalone CMake
  # package, installed alongside holoscan under the same prefix (lib/cmake/holoscan-vision/);
  # for FetchContent/source builds the target is already defined. Optional: not every
  # holoscan-sdk install has it. Pinned to 5.0 like the main holoscan package above, so an
  # incompatible install fails cleanly here instead of surfacing later as a confusing
  # BayerDemosaicOp header/link mismatch.
  if(NOT TARGET holoscan::vision)
    find_package(holoscan-vision 5.0 QUIET CONFIG
      PATHS "${holoscan_DIR}/../holoscan-vision"
            "/opt/nvidia/holoscan/lib/cmake/holoscan-vision")
  endif()
  if(TARGET holoscan::vision)
    set(HOLOLINK_HAS_BAYER_DEMOSAIC ON)
    message(STATUS "holoscan::vision found -- building BayerDemosaicOp-dependent components")
  else()
    set(HOLOLINK_HAS_BAYER_DEMOSAIC OFF)
    message(WARNING "holoscan-vision not found -- BayerDemosaicOp-dependent components will be "
                    "skipped; pass -Dholoscan-vision_DIR=<path>/lib/cmake/holoscan-vision")
  endif()

  # HoloViz library (holoscan::visualization::holoviz) from the SDK's visualization component;
  # SubFrameVisualizerOp drives it directly.
  if(NOT TARGET holoscan::visualization::holoviz)
    find_package(holoscan-visualization QUIET CONFIG
      PATHS "${holoscan_DIR}"
            "${holoscan_DIR}/../holoscan-visualization"
            "/opt/nvidia/holoscan/lib/cmake/holoscan"
            "/opt/nvidia/holoscan/lib/cmake/holoscan-visualization")
    if(holoscan-visualization_FOUND)
      message(STATUS "holoscan-visualization found -- HoloViz-dependent operators enabled")
    else()
      message(STATUS "holoscan-visualization not found -- HoloViz-dependent operators will be skipped")
    endif()
  endif()

  # Newer HSDK5 installs split the schema runtime into holoscan::schema (needed for the generated
  # tensor header); older installs carry it inside holoscan::core. Provide an empty stand-in so the
  # same target_link_libraries() lines work against both.
  if(NOT TARGET holoscan::schema)
    add_library(holoscan::schema INTERFACE IMPORTED)
  endif()

  # The Python bindings hand C++ objects to Holoscan operators through the HSDK5 constructor-handle
  # capsule protocol, implemented by the header-only holoscan/python/constructor_handle_capsule.hpp
  # from holoscan-sdk main-5x (public/include/holoscan/python/). The HSDK 5.0.0 release debs and
  # wheels ship the holoscan/authoring/constructor_handle.hpp it builds on but not the capsule
  # header itself, so python/hsdk5_compat/ carries a byte-identical copy. Use the
  # SDK's own header when the install provides one and fall back to the copy otherwise.
  # TODO(HSDK5): drop python/hsdk5_compat/ once the HSDK debs ship the header.
  get_target_property(_hololink_holoscan_includes holoscan::core INTERFACE_INCLUDE_DIRECTORIES)
  if(NOT _hololink_holoscan_includes)
    set(_hololink_holoscan_includes "")
  endif()
  find_path(HOLOLINK_HSDK_CAPSULE_HEADER_DIR
    NAMES holoscan/python/constructor_handle_capsule.hpp
    HINTS ${_hololink_holoscan_includes} "/opt/nvidia/holoscan/include"
    NO_CACHE)
  add_library(hololink_hsdk5_compat INTERFACE)
  add_library(hololink::hsdk5_compat ALIAS hololink_hsdk5_compat)
  if(HOLOLINK_HSDK_CAPSULE_HEADER_DIR)
    message(STATUS "holoscan/python/constructor_handle_capsule.hpp: using the Holoscan SDK's copy "
                   "in ${HOLOLINK_HSDK_CAPSULE_HEADER_DIR}")
  else()
    target_include_directories(hololink_hsdk5_compat
      INTERFACE "${hololink_SOURCE_DIR}/python/hsdk5_compat/include")
    message(STATUS "holoscan/python/constructor_handle_capsule.hpp: not shipped by this Holoscan "
                   "SDK install -- using the copy in python/hsdk5_compat/include")
  endif()

  # Gate viz-dependent targets on the presence of a supported HoloViz target.
  # HSDK5 ships HolovizOp as holoscan::visualization::op_holoviz (holoscan-visualization
  # package; earlier main-5x builds exported it as holoscan::operators::holoviz).
  # The HSDK5-ported examples link holoscan::visualization::op_holoviz, so only that target (or
  # its earlier name, aliased to it) turns HOLOLINK_HAS_VIZ on.
  if(TARGET holoscan::operators::holoviz AND NOT TARGET holoscan::visualization::op_holoviz)
    add_library(holoscan::visualization::op_holoviz ALIAS holoscan::operators::holoviz)
  endif()
  if(TARGET holoscan::visualization::op_holoviz)
    set(HOLOLINK_HAS_VIZ ON)
    message(STATUS "HoloViz found -- building viz-dependent components")
  else()
    set(HOLOLINK_HAS_VIZ OFF)
    message(STATUS "HoloViz not found -- viz-dependent components will be skipped")
  endif()

  # HolovizOp has no Dear ImGui layer; hololink::ImGuiRenderer drives the HoloViz library's own
  # ImGui backend (holoviz::imgui::Renderer) directly. It and the signal-generator operators
  # whose GUIs it renders are gated on that library target.
  if(TARGET holoscan::visualization::holoviz)
    set(HOLOLINK_HAS_IMGUI ON)
    message(STATUS "holoscan::visualization::holoviz found -- building ImGui-dependent components")
  else()
    set(HOLOLINK_HAS_IMGUI OFF)
    message(STATUS "holoscan::visualization::holoviz not found -- ImGui-dependent components will be skipped")
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
  if(HOLOLINK_BUILD_MODULE)
    add_subdirectory(hololink_module)
  endif()
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
