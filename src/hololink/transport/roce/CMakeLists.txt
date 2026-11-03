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

# Build this directory directly for a CUDA-free CPU RoCE transport, or include
# it with add_subdirectory() in a full Hololink build.
#
#   cmake -S src/hololink/transport/roce -B roce-build
#   cmake --build roce-build --target hololink_transport_roce
#   HololinkRoce_DIR=<roce-build>/HololinkRoce
string(COMPARE EQUAL
  "${CMAKE_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
  HOLOLINK_ROCE_TRANSPORT_STANDALONE)

if(HOLOLINK_ROCE_TRANSPORT_STANDALONE)
  cmake_minimum_required(VERSION 3.24)
  get_filename_component(
    HOLOLINK_SOURCE_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../.."
    ABSOLUTE)
  if(NOT EXISTS "${HOLOLINK_SOURCE_ROOT}/VERSION")
    message(FATAL_ERROR
      "The standalone RoCE transport must remain inside a Hololink source tree")
  endif()

  file(READ "${HOLOLINK_SOURCE_ROOT}/VERSION" HOLOLINK_VERSION)
  string(STRIP "${HOLOLINK_VERSION}" HOLOLINK_VERSION)
  string(REGEX MATCH
    "^([0-9]+\\.[0-9]+\\.[0-9]+)"
    HOLOLINK_VERSION_NUM
    "${HOLOLINK_VERSION}")
  project(HololinkRoceTransport
    LANGUAGES CXX
    VERSION "${HOLOLINK_VERSION_NUM}")

  if(NOT CMAKE_INSTALL_LIBDIR)
    set(CMAKE_INSTALL_LIBDIR lib)
  endif()
  if(NOT CMAKE_INSTALL_INCLUDEDIR)
    set(CMAKE_INSTALL_INCLUDEDIR include)
  endif()
endif()

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The RoCE transport is supported only on Linux")
endif()

# Use the leaf finder so ibverbs::ibverbs exists even after the umbrella's
# variable-only probe. Prepend so this module wins over cmake/Findibverbs.cmake.
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(ibverbs REQUIRED MODULE)
if(NOT TARGET ibverbs::ibverbs)
  message(FATAL_ERROR
    "The RoCE transport requires the ibverbs::ibverbs imported target")
endif()

# ibv_reg_dmabuf_mr is a hard dependency (rdma-core >= 34), including host-only
# builds: the translation unit references the symbol unconditionally. Old
# headers fail at compile; a newer header with an older library fails at link.
# Building the transport requires rdma-core 34 or newer; an installed archive
# does not re-run this source-tree check.
include(CheckSymbolExists)
include(CMakePushCheckState)
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_LIBRARIES ibverbs::ibverbs)
check_symbol_exists(ibv_reg_dmabuf_mr "infiniband/verbs.h"
  HOLOLINK_HAVE_IBV_REG_DMABUF_MR)
cmake_pop_check_state()
if(NOT HOLOLINK_HAVE_IBV_REG_DMABUF_MR)
  message(FATAL_ERROR
    "The RoCE transport requires ibv_reg_dmabuf_mr from rdma-core 34 or newer")
endif()

find_package(Threads REQUIRED)

add_library(hololink_transport_roce STATIC
  cpu_roce_transceiver.cpp
  cpu_roce_transceiver.hpp
  cpu_roce_transceiver_internal.hpp
  host_trailer_key.cpp
  host_trailer_key.hpp
  ibv.cpp
  string_utils.hpp
)
add_library(hololink::transport::roce ALIAS hololink_transport_roce)

set_property(TARGET hololink_transport_roce PROPERTY POSITION_INDEPENDENT_CODE ON)

# The public API and implementation require C++17.
target_compile_features(hololink_transport_roce PUBLIC cxx_std_17)

target_include_directories(hololink_transport_roce
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Implementation-only: the public header does not include verbs.h. CMake still
# exports a $<LINK_ONLY:ibverbs::ibverbs> requirement for this static archive.
target_link_libraries(hololink_transport_roce
  PRIVATE
    ibverbs::ibverbs
    Threads::Threads
)

# Optional. The header supplies the inline MR-list helpers. libmlx5 itself is
# loaded at run time, so a missing library declines to the device-copy path
# and this static archive does not export an absolute libmlx5 path.
find_path(HOLOLINK_MLX5DV_INCLUDE_DIR infiniband/mlx5dv.h)
if(HOLOLINK_MLX5DV_INCLUDE_DIR)
  target_compile_definitions(hololink_transport_roce PRIVATE HOLOLINK_HAS_MLX5DV=1)
  target_include_directories(hololink_transport_roce PRIVATE ${HOLOLINK_MLX5DV_INCLUDE_DIR})
  target_link_libraries(hololink_transport_roce PRIVATE ${CMAKE_DL_LIBS})
  message(STATUS "mlx5 direct verbs header found — device metadata can use an indirect mkey")
else()
  message(STATUS "mlx5 direct verbs header not found — device metadata uses the copy path")
endif()

# Export the transport without the CUDA-dependent Hololink targets.
set_target_properties(hololink_transport_roce PROPERTIES EXPORT_NAME transport::roce)
install(TARGETS hololink_transport_roce
        EXPORT HololinkRoceTransportTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-roce-transport)

install(FILES cpu_roce_transceiver.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/transport/roce
        COMPONENT hololink-roce-transport)

if(HOLOLINK_ROCE_TRANSPORT_STANDALONE OR HOLOLINK_BUILD_EXPORT)
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/HololinkRoceExport.cmake")
endif()
