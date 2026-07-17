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

if(NOT DEFINED ARCH)
  set(ARCH "${CMAKE_SYSTEM_PROCESSOR}")
endif()
if(NOT ARCH STREQUAL "x86_64" AND NOT ARCH STREQUAL "aarch64")
  message(FATAL_ERROR "DOCA is only supported on x86_64 and aarch64; ARCH is \"${ARCH}\".")
endif()

find_package(PkgConfig REQUIRED)
find_package(CUDAToolkit REQUIRED) # Or check for CMAKE_CUDA_COMPILER
find_library(MLX5_LIB mlx5 REQUIRED)
find_package(ibverbs REQUIRED)
SET(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:/opt/mellanox/dpdk/lib/${ARCH}-linux-gnu/pkgconfig:/opt/mellanox/doca/lib/${ARCH}-linux-gnu/pkgconfig:/opt/mellanox/collectx/lib/${ARCH}-linux-gnu/pkgconfig")
SET(DOCA_PATH /opt/mellanox/doca)
pkg_search_module(DOCA_GPUNETIO REQUIRED doca-gpunetio)
pkg_search_module(DOCA_VERBS REQUIRED doca-verbs)

# Check if DOCA is available
message(STATUS "DOCA found at ${DOCA_PATH}")

add_library(gpu_roce_transceiver STATIC
  gpu_roce_transceiver_op.cpp
  gpu_roce_transceiver.cpp
  gpu_roce_transceiver_doca.cpp
  gpu_roce_transceiver_kernel.cu
)

# doca-gpunetio requires sm_80+. We don't filter CMAKE_CUDA_ARCHITECTURES (explicit user intent
# is respected and an unbuildable arch should fail early at link time) but we warn so the source
# of any subsequent doca-gpunetio link error is obvious. "native" is a CMake 3.24+ sentinel
# resolved at configure time and is skipped by the numeric check.
set(_grt_low_archs "")
if (DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(arch STREQUAL "native")
      message(STATUS "gpu_roce_transceiver: skipped \"native\" arch check against the anticipated doca-gpunetio sm_80 minimum.")
      continue()
    endif()
    string(REGEX REPLACE "^([0-9]+).*" "\\1" arch_num "${arch}")
    if(arch_num LESS 80)
      list(APPEND _grt_low_archs ${arch})
    endif()
  endforeach()
endif()
if(_grt_low_archs)
  # Anticipate sm_80 minimum from doca-gpunetio, but do not enforce it here.
  message(WARNING "gpu_roce_transceiver: Requested CUDA arch(s) ${_grt_low_archs} below the anticipated sm_80 minimum "
    "required by doca-gpunetio; build may fail at link time.")
endif()

# to silence doca api warnings publicly
target_compile_options(gpu_roce_transceiver PUBLIC -DDOCA_ALLOW_EXPERIMENTAL_API=1)

add_library(hololink::operators::gpu_roce_transceiver ALIAS gpu_roce_transceiver)

include(GNUInstallDirs)
target_include_directories(gpu_roce_transceiver
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    ${DOCA_GPUNETIO_INCLUDE_DIRS}
    ${DOCA_VERBS_INCLUDE_DIRS}
)

target_compile_options(gpu_roce_transceiver PRIVATE ${DOCA_GPUNETIO_CFLAGS} ${DOCA_VERBS_CFLAGS})

target_link_directories(gpu_roce_transceiver PUBLIC ${DOCA_GPUNETIO_LIBRARY_DIRS} ${DOCA_VERBS_LIBRARY_DIRS})
target_link_libraries(gpu_roce_transceiver
  PUBLIC
    hololink::operators::base_receiver_op
    holoscan::core
  PRIVATE
    CUDA::cuda_driver
    ${DOCA_GPUNETIO_LIBRARIES} -ldoca_gpunetio -ldoca_common
    ${DOCA_VERBS_LIBRARIES}
)

# Suppress deprecated declarations warning for DOCA APIs
target_compile_options(gpu_roce_transceiver PRIVATE -Wno-deprecated-declarations)

# Set RPATH to include DOCA library directories
set_target_properties(gpu_roce_transceiver PROPERTIES
  INSTALL_RPATH "$ORIGIN:${DOCA_GPUNETIO_LIBRARY_DIRS}:${DOCA_VERBS_LIBRARY_DIRS}"
  BUILD_RPATH "${DOCA_GPUNETIO_LIBRARY_DIRS};${DOCA_VERBS_LIBRARY_DIRS}"
)

# Installation of the gpu_roce_transceiver operator
install(TARGETS gpu_roce_transceiver
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-roce-operators)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/gpu_roce_transceiver_op.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/operators/gpu_roce_transceiver
        COMPONENT hololink-roce-operators)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/gpu_roce_transceiver.cuh
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/operators/gpu_roce_transceiver
        COMPONENT hololink-roce-operators)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/gpu_roce_transceiver_common.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/operators/gpu_roce_transceiver
        COMPONENT hololink-roce-operators)
