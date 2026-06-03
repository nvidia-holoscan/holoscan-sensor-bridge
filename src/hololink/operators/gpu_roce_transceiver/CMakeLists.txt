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

enable_language(CUDA)
# Check if DOCA is available
message(STATUS "DOCA found at ${DOCA_PATH}")

add_library(gpu_roce_transceiver STATIC
  gpu_roce_transceiver_op.cpp
  gpu_roce_transceiver.cpp
  gpu_roce_transceiver_doca.cpp
  gpu_roce_transceiver_kernel.cu
)

set(LOCAL_CUDA_ARCHITECTURES "")
if (DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  # Keep only architectures >= 80; compare numeric part only (values may be e.g. 80-virtual, 70-real)
  # architectures < 70 will fail to build/link with doca-gpunetio. doca gpunetio development suggests minimum 80
  foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX REPLACE "^([0-9]+).*" "\\1" arch_num "${arch}")
    if(NOT arch_num LESS 80)
      list(APPEND LOCAL_CUDA_ARCHITECTURES ${arch})
    else()
      message(STATUS "gpu_roce_transceiver: ${arch} compute capability not used when building gpu_roce_transceiver. must be at least 80")
    endif()
  endforeach()
endif()

# Default to compute_80 if empty for compatibility across target platforms and doca-gpunetio
if(LOCAL_CUDA_ARCHITECTURES STREQUAL "")
  list(APPEND LOCAL_CUDA_ARCHITECTURES 80-virtual)
  message(STATUS "gpu_roce_transceiver: adding 80-virtual to CUDA_ARCHITECTURES for a minimally compatible build")
endif()

# If CUDA_NATIVE_ARCH is set, append it to LOCAL_CUDA_ARCHITECTURES; else use -arch=native
if(NOT "$ENV{CUDA_NATIVE_ARCH}" STREQUAL "")
  list(APPEND LOCAL_CUDA_ARCHITECTURES "$ENV{CUDA_NATIVE_ARCH}")
else()
  # Force native architecture for optimized performance on platform if available
  # This can actually fail if container nvcc and host driver are different versions
  # in which case nvcc will default to a significantly older compute capability, which may fail for gpu_roce_transceiver.
  # if this happens and result is not desired, the build should set CUDA_NATIVE_ARCH in the environment to the desired/native architecture
  target_compile_options(gpu_roce_transceiver PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -arch=native
    >)
endif()
message(STATUS "gpu_roce_transceiver: build architectures ${LOCAL_CUDA_ARCHITECTURES} and native")
set_property(TARGET gpu_roce_transceiver PROPERTY CUDA_ARCHITECTURES ${LOCAL_CUDA_ARCHITECTURES})

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
