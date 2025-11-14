# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

find_package(CUDAToolkit) # Or check for CMAKE_CUDA_COMPILER
if (CUDAToolkit_FOUND)
    enable_language(CUDA)
endif()

set(TARGET_ARCH "arm64")

if(DEFINED ENV{TARGET_ARCH})
  set(TARGET_ARCH "$ENV{TARGET_ARCH}")
endif()

message(STATUS "TARGET_ARCH = ${TARGET_ARCH}")

set(DOCA_PATH "/opt/mellanox/doca")

# Determine DOCA library directory based on TARGET_ARCH
if (TARGET_ARCH STREQUAL "amd64" OR TARGET_ARCH STREQUAL "x86_64")
    set(DOCA_LIB_DIR "${DOCA_PATH}/lib/x86_64-linux-gnu")
elseif (TARGET_ARCH STREQUAL "arm64" OR TARGET_ARCH STREQUAL "aarch64")
    set(DOCA_LIB_DIR "${DOCA_PATH}/lib/aarch64-linux-gnu")
else()
    set(DOCA_LIB_DIR "${DOCA_PATH}/lib")
endif()

# Find DOCA libraries
find_path(DOCA_INCLUDE_DIR doca_verbs.h
 PATHS ${DOCA_PATH}/include
 NO_DEFAULT_PATH
)

find_library(DOCA_VERBS_LIB doca_verbs
  PATHS ${DOCA_LIB_DIR}
  NO_DEFAULT_PATH
)

find_library(DOCA_GPUNETIO_LIB doca_gpunetio
  PATHS ${DOCA_LIB_DIR}
  NO_DEFAULT_PATH
)

find_library(DOCA_COMMON_LIB doca_common
  PATHS ${DOCA_LIB_DIR}
  NO_DEFAULT_PATH
)

# Check if DOCA is available
if(DOCA_INCLUDE_DIR AND DOCA_VERBS_LIB AND DOCA_COMMON_LIB AND DOCA_GPUNETIO_LIB)
  message(STATUS "DOCA found at ${DOCA_PATH}")
  message(STATUS "  DOCA include: ${DOCA_INCLUDE_DIR}")
  message(STATUS "  DOCA verbs lib: ${DOCA_VERBS_LIB}")
  message(STATUS "  DOCA gpunetio lib: ${DOCA_GPUNETIO_LIB}")
  message(STATUS "  DOCA common lib: ${DOCA_COMMON_LIB}")

  add_library(gpu_roce_transceiver STATIC
    gpu_roce_transceiver_op.cpp
    gpu_roce_transceiver.cpp
    gpu_roce_transceiver_doca.cpp
    gpu_roce_transceiver_kernel.cu
  )

  set_property(TARGET gpu_roce_transceiver PROPERTY CUDA_ARCHITECTURES 80 90)

  add_library(hololink::operators::gpu_roce_transceiver ALIAS gpu_roce_transceiver)

  target_include_directories(gpu_roce_transceiver
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
      $<INSTALL_INTERFACE:src>
      ${DOCA_INCLUDE_DIR}
      ${DOCA_SAMPLES_DIR}
  )

  target_link_libraries(gpu_roce_transceiver
    PUBLIC
      hololink::operators::roce_receiver
      hololink::operators::base_receiver_op
      holoscan::core
    PRIVATE
      CUDA::cuda_driver
      ${DOCA_VERBS_LIB}
      ${DOCA_GPUNETIO_LIB}
      ${DOCA_COMMON_LIB}
      ${DOCA_RDMA_LIB}
  )

  # Suppress deprecated declarations warning for DOCA APIs
  target_compile_options(gpu_roce_transceiver PRIVATE -Wno-deprecated-declarations)

  # Set RPATH to include DOCA library directory
  set_target_properties(gpu_roce_transceiver PROPERTIES
    INSTALL_RPATH "$ORIGIN:${DOCA_LIB_DIR}"
    BUILD_RPATH "${DOCA_LIB_DIR}"
  )

# Installation of the roce_receiver operator
install(TARGETS gpu_roce_transceiver
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-roce-operators)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/gpu_roce_transceiver_op.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/operators/gpu_roce_transceiver
        COMPONENT hololink-roce-operators)
else()
  message(WARNING "DOCA not found at /opt/doca. DOCA RoCE receiver will not be built.")
endif()
