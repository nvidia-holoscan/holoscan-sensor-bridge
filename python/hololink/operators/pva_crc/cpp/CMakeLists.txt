#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#

cmake_minimum_required(VERSION 3.20)
project(pva_crc_wrapper LANGUAGES CXX)

# PVA CRC Python Wrapper using PyBind11
# This CMakeLists.txt is only included when HOLOLINK_BUILD_PVA=ON
# PVA SDK 2.9 is REQUIRED for building PVA CRC components

# Add the hololink cmake modules to the path
# From python/hololink/operators/pva_crc/cpp -> ../../../../../cmake (go up 5 levels to hololink root)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../cmake")

# Find Python and pybind11 (using hololink's pybind11 setup)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include(hololink_deps/pybind11)

# Find Holoscan SDK (available in hololink-demo container)
find_package(holoscan REQUIRED CONFIG PATHS "/opt/nvidia/holoscan")

# PyBind11 Python module
pybind11_add_module(pva_crc
  pva_crc_pybind.cpp
)

# Include path for <hololink/operators/pva_crc/pva_crc.hpp>
set(PVA_CRC_SRC_DIR "${CMAKE_SOURCE_DIR}/src")
if(NOT EXISTS "${PVA_CRC_SRC_DIR}/hololink/operators/pva_crc/pva_crc.hpp")
  # Standalone configure from cpp: repo root is 5 levels up (cpp -> pva_crc -> operators -> hololink -> python -> root)
  set(PVA_CRC_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../src")
endif()
target_include_directories(pva_crc
  PRIVATE
    ${PVA_CRC_SRC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(pva_crc_lib STATIC IMPORTED)

# PVA SDK 2.9 is REQUIRED for PVA CRC functionality
# Allow override via PVA_SDK_PATH environment variable
if(DEFINED ENV{PVA_SDK_PATH} AND EXISTS "$ENV{PVA_SDK_PATH}")
    set(PVA_SDK_BASE "$ENV{PVA_SDK_PATH}")
    message(STATUS "Using PVA SDK path: ${PVA_SDK_BASE}")
elseif(EXISTS "/opt/nvidia/pva-sdk-2.9")
    set(PVA_SDK_BASE "/opt/nvidia/pva-sdk-2.9")
    message(STATUS "PVA SDK 2.9 found at ${PVA_SDK_BASE}")
else()
    message(FATAL_ERROR
        "PVA SDK 2.9 is REQUIRED for PVA CRC functionality.\n"
        "/opt/nvidia/pva-sdk-2.9 was not found.\n"
        "Set PVA_SDK_PATH environment variable to specify custom location.\n"
        "PVA SDK installation guide: https://docs.nvidia.com/pva/sdk/2.9.0/installation.html")
endif()

set(PVA_CRC_LIB_NAME "libpva_2.9_crc.a")

# Define the location in the build directory where the library will be used
set(PVA_CRC_LIB_DEST "${CMAKE_CURRENT_BINARY_DIR}/lib/${PVA_CRC_LIB_NAME}")

# Define the destination path in the build directory for allowlist
set(CUPVA_ALLOWLIST_DEST "${CMAKE_CURRENT_BINARY_DIR}/lib/cupva_allowlist_pva_crc_2.9")
set(PVA_CRC_LIB_URL "https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/pva/crc/v2.9/libpva_2.9_crc.a")
set(CUPVA_ALLOWLIST_URL "https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/pva/crc/v2.9/cupva_allowlist_pva_crc_2.9")

# Prepare dependencies: download library if missing, then verify.
# OUTPUT must include both the .a and stamp so Ninja has a rule to produce the .a when the linker needs it.
add_custom_command(
  OUTPUT ${PVA_CRC_LIB_DEST} "${CMAKE_CURRENT_BINARY_DIR}/lib/prepared.stamp"
  COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --cyan "Preparing PVA CRC dependencies..."
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/lib"
  COMMAND ${CMAKE_COMMAND}
    -D PVA_CRC_LIB_DEST="${PVA_CRC_LIB_DEST}"
    -D CUPVA_ALLOWLIST_DEST="${CUPVA_ALLOWLIST_DEST}"
    -D PVA_CRC_LIB_URL="${PVA_CRC_LIB_URL}"
    -P "${CMAKE_CURRENT_LIST_DIR}/PreparePVADependencies.cmake"
  COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_CURRENT_BINARY_DIR}/lib/prepared.stamp"
  COMMENT "Preparing ${PVA_CRC_LIB_NAME}"
)

# Define a custom target that depends on the custom command outputs (so Ninja builds them before pva_crc links)
add_custom_target(prepare_pva_dependencies ALL
  DEPENDS ${PVA_CRC_LIB_DEST} "${CMAKE_CURRENT_BINARY_DIR}/lib/prepared.stamp"
)

# Ensure the custom target is built before the wrapper
add_dependencies(pva_crc prepare_pva_dependencies)

# Update the IMPORTED_LOCATION to the new path in the build directory
set_target_properties(pva_crc_lib PROPERTIES IMPORTED_LOCATION ${PVA_CRC_LIB_DEST})

# Find cuPVA host library (PVA SDK 2.9)
set(PVA_SDK_LIB_DIR "${PVA_SDK_BASE}/lib/aarch64-linux-gnu")
set(PVA_SDK_LIB_DIR_ALT "${PVA_SDK_BASE}/lib")

find_library(CUPVAHOST_LIB
  NAMES libcupva_host.so.2.9
  PATHS
    ${PVA_SDK_LIB_DIR}
    ${PVA_SDK_LIB_DIR_ALT}
  REQUIRED
  NO_DEFAULT_PATH)

find_library(CUPVAHOST_UTILS_LIB
  NAMES libcupva_host_utils.so.2.9
  PATHS
    ${PVA_SDK_LIB_DIR}
    ${PVA_SDK_LIB_DIR_ALT}
  REQUIRED
  NO_DEFAULT_PATH)

# Find NvSciBuf library required by CUPVA
find_library(NVSCIBUF_LIB
  NAMES nvscibuf
  PATHS /usr/lib/aarch64-linux-gnu/nvidia /usr/lib/aarch64-linux-gnu/tegra
  REQUIRED)

# Find CUDA libraries (same as rest of hololink: CUDAToolkit works in Docker/sbsa and host)
find_package(CUDAToolkit REQUIRED)

# Link PVA libraries to pybind11 module
target_link_libraries(pva_crc
  PUBLIC
  pva_crc_lib
  ${CUPVAHOST_LIB}
  ${CUPVAHOST_UTILS_LIB}
  ${NVSCIBUF_LIB}
  CUDA::cudart
  CUDA::cuda_driver
)
