# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# PVA CRC Python wrapper, nanobind (HSDK5; this file was pybind11 pre-rewrite -- see
# python/hololink/_hololink.cpp's own header comment for the project-wide switch). Only
# reached via the normal top-level build (python/hololink/operators/CMakeLists.txt's
# HOLOLINK_BUILD_PVA-gated add_subdirectory(pva_crc)), so -- unlike the pre-rewrite file --
# this has no standalone project()/find_package(holoscan)/CMAKE_MODULE_PATH scaffolding of its
# own: nanobind_add_module and every CMAKE_MODULE_PATH entry are already available from the
# parent scope (python/hololink/CMakeLists.txt's own include(hololink_deps/nanobind)), the same
# as every other operator's CMakeLists.txt in this tree.

nanobind_add_module(pva_crc NB_STATIC STABLE_ABI pva_crc_pybind.cpp)

# Include path for <hololink/operators/pva_crc/pva_crc.hpp>.
target_include_directories(pva_crc
  PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(pva_crc_lib STATIC IMPORTED)

# PVA SDK 2.9 is REQUIRED for PVA CRC functionality. Allow override via PVA_SDK_PATH
# environment variable.
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

# Location in the build directory where the library will be used.
set(PVA_CRC_LIB_DEST "${CMAKE_CURRENT_BINARY_DIR}/lib/${PVA_CRC_LIB_NAME}")
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

add_custom_target(prepare_pva_dependencies ALL
  DEPENDS ${PVA_CRC_LIB_DEST} "${CMAKE_CURRENT_BINARY_DIR}/lib/prepared.stamp"
)

add_dependencies(pva_crc prepare_pva_dependencies)

set_target_properties(pva_crc_lib PROPERTIES IMPORTED_LOCATION ${PVA_CRC_LIB_DEST})

# Find cuPVA host library (PVA SDK 2.9).
set(PVA_SDK_LIB_DIR "${PVA_SDK_BASE}/lib/aarch64-linux-gnu")
set(PVA_SDK_LIB_DIR_ALT "${PVA_SDK_BASE}/lib")

find_library(CUPVAHOST_LIB
  NAMES libcupva_host.so.2.9
  PATHS ${PVA_SDK_LIB_DIR} ${PVA_SDK_LIB_DIR_ALT}
  REQUIRED
  NO_DEFAULT_PATH)

find_library(CUPVAHOST_UTILS_LIB
  NAMES libcupva_host_utils.so.2.9
  PATHS ${PVA_SDK_LIB_DIR} ${PVA_SDK_LIB_DIR_ALT}
  REQUIRED
  NO_DEFAULT_PATH)

# Find NvSciBuf library required by CUPVA.
find_library(NVSCIBUF_LIB
  NAMES nvscibuf
  PATHS /usr/lib/aarch64-linux-gnu/nvidia /usr/lib/aarch64-linux-gnu/tegra
  REQUIRED)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(pva_crc
  PRIVATE
  pva_crc_lib
  ${CUPVAHOST_LIB}
  ${CUPVAHOST_UTILS_LIB}
  ${NVSCIBUF_LIB}
  CUDA::cudart
  CUDA::cuda_driver
)
