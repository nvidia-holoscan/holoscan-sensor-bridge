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

# This script prepares PVA CRC dependencies by copying them from the source directory
# to the build directory during the CMake build process.
# PVA SDK 2.9 only.

# Optional source locations: build dir, then src/hololink/operators/pva_crc/lib (C++ implementation tree)
set(PVA_CRC_SRC_LIB_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../src/hololink/operators/pva_crc/lib")
if(NOT DEFINED PVA_CRC_LIB_SRC)
  set(PVA_CRC_LIB_NAME "libpva_2.9_crc.a")
  if(DEFINED PVA_CRC_LIB_DEST AND EXISTS "${PVA_CRC_LIB_DEST}")
    set(PVA_CRC_LIB_SRC "${PVA_CRC_LIB_DEST}")
    message(STATUS "Using ${PVA_CRC_LIB_NAME} from build directory")
  elseif(EXISTS "${PVA_CRC_SRC_LIB_DIR}/${PVA_CRC_LIB_NAME}")
    set(PVA_CRC_LIB_SRC "${PVA_CRC_SRC_LIB_DIR}/${PVA_CRC_LIB_NAME}")
    message(STATUS "Using ${PVA_CRC_LIB_NAME} from source directory (src/hololink/operators/pva_crc/lib)")
  else()
    message(FATAL_ERROR
        "PVA CRC library not found.\n"
        "The library is downloaded during the wheel build when PVA_CRC_LIB_URL is set, or\n"
        "manually download ${PVA_CRC_LIB_NAME} from:\n"
        "https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/pva/crc/v2.9/")
  endif()
endif()

if(NOT EXISTS "${PVA_CRC_LIB_DEST}")
  get_filename_component(LIB_NAME "${PVA_CRC_LIB_SRC}" NAME)
  message(STATUS "Copying ${LIB_NAME} to ${PVA_CRC_LIB_DEST}")
  execute_process(COMMAND ${CMAKE_COMMAND} -E copy
                  "${PVA_CRC_LIB_SRC}"
                  "${PVA_CRC_LIB_DEST}"
                  RESULT_VARIABLE result
                  OUTPUT_QUIET)
  if(NOT result EQUAL "0")
    message(FATAL_ERROR "Error copying ${LIB_NAME} from ${PVA_CRC_LIB_SRC}")
  endif()
endif()

# Allowlist: use build directory or source directory
if(NOT DEFINED CUPVA_ALLOWLIST_SRC)
  if(DEFINED CUPVA_ALLOWLIST_DEST AND EXISTS "${CUPVA_ALLOWLIST_DEST}")
    set(CUPVA_ALLOWLIST_SRC "${CUPVA_ALLOWLIST_DEST}")
    message(STATUS "Using cupva_allowlist_pva_crc_2.9 from build directory")
  elseif(EXISTS "${PVA_CRC_SRC_LIB_DIR}/cupva_allowlist_pva_crc_2.9")
    set(CUPVA_ALLOWLIST_SRC "${PVA_CRC_SRC_LIB_DIR}/cupva_allowlist_pva_crc_2.9")
    message(STATUS "Using cupva_allowlist_pva_crc_2.9 from source directory (src/hololink/operators/pva_crc/lib)")
  endif()
endif()

if(DEFINED CUPVA_ALLOWLIST_SRC AND NOT EXISTS "${CUPVA_ALLOWLIST_DEST}")
  get_filename_component(ALLOWLIST_NAME "${CUPVA_ALLOWLIST_SRC}" NAME)
  message(STATUS "Copying ${ALLOWLIST_NAME} to ${CUPVA_ALLOWLIST_DEST}")
  execute_process(COMMAND ${CMAKE_COMMAND} -E copy
                  "${CUPVA_ALLOWLIST_SRC}"
                  "${CUPVA_ALLOWLIST_DEST}"
                  RESULT_VARIABLE result_allowlist
                  OUTPUT_QUIET)
  if(NOT result_allowlist EQUAL "0")
    message(FATAL_ERROR "Error copying ${ALLOWLIST_NAME}")
  endif()
endif()
