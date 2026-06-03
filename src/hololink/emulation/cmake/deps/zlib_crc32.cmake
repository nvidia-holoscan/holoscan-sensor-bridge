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
#
# See README.md for detailed information.

cmake_minimum_required(VERSION 3.22)
include_guard(GLOBAL)

if (NOT DEFINED ZLIB_PATH OR ZLIB_PATH STREQUAL "")
  set(_zlib_crc32_build_inc "${CMAKE_BINARY_DIR}/_deps/zlib_crc32")
  set(ZLIB_PATH "${_zlib_crc32_build_inc}")
  # if the crc32.c file already exists, return
  if (EXISTS "${_zlib_crc32_build_inc}/crc32.c")
    return()
  endif()

  # zlib on GitHub uses branch "master", not "main"
  set(_zlib_crc32_url "https://raw.githubusercontent.com/madler/zlib/master/")
  message(STATUS "zlib: downloading crc32 source to ${_zlib_crc32_build_inc}")
  file(MAKE_DIRECTORY "${_zlib_crc32_build_inc}")
  file(DOWNLOAD "${_zlib_crc32_url}/crc32.c" "${_zlib_crc32_build_inc}/crc32.c"
        SHOW_PROGRESS STATUS _zlib_crc32_dl_status)
  list(GET _zlib_crc32_dl_status 0 _zlib_crc32_dl_code)
  if(NOT _zlib_crc32_dl_code EQUAL 0)
    message(FATAL_ERROR "Failed to download crc32.c")
  endif() 

  file(DOWNLOAD "${_zlib_crc32_url}/crc32.h" "${_zlib_crc32_build_inc}/crc32.h"
        SHOW_PROGRESS STATUS _zlib_crc32_dl_status)
  list(GET _zlib_crc32_dl_status 0 _zlib_crc32_dl_code)
  if(NOT _zlib_crc32_dl_code EQUAL 0)
    message(FATAL_ERROR "Failed to download crc32.h")
  endif() 

  file(DOWNLOAD "${_zlib_crc32_url}/zlib.h" "${_zlib_crc32_build_inc}/zlib.h"
      SHOW_PROGRESS STATUS _zlib_crc32_dl_status)
  list(GET _zlib_crc32_dl_status 0 _zlib_crc32_dl_code)
  if(NOT _zlib_crc32_dl_code EQUAL 0)
    message(FATAL_ERROR "Failed to download zlib.h")
  endif()

  file(DOWNLOAD "${_zlib_crc32_url}/zutil.h" "${_zlib_crc32_build_inc}/zutil.h"
    SHOW_PROGRESS STATUS _zlib_crc32_dl_status)
  list(GET _zlib_crc32_dl_status 0 _zlib_crc32_dl_code)
  if(NOT _zlib_crc32_dl_code EQUAL 0)
    message(FATAL_ERROR "Failed to download zutil.h")
  endif()

  file(DOWNLOAD "${_zlib_crc32_url}/zconf.h" "${_zlib_crc32_build_inc}/zconf.h"
      SHOW_PROGRESS STATUS _zlib_crc32_dl_status)
  list(GET _zlib_crc32_dl_status 0 _zlib_crc32_dl_code)
  if(NOT _zlib_crc32_dl_code EQUAL 0)
    message(FATAL_ERROR "Failed to download zconf.h")
  endif()

  if (_zlib_crc32_dl_code EQUAL 0 AND NOT EXISTS "${_zlib_crc32_build_inc}/crc32.c")
    message(FATAL_ERROR "Failed to download zlib_crc32")
  endif()
  
endif()

get_filename_component(ZLIB_PATH "${ZLIB_PATH}" ABSOLUTE)
