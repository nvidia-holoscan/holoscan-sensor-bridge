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

foreach(_required BUILD_CONFIG BINARY_DIR SOURCE_DIR)
  if(NOT DEFINED ${_required})
    message(FATAL_ERROR "${_required} is required")
  endif()
endforeach()

file(REMOVE_RECURSE "${BINARY_DIR}")

set(_configure_command
    "${CMAKE_COMMAND}"
    -S "${SOURCE_DIR}"
    -B "${BINARY_DIR}"
    -DCMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE
    -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE)
if(NOT BUILD_CONFIG STREQUAL "")
  list(APPEND _configure_command "-DCMAKE_BUILD_TYPE=${BUILD_CONFIG}")
endif()

if(DEFINED HololinkRoce_DIR AND NOT HololinkRoce_DIR STREQUAL "")
  # Direct package discovery without CMAKE_PREFIX_PATH.
  list(APPEND _configure_command "-DHololinkRoce_DIR=${HololinkRoce_DIR}")
else()
  if(NOT DEFINED INSTALL_PREFIX)
    message(FATAL_ERROR
      "INSTALL_PREFIX or HololinkRoce_DIR is required")
  endif()
  list(APPEND _configure_command "-DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}")
endif()

# Forward the outer build's generator/toolchain/compiler selection so this
# nested consumer configure is not silently coerced onto the host default
# (e.g. a cross-compile or non-default generator/toolset outer build).
if(DEFINED PARENT_GENERATOR AND NOT PARENT_GENERATOR STREQUAL "")
  list(APPEND _configure_command -G "${PARENT_GENERATOR}")
endif()
if(DEFINED PARENT_GENERATOR_PLATFORM AND NOT PARENT_GENERATOR_PLATFORM STREQUAL "")
  list(APPEND _configure_command -A "${PARENT_GENERATOR_PLATFORM}")
endif()
if(DEFINED PARENT_GENERATOR_TOOLSET AND NOT PARENT_GENERATOR_TOOLSET STREQUAL "")
  list(APPEND _configure_command -T "${PARENT_GENERATOR_TOOLSET}")
endif()
if(DEFINED PARENT_TOOLCHAIN_FILE AND NOT PARENT_TOOLCHAIN_FILE STREQUAL "")
  list(APPEND _configure_command --toolchain "${PARENT_TOOLCHAIN_FILE}")
endif()
if(DEFINED PARENT_C_COMPILER AND NOT PARENT_C_COMPILER STREQUAL "")
  list(APPEND _configure_command "-DCMAKE_C_COMPILER=${PARENT_C_COMPILER}")
endif()
if(DEFINED PARENT_CXX_COMPILER AND NOT PARENT_CXX_COMPILER STREQUAL "")
  list(APPEND _configure_command "-DCMAKE_CXX_COMPILER=${PARENT_CXX_COMPILER}")
endif()

execute_process(
  COMMAND ${_configure_command}
  RESULT_VARIABLE _configure_result
  COMMAND_ECHO STDOUT
)
if(NOT _configure_result EQUAL 0)
  message(FATAL_ERROR "Consumer configuration failed")
endif()

set(_build_command "${CMAKE_COMMAND}" --build "${BINARY_DIR}")
if(NOT BUILD_CONFIG STREQUAL "")
  list(APPEND _build_command --config "${BUILD_CONFIG}")
endif()

execute_process(
  COMMAND ${_build_command}
  RESULT_VARIABLE _build_result
  COMMAND_ECHO STDOUT
)
if(NOT _build_result EQUAL 0)
  message(FATAL_ERROR "Consumer build failed")
endif()
