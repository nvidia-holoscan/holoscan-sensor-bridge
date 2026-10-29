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

foreach(_required
    BUILD_CONFIG
    CONSUMER_SOURCE_DIR
    PRODUCER_BINARY_DIR
    PRODUCER_SOURCE_DIR
    WORK_DIR)
  if(NOT DEFINED ${_required})
    message(FATAL_ERROR "${_required} is required")
  endif()
endforeach()

file(REMOVE_RECURSE "${PRODUCER_BINARY_DIR}" "${WORK_DIR}")

set(_configure_command
    "${CMAKE_COMMAND}"
    -S "${PRODUCER_SOURCE_DIR}"
    -DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=TRUE
    -DCMAKE_DISABLE_FIND_PACKAGE_Hololink=TRUE
    -DCMAKE_DISABLE_FIND_PACKAGE_holoscan=TRUE
    -DCMAKE_DISABLE_FIND_PACKAGE_fmt=TRUE)
if(NOT BUILD_CONFIG STREQUAL "")
  list(APPEND _configure_command "-DCMAKE_BUILD_TYPE=${BUILD_CONFIG}")
endif()

# Forward the outer build's generator/toolchain/compiler selection to the
# producer configure, and collect the same set of -D flags to relay into the
# nested ConfigureAndBuildConsumer.cmake invocations below, so a non-default
# generator/toolset/toolchain/compiler is not silently dropped at either
# nesting level.
set(_forward_args)
if(DEFINED PARENT_GENERATOR AND NOT PARENT_GENERATOR STREQUAL "")
  list(APPEND _configure_command -G "${PARENT_GENERATOR}")
  list(APPEND _forward_args "-DPARENT_GENERATOR=${PARENT_GENERATOR}")
endif()
if(DEFINED PARENT_GENERATOR_PLATFORM AND NOT PARENT_GENERATOR_PLATFORM STREQUAL "")
  list(APPEND _configure_command -A "${PARENT_GENERATOR_PLATFORM}")
  list(APPEND _forward_args "-DPARENT_GENERATOR_PLATFORM=${PARENT_GENERATOR_PLATFORM}")
endif()
if(DEFINED PARENT_GENERATOR_TOOLSET AND NOT PARENT_GENERATOR_TOOLSET STREQUAL "")
  list(APPEND _configure_command -T "${PARENT_GENERATOR_TOOLSET}")
  list(APPEND _forward_args "-DPARENT_GENERATOR_TOOLSET=${PARENT_GENERATOR_TOOLSET}")
endif()
if(DEFINED PARENT_TOOLCHAIN_FILE AND NOT PARENT_TOOLCHAIN_FILE STREQUAL "")
  list(APPEND _configure_command --toolchain "${PARENT_TOOLCHAIN_FILE}")
  list(APPEND _forward_args "-DPARENT_TOOLCHAIN_FILE=${PARENT_TOOLCHAIN_FILE}")
endif()
if(DEFINED PARENT_C_COMPILER AND NOT PARENT_C_COMPILER STREQUAL "")
  list(APPEND _configure_command "-DCMAKE_C_COMPILER=${PARENT_C_COMPILER}")
  list(APPEND _forward_args "-DPARENT_C_COMPILER=${PARENT_C_COMPILER}")
endif()
if(DEFINED PARENT_CXX_COMPILER AND NOT PARENT_CXX_COMPILER STREQUAL "")
  list(APPEND _configure_command "-DCMAKE_CXX_COMPILER=${PARENT_CXX_COMPILER}")
  list(APPEND _forward_args "-DPARENT_CXX_COMPILER=${PARENT_CXX_COMPILER}")
endif()

execute_process(
  COMMAND ${_configure_command} -B "${PRODUCER_BINARY_DIR}"
  RESULT_VARIABLE _result
  COMMAND_ECHO STDOUT
)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR "Standalone RoCE transport configuration failed")
endif()

set(_install_libdir "lib")
if(EXISTS "${PRODUCER_BINARY_DIR}/CMakeCache.txt")
  file(STRINGS "${PRODUCER_BINARY_DIR}/CMakeCache.txt" _install_libdir_line
    REGEX "^CMAKE_INSTALL_LIBDIR:")
  if(_install_libdir_line)
    list(GET _install_libdir_line 0 _install_libdir_line)
    string(REGEX REPLACE "^CMAKE_INSTALL_LIBDIR:[^=]+=" ""
      _install_libdir "${_install_libdir_line}")
  endif()
  unset(_install_libdir_line)
endif()

set(_build_command
    "${CMAKE_COMMAND}"
    --build "${PRODUCER_BINARY_DIR}"
    --target hololink_transport_roce)
if(NOT BUILD_CONFIG STREQUAL "")
  list(APPEND _build_command --config "${BUILD_CONFIG}")
endif()
execute_process(
  COMMAND ${_build_command}
  RESULT_VARIABLE _result
  COMMAND_ECHO STDOUT
)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR "Standalone RoCE transport build failed")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}"
          "-DBINARY_DIR=${WORK_DIR}/build-tree-consumer"
          "-DBUILD_CONFIG=${BUILD_CONFIG}"
          "-DSOURCE_DIR=${CONSUMER_SOURCE_DIR}"
          "-DHololinkRoce_DIR=${PRODUCER_BINARY_DIR}/HololinkRoce"
          ${_forward_args}
          -P "${CMAKE_CURRENT_LIST_DIR}/ConfigureAndBuildConsumer.cmake"
  RESULT_VARIABLE _result
  COMMAND_ECHO STDOUT
)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR "Standalone RoCE build-tree consumer failed")
endif()

set(_install_prefix "${WORK_DIR}/install")
execute_process(
  COMMAND "${CMAKE_COMMAND}"
          "-DBUILD_CONFIG=${BUILD_CONFIG}"
          "-DBUILD_DIR=${PRODUCER_BINARY_DIR}"
          -DCOMPONENT=hololink-roce-transport
          "-DINSTALL_PREFIX=${_install_prefix}"
          -P "${CMAKE_CURRENT_LIST_DIR}/InstallConsumerFixture.cmake"
  RESULT_VARIABLE _result
  COMMAND_ECHO STDOUT
)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR "Standalone RoCE transport install failed")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}"
          "-DBINARY_DIR=${WORK_DIR}/installed-consumer"
          "-DBUILD_CONFIG=${BUILD_CONFIG}"
          "-DSOURCE_DIR=${CONSUMER_SOURCE_DIR}"
          "-DHololinkRoce_DIR=${_install_prefix}/${_install_libdir}/cmake/HololinkRoce"
          ${_forward_args}
          -P "${CMAKE_CURRENT_LIST_DIR}/ConfigureAndBuildConsumer.cmake"
  RESULT_VARIABLE _result
  COMMAND_ECHO STDOUT
)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR "Standalone RoCE installed consumer failed")
endif()
