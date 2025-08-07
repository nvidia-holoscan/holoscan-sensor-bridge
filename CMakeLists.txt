# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.20)

# Options
option(HOLOLINK_BUILD_PYTHON "Build Hololink Python Bindings" ON)
option(HOLOLINK_BUILD_TESTS "Build Hololink Tests" ON)
option(HOLOLINK_BUILD_ONLY_NATIVE "Build only native (i.e. C++) components" OFF)
option(HOLOLINK_BUILD_SIPL "Build NvSIPL capture components" OFF)

file(READ "VERSION" HOLOLINK_VERSION)
string(STRIP ${HOLOLINK_VERSION} HOLOLINK_VERSION)
string(REGEX MATCH "^([0-9]+\.[0-9]+\.[0-9]+)" HOLOLINK_VERSION_NUM ${HOLOLINK_VERSION})

set(PROJECT_LANGUAGES CXX)
if (NOT HOLOLINK_BUILD_ONLY_NATIVE)
  list(APPEND PROJECT_LANGUAGES CUDA)
endif()

project(hololink
  LANGUAGES ${PROJECT_LANGUAGES}
  VERSION ${HOLOLINK_VERSION_NUM})

if (NOT HOLOLINK_BUILD_ONLY_NATIVE)
  include(CTest)

  # Find Holoscan
  find_package(holoscan 3.3 REQUIRED CONFIG PATHS "/opt/nvidia/holoscan")

  # Find pybind11
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

  # We fetch pybind11 since we need the same version as the Holoscan SDK
  # and it's not necessarily available on all the platforms
  include(FetchContent)
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG v2.13.6
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(pybind11)

  # We fetch googletest since we need the same version as the Holoscan SDK.
  # As recommended by the user guide (https://google.github.io/googletest/quickstart-cmake.html),
  # we use Git commit hash of the GoogleTest version to use the same version as the Holoscan SDK.
  # (https://github.com/google/googletest/releases/tag/release-1.12.1)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/58d77fa8070e8cec2dc1ed015d66b454c8d78850.zip
  )
  FetchContent_MakeAvailable(googletest)

  # add CMAKE_MODULE_PATH for pybind11_add_hololink_module
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

  # enabled trace and debug log messages in debug build only
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_definitions(HOLOSCAN_LOG_ACTIVE_LEVEL=2)
  else()
    add_compile_definitions(HOLOSCAN_LOG_ACTIVE_LEVEL=0)
  endif()
endif()

if(NOT CMAKE_INSTALL_LIBDIR)
  set(CMAKE_INSTALL_LIBDIR lib)
endif()

add_subdirectory(src)
add_subdirectory(tools)
add_subdirectory(examples)

if(NOT HOLOLINK_BUILD_ONLY_NATIVE)
if(HOLOLINK_BUILD_TESTS)
  add_subdirectory(tests)
endif()
if(HOLOLINK_BUILD_PYTHON)
  add_subdirectory(python)
endif()
add_subdirectory(scripts)
endif()

# Add packaging
include(cmake/HololinkCPack.cmake)
