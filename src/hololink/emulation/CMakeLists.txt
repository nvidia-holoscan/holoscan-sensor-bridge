# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.17)

# command line options
option(HSB_EMULATOR_BUILD_PYTHON "When building standalone HSB Emulator, including building the python environment." ON)
set(HSB_EMULATOR_PYTHON_VENV "env" CACHE STRING "relative path (from build directory) to the standalone HSB Emulator's Python virtual environment")
set(HSB_EMULATOR_TARGET "linux" CACHE STRING "the target platform for the standalone HSB Emulator")

set(HSB_EMULATOR_EMBEDDED_TARGETS STM32F767ZI)
set(HSB_EMULATOR_TARGETS linux)

# This checks if user is trying to build the emulator only (no HSB components)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)    
    # this is for building emulator-only
    
    # configure standalone project
    set(PROJECT_LANGUAGES C CXX)
    project(hololink
        LANGUAGES ${PROJECT_LANGUAGES})
    set(HSB_EMULATOR_TOT_BUILD FALSE)
    set(HOLOLINK_REL_PATH "${CMAKE_SOURCE_DIR}/../../..")
    list(APPEND HSB_EMULATOR_TARGETS ${HSB_EMULATOR_EMBEDDED_TARGETS})
else()
    # in this case the HSB_EMULATOR_TARGET *must* be linux as MCU targets will not build the target libraries
    if (NOT ${HSB_EMULATOR_TARGET} STREQUAL "linux")
        message(FATAL_ERROR "HSB_EMULATOR_TARGET must be 'linux' (currently ${HSB_EMULATOR_TARGET}) when building as part of HSB; MCU targets do not build the emulator target libraries.")
    endif()
    # No python environment building because we will use TOT python bindings
    set(HSB_EMULATOR_CUPY OFF)
    set(HSB_EMULATOR_BUILD_PYTHON ${HOLOLINK_BUILD_PYTHON})
    set(HSB_EMULATOR_TOT_BUILD TRUE)
    set(HOLOLINK_REL_PATH "${hololink_SOURCE_DIR}")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-field-initializers")

if(NOT HSB_EMULATOR_TARGET IN_LIST HSB_EMULATOR_TARGETS)
    message(FATAL_ERROR "HSB_EMULATOR_TARGET must be one of ${HSB_EMULATOR_TARGETS}, got: ${HSB_EMULATOR_TARGET}")
endif()

# configure for target platform
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/targets/${HSB_EMULATOR_TARGET}.cmake")

# configure target builds
add_subdirectory(src)
add_subdirectory(sensors)
add_subdirectory(examples)

# install
set(HSB_EMULATOR_INCS
    ${CMAKE_CURRENT_SOURCE_DIR}/base_transmitter.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data_plane.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/coe_data_plane.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/linux_data_plane.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hsb_config.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hsb_emulator.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/i2c_interface.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/address_memory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/net.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.hpp)

install(FILES ${HSB_EMULATOR_INCS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/emulation
    COMPONENT hololink-emulation)
