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

set(HSB_EMULATOR_PLATFORM "linux")
set(HSB_HAL_DEFINES "")

enable_language(CUDA)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 80 86 87) # covers all of Orin
    endif()
endif()

# check dependencies
find_package(ZLIB)
find_package(CUDAToolkit)
include("${HOLOLINK_REL_PATH}/cmake/hololink_deps/dlpack.cmake")

##### Standalone python environment handling

if (HSB_EMULATOR_BUILD_PYTHON AND NOT HSB_EMULATOR_TOT_BUILD)
    # we only need to find CUDAToolkit above, but if Python bindings are required, it will be required
    find_package(CUDAToolkit REQUIRED)
    # Find pybind11 for Python extension modules when building standalone
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    find_package(pybind11 REQUIRED)
    add_library(_emulation SHARED python/hololink/emulation/_emulation.cpp)

    target_include_directories(_emulation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${pybind11_INCLUDE_DIRS})
    
    # Link _emulation with the core emulation libraries
    target_link_libraries(_emulation PRIVATE hololink::emulation ${pybind11_LIBRARIES})
    target_link_libraries(_emulation PRIVATE hololink::emulation::coe)
    if (ZLIB_FOUND)
        target_link_libraries(_emulation PRIVATE hololink::emulation::roce)
    endif()
    
    # Set target properties for _emulation
    set_target_properties(_emulation PROPERTIES
        OUTPUT_NAME "_emulation"
        PREFIX ""
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )
    set_property(TARGET _emulation PROPERTY POSITION_INDEPENDENT_CODE ON)
    
    # Create _sensors Python extension module (renamed to _emulation_sensors to match Makefile)
    add_library(_emulation_sensors SHARED python/hololink/emulation/sensors/_sensors.cpp)

    target_include_directories(_emulation_sensors PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${pybind11_INCLUDE_DIRS})
    
    # Link _sensors with the sensors library and _emulation
    target_link_libraries(_emulation_sensors PRIVATE hololink::emulation::sensors _emulation ${pybind11_LIBRARIES})
    
    # Set target properties for _sensors
    set_target_properties(_emulation_sensors PROPERTIES
        OUTPUT_NAME "_emulation_sensors"
        PREFIX ""
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )
    set_property(TARGET _emulation_sensors PROPERTY POSITION_INDEPENDENT_CODE ON)
    
    add_custom_command(TARGET _emulation_sensors POST_BUILD
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/build_pyvenv.sh $<TARGET_FILE_NAME:_emulation> $<TARGET_FILE_NAME:_emulation_sensors> "${CMAKE_CURRENT_SOURCE_DIR}/python" "${CMAKE_BINARY_DIR}/${HSB_EMULATOR_PYTHON_VENV}" ${CUDAToolkit_VERSION_MAJOR}
        COMMENT "Building standalone hololink python environment"
    )
    
endif()