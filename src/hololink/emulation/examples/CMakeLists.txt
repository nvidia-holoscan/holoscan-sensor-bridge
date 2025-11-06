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

# Create emulator_utils library for common functionality
add_library(emulator_utils STATIC
    emulator_utils.cu
)

target_include_directories(emulator_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../sensors
    ${CUDAToolkit_INCLUDE_DIRS}
    ${DLPACK_PATH}
)
target_link_libraries(emulator_utils PUBLIC
    hololink::emulation
    hololink::emulation::sensors)

set_property(TARGET emulator_utils PROPERTY POSITION_INDEPENDENT_CODE ON)

# Basic examples that only need HSBEmulator
if (CUDAToolkit_FOUND)
    # COE-based examples
    add_executable(serve_coe_file
        serve_coe_file.cpp
    )
    set_target_properties(serve_coe_file PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
    
    target_include_directories(serve_coe_file PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../../..
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    
    target_link_libraries(serve_coe_file PRIVATE
        hololink::emulation
        hololink::emulation::coe
        emulator_utils
        CUDA::cudart
    )

    add_executable(serve_coe_vb1940_file
        serve_coe_vb1940_file.cpp
    )
    set_target_properties(serve_coe_vb1940_file PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
    
    target_include_directories(serve_coe_vb1940_file PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../../..
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    
    target_link_libraries(serve_coe_vb1940_file PRIVATE
        hololink::emulation
        hololink::emulation::coe
        hololink::emulation::sensors
        emulator_utils
        CUDA::cudart
    )
    
    if (ZLIB_FOUND)
        # Linux-based examples
        add_executable(serve_linux_file
            serve_linux_file.cpp
        )
        set_target_properties(serve_linux_file PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
        
        target_include_directories(serve_linux_file PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}/../../..
            ${ZLIB_INCLUDE_DIRS}
            ${CUDAToolkit_INCLUDE_DIRS}
        )
        
        target_link_libraries(serve_linux_file PRIVATE
            hololink::emulation
            hololink::emulation::roce
            emulator_utils
            ${ZLIB_LIBRARIES}
            CUDA::cudart
        )
    endif()
endif()

# VB1940 sensor examples that need both emulation and sensor libraries
if (CUDAToolkit_FOUND)
    # COE-based VB1940 examples
    add_executable(serve_coe_single_vb1940_frames
        serve_coe_single_vb1940_frames.cpp
    )
    set_target_properties(serve_coe_single_vb1940_frames PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
    
    target_include_directories(serve_coe_single_vb1940_frames PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../../..
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    
    target_link_libraries(serve_coe_single_vb1940_frames PRIVATE
        hololink::emulation
        hololink::emulation::coe
        hololink::emulation::sensors
        emulator_utils
        CUDA::cudart
    )
    
    add_executable(serve_coe_stereo_vb1940_frames
        serve_coe_stereo_vb1940_frames.cpp
    )
    set_target_properties(serve_coe_stereo_vb1940_frames PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
    
    target_include_directories(serve_coe_stereo_vb1940_frames PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../../..
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    
    target_link_libraries(serve_coe_stereo_vb1940_frames PRIVATE
        hololink::emulation
        hololink::emulation::coe
        hololink::emulation::sensors
        emulator_utils
        CUDA::cudart
    )
    
    if (ZLIB_FOUND)
        # Linux-based VB1940 examples
        add_executable(serve_linux_single_vb1940_frames
            serve_linux_single_vb1940_frames.cpp
        )
        set_target_properties(serve_linux_single_vb1940_frames PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
        
        target_include_directories(serve_linux_single_vb1940_frames PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}/../../..
            ${ZLIB_INCLUDE_DIRS}
            ${CUDAToolkit_INCLUDE_DIRS}
        )
        
        target_link_libraries(serve_linux_single_vb1940_frames PRIVATE
            hololink::emulation
            hololink::emulation::roce
            hololink::emulation::sensors
            emulator_utils
            ${ZLIB_LIBRARIES}
            CUDA::cudart
        )
        
        add_executable(serve_linux_stereo_vb1940_frames
            serve_linux_stereo_vb1940_frames.cpp
        )
        set_target_properties(serve_linux_stereo_vb1940_frames PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
        
        target_include_directories(serve_linux_stereo_vb1940_frames PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}/../../..
            ${ZLIB_INCLUDE_DIRS}
            ${CUDAToolkit_INCLUDE_DIRS}
        )
        
        target_link_libraries(serve_linux_stereo_vb1940_frames PRIVATE
            hololink::emulation
            hololink::emulation::roce
            hololink::emulation::sensors
            emulator_utils
            ${ZLIB_LIBRARIES}
            CUDA::cudart
        )
    endif()
endif()
