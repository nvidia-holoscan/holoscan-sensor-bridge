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

# Compile-time configuration for the cross-target hsb_control / serve_coe_msg / serve_roce_msg
# trio. Each variable maps to a #define of the same name inside the .cpp. Strings need to be
# quoted into a C string literal here so the resulting -D... reaches the compiler with the
# embedded quotes intact (cmake strips one level).
set(IPV4_ADDRESS "192.168.0.2" CACHE STRING "IPv4 address baked into hsb_control / serve_*_msg")
set(DATA_PLANE_ID 0 CACHE STRING "data_plane_id baked into hsb_control / serve_*_msg")
set(SENSOR_ID 0 CACHE STRING "sensor_id baked into hsb_control / serve_*_msg")
set(TRANSPORT_MSG "test msg 0123456789abcdefg" CACHE STRING "message payload baked into serve_*_msg")
set(FRAME_RATE 30 CACHE STRING "send rate in frames-per-second for serve_*_msg (loop sleeps 1000/FRAME_RATE ms)")

set(_emu_msg_defs
    IPV4_ADDRESS="${IPV4_ADDRESS}"
    DATA_PLANE_ID=${DATA_PLANE_ID}
    SENSOR_ID=${SENSOR_ID}
)
set(_emu_msg_msg_defs
    ${_emu_msg_defs}
    TRANSPORT_MSG="${TRANSPORT_MSG}"
    FRAME_RATE=${FRAME_RATE}
)

# Cross-target examples include platform-specific headers (e.g. net.hpp)
# via bare `#include "net.hpp"`. Put src/${HSB_EMULATOR_PLATFORM}/ first so the
# target's net.hpp wins over emulation/net.hpp from transitive include paths.
set(_hsb_cross_target_platform_include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/${HSB_EMULATOR_PLATFORM})

# build for all targets
add_executable(hsb_control
    hsb_control.cpp
)

target_compile_definitions(hsb_control PRIVATE ${_emu_msg_defs})
target_link_libraries(hsb_control PRIVATE
    hololink::emulation
)

add_executable(serve_coe_msg
    serve_coe_msg.cpp
)
target_compile_definitions(serve_coe_msg PRIVATE ${_emu_msg_msg_defs})
target_link_libraries(serve_coe_msg PRIVATE
    hololink::emulation
    hololink::emulation::coe
)

add_executable(serve_roce_msg
    serve_roce_msg.cpp
)
target_compile_definitions(serve_roce_msg PRIVATE ${_emu_msg_msg_defs})
target_link_libraries(serve_roce_msg PRIVATE
    hololink::emulation
    hololink::emulation::roce
)

foreach(_target IN ITEMS hsb_control serve_coe_msg serve_roce_msg)
    target_include_directories(${_target} BEFORE PRIVATE
        ${_hsb_cross_target_platform_include})
endforeach()

# MCU-only examples that exercise STM32-specific peripherals (timers, GPIO, TIM IRQs).
if (${HSB_EMULATOR_TARGET} STREQUAL "STM32F767ZI")
    add_executable(gpio_send_roce
        gpio_send_roce.cpp
    )
    target_link_libraries(gpio_send_roce PRIVATE
        hololink::emulation
        hololink::emulation::roce
    )

    add_executable(gpio_trigger_send_roce
        gpio_trigger_send_roce.cpp
    )
    target_link_libraries(gpio_trigger_send_roce PRIVATE
        hololink::emulation
        hololink::emulation::roce
    )
endif()

if (${HSB_EMULATOR_TARGET} STREQUAL "linux")

# Create emulator_utils library for common functionality
add_library(emulator_utils STATIC
    emulator_utils.cu
)

# Own dir is exposed PUBLIC so consumers can include "emulator_utils.hpp" by
# bare name. CUDA includes are PUBLIC because emulator_utils.hpp pulls in
# <cuda_runtime.h>, which the C++ consumers (serve_*) won't auto-resolve
# without this (only the .cu sources get CUDA includes from nvcc itself).
# Everything else (dlpack, sensors dir, src/) arrives transitively via
# hololink::emulation and hololink::emulation::sensors below.
target_include_directories(emulator_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
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

    target_link_libraries(serve_coe_single_vb1940_frames PRIVATE
        hololink::emulation
        hololink::emulation::coe
        hololink::emulation::sensors
        emulator_utils
        CUDA::cudart
    )
    
    add_executable(serve_coe_test_data
        serve_coe_test_data.cpp
    )
    set_target_properties(serve_coe_test_data PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

    target_include_directories(serve_coe_test_data PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../..
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(serve_coe_test_data PRIVATE
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

        target_link_libraries(serve_linux_stereo_vb1940_frames PRIVATE
            hololink::emulation
            hololink::emulation::roce
            hololink::emulation::sensors
            emulator_utils
            ${ZLIB_LIBRARIES}
            CUDA::cudart
        )

        add_executable(serve_linux_single_imx274_frames
            serve_linux_single_imx274_frames.cpp
        )
        set_target_properties(serve_linux_single_imx274_frames PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

        target_link_libraries(serve_linux_single_imx274_frames PRIVATE
            hololink::emulation
            hololink::emulation::roce
            hololink::emulation::sensors
            emulator_utils
            ${ZLIB_LIBRARIES}
            CUDA::cudart
        )

        add_executable(serve_linux_stereo_imx274_frames
            serve_linux_stereo_imx274_frames.cpp
        )
        set_target_properties(serve_linux_stereo_imx274_frames PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

        target_link_libraries(serve_linux_stereo_imx274_frames PRIVATE
            hololink::emulation
            hololink::emulation::roce
            hololink::emulation::sensors
            emulator_utils
            ${ZLIB_LIBRARIES}
            CUDA::cudart
        )
    endif()
endif()

endif()