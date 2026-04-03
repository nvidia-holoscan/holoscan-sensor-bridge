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

find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include(hololink_deps/pybind11)

# Main hsb_flasher executable
add_executable(hsb_flasher
    main.cpp
    hsb_flasher_args.cpp
    hsb_discovery.cpp
    hsb_flasher.cpp
    hsb_fw_lookup.cpp
)

target_link_libraries(hsb_flasher
    PRIVATE
    hololink_core
    yaml-cpp
    CURL::libcurl
    OpenSSL::Crypto
)

target_compile_features(hsb_flasher PRIVATE cxx_std_17)

# Traditional SPI/I2C peripherals (legacy support)
add_subdirectory(flash_tools/traditional)

# flash_strategies pybind11 module
pybind11_add_module(flash_strategies
    flash_tools/hsb_lite/flash_strategies.cpp
    flash_tools/hsb_lite/winbond_w25q128jw.cpp
    flash_tools/hsb_lite/clnx_flash.cpp
    flash_tools/hsb_lite/cpnx_flash.cpp
    flash_tools/hsb_lite/hsb_lite_flash_2407.cpp
    flash_tools/hsb_lite/hsb_lite_flash_2504.cpp
    flash_tools/hsb_lite/hsb_lite_flash_2507.cpp
)

target_link_libraries(flash_strategies
    PRIVATE
    hololink_core
    traditional_peripherals
    fmt::fmt-header-only
)

# Copy flash_strategies module to hsb_flasher output directory
add_custom_command(TARGET flash_strategies POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:flash_strategies>
        $<TARGET_FILE_DIR:hsb_flasher>/$<TARGET_FILE_NAME:flash_strategies>
    COMMENT "Copying flash_strategies module to hsb_flasher directory"
)

# hsb_leopard_flash_strategies pybind11 module
pybind11_add_module(hsb_leopard_flash_strategies
    flash_tools/hsb_leopard/flash_strategies.cpp
    flash_tools/hsb_leopard/macronix_mx25u25645g.cpp
    flash_tools/hsb_leopard/cpnx_flash.cpp
    flash_tools/hsb_leopard/hsb_leopard_flash_2507.cpp
)

target_link_libraries(hsb_leopard_flash_strategies
    PRIVATE
    hololink_core
    traditional_peripherals
    fmt::fmt-header-only
)

# Copy hsb_leopard_flash_strategies module to hsb_flasher output directory
add_custom_command(TARGET hsb_leopard_flash_strategies POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:hsb_leopard_flash_strategies>
        $<TARGET_FILE_DIR:hsb_flasher>/$<TARGET_FILE_NAME:hsb_leopard_flash_strategies>
    COMMENT "Copying hsb_leopard_flash_strategies module to hsb_flasher directory"
)

# Collect firmware_information files for dependency tracking
file(GLOB_RECURSE FIRMWARE_INFO_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/firmware_information/*.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/firmware_information/*.yml"
)

# Copy firmware_information with proper dependencies
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/firmware_information/.stamp
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/firmware_information
        ${CMAKE_CURRENT_BINARY_DIR}/firmware_information
    COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/firmware_information/.stamp
    DEPENDS ${FIRMWARE_INFO_FILES}
    COMMENT "Copying firmware_information to build directory"
)

add_custom_target(firmware_information_copy ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/firmware_information/.stamp
)

# Collect firmware_flash_strategies files for dependency tracking
file(GLOB_RECURSE FLASH_STRATEGY_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/firmware_flash_strategies/*.py"
)

# Copy firmware_flash_strategies with proper dependencies
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/firmware_flash_strategies/.stamp
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/firmware_flash_strategies
        ${CMAKE_CURRENT_BINARY_DIR}/firmware_flash_strategies
    COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/firmware_flash_strategies/.stamp
    DEPENDS ${FLASH_STRATEGY_FILES}
    COMMENT "Copying firmware_flash_strategies to build directory"
)

add_custom_target(firmware_flash_strategies_copy ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/firmware_flash_strategies/.stamp
)

# Make hsb_flasher depend on these copy targets
add_dependencies(hsb_flasher firmware_information_copy firmware_flash_strategies_copy)

install(TARGETS hsb_flasher flash_strategies hsb_leopard_flash_strategies traditional_peripherals_py
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT hololink-tools
)

install(DIRECTORY firmware_information
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT hololink-tools
)

install(DIRECTORY firmware_flash_strategies
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT hololink-tools
)

if(HOLOLINK_BUILD_TESTS)
    include(hololink_deps/gtest)

    add_executable(test_hsb_flasher
        tests/test_hsb_flasher_hsb_lite.cpp
        tests/test_hsb_flasher_entry.cpp
        tests/test_hsb_flasher_edge_cases.cpp
        hsb_flasher_args.cpp
        hsb_discovery.cpp
        hsb_fw_lookup.cpp
        hsb_flasher.cpp
    )

    target_include_directories(test_hsb_flasher PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    target_link_libraries(test_hsb_flasher
        PRIVATE
        hololink_core
        yaml-cpp
        CURL::libcurl
        OpenSSL::Crypto
        GTest::gtest_main
    )

    target_compile_features(test_hsb_flasher PRIVATE cxx_std_17)

    add_dependencies(test_hsb_flasher
        firmware_information_copy
        firmware_flash_strategies_copy
        flash_strategies
        hsb_leopard_flash_strategies
    )

    add_test(NAME test_hsb_flasher COMMAND test_hsb_flasher)
endif()
