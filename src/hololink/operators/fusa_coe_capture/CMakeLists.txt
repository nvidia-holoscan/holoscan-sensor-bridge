# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

include(sipl_api)

add_library(fusa_coe_capture_core STATIC
    fusa_coe_capture_core.cpp
)

set_property(TARGET fusa_coe_capture_core PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::operators::fusa_coe_capture_core ALIAS fusa_coe_capture_core)

target_link_directories(fusa_coe_capture_core
  PUBLIC
    $<$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},aarch64>:/usr/lib/aarch64-linux-gnu/nvidia>
)

if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  message(WARNING
    "fusa_coe_capture_core: FUSA capture is only supported on aarch64 Jetson platforms")
endif()

target_link_libraries(fusa_coe_capture_core
  PUBLIC
    hololink
    holoscan::core
    nvfusacapext
    nvscibuf
)

target_include_directories(fusa_coe_capture_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<$<EQUAL:${NVSIPL_API_MAJOR_VERSION},1>:/usr/src/jetson_sipl_api/sipl/fusa/include>
    $<$<EQUAL:${NVSIPL_API_MAJOR_VERSION},2>:/usr/src/jetson_sipl_api/sipl/capture/include>
    /usr/src/jetson_sipl_api/sipl/include/nvsci
)

set_target_properties(fusa_coe_capture_core PROPERTIES EXPORT_NAME operators::fusa_coe_capture_core)
install(TARGETS fusa_coe_capture_core
        EXPORT HololinkTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/fusa_coe_capture_core.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/operators/fusa_coe_capture
        COMPONENT hololink-operators)

add_library(fusa_coe_capture STATIC
    fusa_coe_capture.cpp
)

set_property(TARGET fusa_coe_capture PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::operators::fusa_coe_capture ALIAS fusa_coe_capture)

target_link_libraries(fusa_coe_capture
  PUBLIC
    hololink::operators::fusa_coe_capture_core
  PRIVATE
    holoscan::core
)

target_include_directories(fusa_coe_capture
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set_target_properties(fusa_coe_capture PROPERTIES EXPORT_NAME operators::fusa_coe_capture)
install(TARGETS fusa_coe_capture
        EXPORT HololinkTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/fusa_coe_capture.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/operators/fusa_coe_capture
        COMPONENT hololink-operators)
