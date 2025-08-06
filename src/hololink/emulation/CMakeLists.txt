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

file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/dlpack")
file(DOWNLOAD
    https://raw.githubusercontent.com/dmlc/dlpack/main/include/dlpack/dlpack.h
    "${CMAKE_CURRENT_SOURCE_DIR}/dlpack/dlpack.h" SHOW_PROGRESS)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3")

add_library(emulation STATIC
  data_plane.cpp
  hsb_config.cpp
  hsb_emulator.cpp
  linux_data_plane.cpp
  linux_transmitter.cpp
  mem_register.cpp
  net.cpp
  utils.cpp
)
target_compile_options(emulation PRIVATE -DHOLOSCAN_SDK_VERSION=${HOLOSCAN_SDK_VERSION} -g3)
set_property(TARGET emulation PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::emulation ALIAS emulation)

find_package(CUDAToolkit REQUIRED)

target_include_directories(emulation
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:src>
)

target_link_libraries(emulation
  PUBLIC
    CUDA::cudart
    holoscan::core
)

install(TARGETS emulation
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-emulation)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/base_transmitter.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/data_plane.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/hsb_config.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/hsb_emulator.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/linux_data_plane.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/linux_transmitter.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/mem_register.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/net.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/utils.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/emulation
        COMPONENT hololink-emulation)

