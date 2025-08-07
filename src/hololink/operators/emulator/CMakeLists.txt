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

add_library(emulator STATIC
  linux_data_plane_op.cpp
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3")
set_property(TARGET emulator PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::operators::emulator ALIAS emulator)

target_include_directories(emulator
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:src>
)

target_link_libraries(emulator
  PRIVATE
    hololink::emulation
    hololink::core
    holoscan::core
)

install(TARGETS emulator
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/linux_data_plane_op.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/operators/emulator
        COMPONENT hololink-operators)

