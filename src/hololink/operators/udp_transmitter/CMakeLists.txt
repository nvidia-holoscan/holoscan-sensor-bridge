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

add_library(udp_transmitter STATIC
  udp_transmitter_op.cpp
)

set_property(TARGET udp_transmitter PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::operators::udp_transmitter ALIAS udp_transmitter)

target_include_directories(udp_transmitter
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:src>
)

target_link_libraries(udp_transmitter
  PRIVATE
    hololink
    holoscan::core
)

# Installation of the udp_transmitter operator
install(TARGETS udp_transmitter
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/udp_transmitter_op.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/operators/udp_transmitter
        COMPONENT hololink-operators)
