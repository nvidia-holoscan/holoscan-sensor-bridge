# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

find_package(ibverbs REQUIRED)

add_library(roce_transmitter STATIC
  ibv.cpp
  roce_transmitter_op.cpp
)

set_property(TARGET roce_transmitter PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::operators::roce_transmitter ALIAS roce_transmitter)

target_include_directories(roce_transmitter
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:src>
    ${ibverbs_INCLUDE_DIRS}
)

target_link_libraries(roce_transmitter
  PRIVATE
    hololink::core
    holoscan::core
    CUDA::cuda_driver
    ${ibverbs_LIBRARIES}
)

# Installation of the roce_transmitter operator
install(TARGETS roce_transmitter
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-roce-operators)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/roce_transmitter_op.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/operators/roce_transmitter
        COMPONENT hololink-roce-operators)
