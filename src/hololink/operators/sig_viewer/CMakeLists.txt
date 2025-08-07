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

add_library(sig_viewer STATIC
    sig_viewer_op.cpp
    sig_viewer_op.cu
)

set_property(TARGET sig_viewer PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::operators::sig_viewer ALIAS sig_viewer)

target_include_directories(sig_viewer
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:src>
)

target_link_libraries(sig_viewer
  PRIVATE
    cufft
    hololink
    holoscan::core
    holoscan::viz::imgui
)

# Installation of the sig_viewer operator
install(TARGETS sig_viewer
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/sig_viewer_op.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/operators/sig_viewer
        COMPONENT hololink-operators)
