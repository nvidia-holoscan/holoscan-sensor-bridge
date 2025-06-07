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

add_library(iq_enc STATIC
  iq_enc_op.cpp
  iq_enc_op.cu
)

set_property(TARGET iq_enc PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::operators::iq_enc ALIAS iq_enc)

target_include_directories(iq_enc
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:src>
)

target_link_libraries(iq_enc
  PRIVATE
    hololink
    holoscan::core
    holoscan::viz::imgui
)

# Installation of the iq_enc operator
install(TARGETS iq_enc
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/iq_enc_op.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/operators/iq_enc
        COMPONENT hololink-operators)

