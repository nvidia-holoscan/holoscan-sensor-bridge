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

# we don't expose the C++ interface of the operator yet, therefore link
# statically

add_library(audio_packetizer STATIC
  audio_packetizer.cpp
  )

set_property(TARGET audio_packetizer PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::operators::audio_packetizer ALIAS audio_packetizer)

target_include_directories(audio_packetizer
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:src>
  )

target_link_libraries(audio_packetizer
  PRIVATE
    hololink::core
    holoscan::core
    CUDA::nvrtc
    CUDA::nppicc
  )

# Installation of the audio_packetizer operator
install(TARGETS audio_packetizer
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/audio_packetizer.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/operators/audio_packetizer
        COMPONENT hololink-operators)
