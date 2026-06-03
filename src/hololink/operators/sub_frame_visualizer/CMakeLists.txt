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

add_library(sub_frame_visualizer STATIC
    sub_frame_visualizer.cpp
)

set_property(TARGET sub_frame_visualizer PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::operators::sub_frame_visualizer ALIAS sub_frame_visualizer)

target_include_directories(sub_frame_visualizer
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

target_link_libraries(sub_frame_visualizer
  PRIVATE
    hololink
    holoscan::core
    holoscan::viz
)

# Installation of the sub_frame_visualizer operator
set_target_properties(sub_frame_visualizer PROPERTIES EXPORT_NAME operators::sub_frame_visualizer)
install(TARGETS sub_frame_visualizer
        EXPORT HololinkTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)

install(FILES
          ${CMAKE_CURRENT_SOURCE_DIR}/sub_frame_visualizer.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/operators/sub_frame_visualizer
        COMPONENT hololink-operators)
