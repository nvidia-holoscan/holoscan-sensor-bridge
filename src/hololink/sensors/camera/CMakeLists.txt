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

# we don't expose the C++ interface of the operator yet, therefore link
# statically

add_library(hololink_camera_sensor STATIC
  camera_mode.cpp
  camera_sensor.cpp
  )

set_property(TARGET hololink_camera_sensor PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::sensors::camera_sensor ALIAS hololink_camera_sensor)

target_include_directories(hololink_camera_sensor
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

target_link_libraries(hololink_camera_sensor
  PUBLIC
    hololink::sensor
    fmt::fmt-header-only
  )

set_target_properties(hololink_camera_sensor PROPERTIES EXPORT_NAME sensors::camera_sensor)
install(TARGETS hololink_camera_sensor
        EXPORT HololinkTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-camera-sensor)

install(FILES
        camera_sensor.hpp
        camera_mode.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/sensors/camera
        COMPONENT hololink-camera-sensor)

# Add camera implementation subdirectories
add_subdirectory(imx274)
add_subdirectory(vb1940)
