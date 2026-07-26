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

add_library(hololink_module_sensors_max96716a STATIC max96716a.cpp)
add_library(hololink::module::sensors::max96716a ALIAS hololink_module_sensors_max96716a)

target_compile_features(hololink_module_sensors_max96716a PUBLIC cxx_std_17)
target_include_directories(hololink_module_sensors_max96716a
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(hololink_module_sensors_max96716a
    PUBLIC
        hololink::module_headers
        hololink::core
        fmt::fmt-header-only)
set_target_properties(hololink_module_sensors_max96716a PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
