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

cmake_minimum_required(VERSION 3.24)
project(hololink_roce_transport_consumer LANGUAGES CXX)

set(CMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit TRUE)
set(CMAKE_DISABLE_FIND_PACKAGE_Hololink TRUE)
set(CMAKE_DISABLE_FIND_PACKAGE_holoscan TRUE)
set(CMAKE_DISABLE_FIND_PACKAGE_fmt TRUE)

find_package(HololinkRoce REQUIRED CONFIG COMPONENTS transport)

if(NOT TARGET hololink::transport::roce)
  message(FATAL_ERROR
    "HololinkRoce COMPONENTS transport must import hololink::transport::roce")
endif()
if(TARGET CUDA::cuda_driver
   OR TARGET holoscan::core
   OR TARGET hololink::core
   OR TARGET fmt::fmt
   OR TARGET fmt::fmt-header-only)
  message(FATAL_ERROR
    "HololinkRoce transport must not load CUDA, Holoscan, Hololink core, or fmt")
endif()
if(TARGET hololink::operators::roce_transmitter OR TARGET hololink::operators::roce_receiver)
  message(FATAL_ERROR
    "HololinkRoce COMPONENTS transport must not define RoCE operator targets")
endif()

get_target_property(_transport_links
  hololink::transport::roce INTERFACE_LINK_LIBRARIES)
if(_transport_links MATCHES
   "CUDA|cuda|Holoscan|holoscan|hololink::core|fmt::")
  message(FATAL_ERROR
    "HololinkRoce transport has a forbidden link dependency: ${_transport_links}")
endif()

add_executable(hololink_roce_transport_consumer main.cpp)
target_link_libraries(hololink_roce_transport_consumer
  PRIVATE hololink::transport::roce)
add_custom_command(TARGET hololink_roce_transport_consumer POST_BUILD
  COMMAND "${CMAKE_COMMAND}"
          "-DBINARY=$<TARGET_FILE:hololink_roce_transport_consumer>"
          "-DREADELF=${CMAKE_READELF}"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/verify_binary.cmake"
  VERBATIM)
