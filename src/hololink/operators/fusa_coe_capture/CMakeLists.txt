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

add_library(fusa_coe_capture STATIC
    fusa_coe_capture.cpp
)

set_property(TARGET fusa_coe_capture PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::operators::fusa_coe_capture ALIAS fusa_coe_capture)

target_link_directories(fusa_coe_capture
  PUBLIC
    /usr/lib/aarch64-linux-gnu/nvidia
)

target_link_libraries(fusa_coe_capture
  PUBLIC
    hololink
  PRIVATE
    holoscan::core
    nvfusacapext
    nvscibuf
)

target_include_directories(fusa_coe_capture
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    /usr/src/jetson_sipl_api/sipl/fusa/include
    /usr/src/jetson_sipl_api/sipl/include/nvsci
)
