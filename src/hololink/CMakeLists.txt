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

add_subdirectory(native)
add_subdirectory(operators)

add_library(hololink STATIC
  data_channel.cpp
  enumerator.cpp
  hololink.cpp
  logging.cpp
  metadata.cpp
  timeout.cpp
  )

set_property(TARGET hololink PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(hololink
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:src>
  )

target_link_libraries(hololink
  PRIVATE
    hololink::native
    holoscan::core
  )
