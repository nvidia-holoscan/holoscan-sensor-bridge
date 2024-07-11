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

add_library(base_receiver_op STATIC
  base_receiver_op.cpp
  )

set_property(TARGET base_receiver_op PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::operators::base_receiver_op ALIAS base_receiver_op)

target_include_directories(base_receiver_op
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:src>
  )

target_link_libraries(base_receiver_op
  PUBLIC
    holoscan::core
    hololink
    hololink::native
    CUDA::cuda_driver
  )

add_subdirectory(csi_to_bayer)
if(DEFINED ENV{CONTAINER_TYPE} AND "$ENV{CONTAINER_TYPE}" STREQUAL "igpu")
  add_subdirectory(argus_isp)
endif()
add_subdirectory(gamma_correction)
add_subdirectory(image_processor)
add_subdirectory(linux_receiver)
add_subdirectory(roce_receiver)
add_subdirectory(roce_transmitter)
