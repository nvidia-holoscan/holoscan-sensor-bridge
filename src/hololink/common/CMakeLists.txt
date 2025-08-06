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


add_library(hololink STATIC
  gui_renderer.cpp
  holoargs.cpp
  cuda_helper.cpp
  )

set_property(TARGET hololink PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(hololink
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:src>
  )

if(TARGET CUDA::nvtx3)
  set(CUDA_NVTX_TARGET CUDA::nvtx3)
else()
  set(CUDA_NVTX_TARGET CUDA::nvToolsExt)
endif()

target_link_libraries(hololink
  PRIVATE
    holoscan::core

    # Required by the gui_renderer
    holoscan::viz
    holoscan::viz::imgui
  PUBLIC
    hololink::core
    CUDA::cuda_driver
    CUDA::nvrtc
    ${CUDA_NVTX_TARGET}
  )

# Installation of the hololink library
install(TARGETS hololink
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-common)

install(FILES cuda_error.hpp
        cuda_helper.hpp
        gui_renderer.hpp
        holoargs.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/common
        COMPONENT hololink-common)
