# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


add_library(hololink STATIC)

target_sources(hololink
  PRIVATE
    holoargs.cpp
    cuda_helper.cpp
    shutdown_signal_disposition_guard.cpp
    tools.cpp
)

set_property(TARGET hololink PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(hololink
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

if(TARGET CUDA::nvtx3)
  set(CUDA_NVTX_TARGET CUDA::nvtx3)
else()
  set(CUDA_NVTX_TARGET CUDA::nvToolsExt)
endif()

target_link_libraries(hololink
  PRIVATE
    holoscan::core
  PUBLIC
    hololink::core
    CUDA::cuda_driver
    CUDA::nvrtc
    ${CUDA_NVTX_TARGET}
    fmt::fmt-header-only
  )

if (HOLOLINK_BUILD_ROCE)
  find_package(ibverbs REQUIRED)
  # set HAS_IBVERBS if ibverbs found
  target_compile_options(hololink PRIVATE -DHAS_IBVERBS=1)
  target_link_libraries(hololink PRIVATE ${ibverbs_LIBRARIES})
endif() # HOLOLINK_BUILD_ROCE

# Installation of the hololink library
set_target_properties(hololink PROPERTIES EXPORT_NAME common)
install(TARGETS hololink
        EXPORT HololinkTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-common)

install(FILES cuda_error.hpp
        cuda_helper.hpp
        holoargs.hpp
        shutdown_signal_disposition_guard.hpp
        tools.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/common
        COMPONENT hololink-common)

# hololink::gui_renderer -- the Dear ImGui window operators draw their GUIs into. A separate
# target so the HoloViz dependency stays off everything else in hololink::common.
if(HOLOLINK_HAS_IMGUI)
  add_library(gui_renderer STATIC gui_renderer.cpp gui_staging.cpp)
  add_library(hololink::gui_renderer ALIAS gui_renderer)
  set_property(TARGET gui_renderer PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_include_directories(gui_renderer
    INTERFACE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
  target_link_libraries(gui_renderer
    PRIVATE
      hololink::core
      holoscan::visualization::holoviz
      CUDA::cuda_driver
      CUDA::cudart
    )
  set_target_properties(gui_renderer PROPERTIES EXPORT_NAME gui_renderer)
  install(TARGETS gui_renderer
          EXPORT HololinkTargets
          DESTINATION ${CMAKE_INSTALL_LIBDIR}
          COMPONENT hololink-common)
  install(FILES gui_renderer.hpp gui_staging.hpp
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/common
          COMPONENT hololink-common)
endif()
