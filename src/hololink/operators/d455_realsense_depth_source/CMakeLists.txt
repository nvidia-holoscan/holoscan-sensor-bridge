# SPDX-License-Identifier: Apache-2.0
add_library(d455_realsense_depth_source STATIC
  realsense_depth_source.cpp
)

set_property(TARGET d455_realsense_depth_source PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::operators::d455_realsense_depth_source ALIAS d455_realsense_depth_source)

target_include_directories(d455_realsense_depth_source
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:src>
)

target_link_libraries(d455_realsense_depth_source
  PUBLIC
    hololink::native
    realsense2
  PRIVATE
    holoscan::core
)
