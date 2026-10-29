# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(cpu_roce_frame_receiver STATIC
  cpu_roce_frame_receiver.cpp
  cpu_roce_frame_receiver_internal.hpp
)
add_library(hololink::operators::cpu_roce_frame_receiver ALIAS cpu_roce_frame_receiver)

set_property(TARGET cpu_roce_frame_receiver PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(cpu_roce_frame_receiver
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

find_package(Threads REQUIRED)
target_link_libraries(cpu_roce_frame_receiver
  PUBLIC
    hololink::transport::roce
    Threads::Threads
  PRIVATE
    ibverbs::ibverbs
    CUDA::cuda_driver
)

# Do not EXPORT HololinkTargets: hololink::transport::roce lives in
# HololinkRoceTransportTargets (same pattern as gpu_roce_transceiver).
install(TARGETS cpu_roce_frame_receiver
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-roce-operators)

install(FILES cpu_roce_frame_receiver.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/operators/cpu_roce_transceiver
        COMPONENT hololink-roce-operators)

add_library(cpu_roce_frame_transmitter STATIC
  cpu_roce_frame_transmitter.cpp
  cpu_roce_frame_transmitter_internal.hpp
)
add_library(hololink::operators::cpu_roce_frame_transmitter ALIAS cpu_roce_frame_transmitter)

set_property(TARGET cpu_roce_frame_transmitter PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(cpu_roce_frame_transmitter
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(cpu_roce_frame_transmitter
  PUBLIC
    hololink::transport::roce
    Threads::Threads
  PRIVATE
    ibverbs::ibverbs
    CUDA::cuda_driver
)

# Do not EXPORT HololinkTargets: hololink::transport::roce lives in
# HololinkRoceTransportTargets (same pattern as gpu_roce_transceiver).
install(TARGETS cpu_roce_frame_transmitter
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-roce-operators)

install(FILES cpu_roce_frame_transmitter.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/operators/cpu_roce_transceiver
        COMPONENT hololink-roce-operators)

add_library(cpu_roce_transceiver_op STATIC
  cpu_roce_transceiver_op.cpp
)
add_library(hololink::operators::cpu_roce_transceiver ALIAS cpu_roce_transceiver_op)

set_property(TARGET cpu_roce_transceiver_op PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(cpu_roce_transceiver_op
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(cpu_roce_transceiver_op
  PUBLIC
    hololink::operators::base_receiver_op
    hololink::transport::roce
  PRIVATE
    hololink::core
    holoscan::core
)

# Do not EXPORT HololinkTargets: hololink::transport::roce lives in
# HololinkRoceTransportTargets (same pattern as gpu_roce_transceiver).
install(TARGETS cpu_roce_transceiver_op
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-roce-operators)

install(FILES cpu_roce_transceiver_op.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/operators/cpu_roce_transceiver
        COMPONENT hololink-roce-operators)
