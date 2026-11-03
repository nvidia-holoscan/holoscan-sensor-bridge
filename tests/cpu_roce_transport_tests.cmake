# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function(ConfigureCpuRoceTransportTests)
  list(PREPEND CMAKE_MODULE_PATH
    "${hololink_SOURCE_DIR}/src/hololink/transport/roce/cmake")
  find_package(ibverbs REQUIRED MODULE)
  find_package(Threads REQUIRED)

  # hololink_transport_roce links ibverbs PRIVATE, so LINK_ONLY strips its
  # include dirs from consumers. Any test here that includes verbs.h or the
  # private ibv.hpp directly must link ibverbs::ibverbs itself.

  add_executable(hololink_cpu_roce_ibv_test cpu_roce_ibv_test.cpp)
  target_compile_features(hololink_cpu_roce_ibv_test PRIVATE cxx_std_17)
  target_link_libraries(hololink_cpu_roce_ibv_test
    PRIVATE
      ibverbs::ibverbs
      Threads::Threads
      GTest::gtest_main)

  add_executable(
    hololink_cpu_roce_ibv_link_test
    cpu_roce_ibv_link_test.cpp)
  target_link_libraries(hololink_cpu_roce_ibv_link_test
    PRIVATE
      hololink::transport::roce
      ibverbs::ibverbs
      GTest::gtest_main)

  add_executable(
    hololink_cpu_roce_transceiver_ring_test
    cpu_roce_transceiver_ring_test.cpp)
  target_link_libraries(hololink_cpu_roce_transceiver_ring_test
    PRIVATE
      hololink::transport::roce
      ibverbs::ibverbs
      Threads::Threads
      GTest::gtest_main)

  foreach(_target
      hololink_cpu_roce_ibv_test
      hololink_cpu_roce_ibv_link_test
      hololink_cpu_roce_transceiver_ring_test)
    set_target_properties(${_target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${hololink_BINARY_DIR}/gtests")
    add_test(
      NAME ${_target}
      COMMAND ${_target}
      WORKING_DIRECTORY "${hololink_BINARY_DIR}")
  endforeach()

  # Stop CI if close() deadlocks in blocking_monitor().
  set_tests_properties(hololink_cpu_roce_transceiver_ring_test PROPERTIES
    TIMEOUT 120)

  if(HOLOLINK_BUILD_EXPORT)
    add_subdirectory(cpu_roce_transport_consumer)
  endif()
endfunction()

ConfigureCpuRoceTransportTests()
