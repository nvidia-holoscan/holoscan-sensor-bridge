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

# Find or fetch gtest
include(hololink_deps/gtest)

# ##################################################################################################
# * compiler function -----------------------------------------------------------------------------

# This function takes in a test name, test sources and test dependencies (link libraries)
# and handles setting all of the associated
# properties and linking to build the test
function(ConfigureTest)
  cmake_parse_arguments("CMAKE_TEST"        # PREFIX
    ""                                      # OPTIONS
    "NAME"                                  # ONEVAL
    "SOURCES;LINK_LIBRARIES"                # MULTIVAL
    ${ARGN}
  )

  # The first parameter can implicitly be the NAME
  if(NOT CMAKE_TEST_NAME)
    list(GET ARGN 0 CMAKE_TEST_NAME)
  endif()

  add_executable(${CMAKE_TEST_NAME} ${CMAKE_TEST_SOURCES})

  set(BIN_DIR ${hololink_BINARY_DIR})

  set_target_properties(
    ${CMAKE_TEST_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${BIN_DIR}/gtests>"
  )

  target_link_libraries(${CMAKE_TEST_NAME}
    PRIVATE
      holoscan::core
      hololink
      GTest::gmock_main
      GTest::gtest_main
      ${CMAKE_TEST_LINK_LIBRARIES}
  )

  # Run the test from the main bin directory to access data as needed
  add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_property(TEST ${CMAKE_TEST_NAME} PROPERTY ENVIRONMENT
    "LD_LIBRARY_PATH=${BIN_DIR}/lib:$ENV{LD_LIBRARY_PATH}")

endfunction()

# ##################################################################################################
# * core tests ----------------------------------------------------------------------------------
if (HOLOLINK_BUILD_ROCE)
  ConfigureTest(
      hololink_iq_encoder_decoder_test
    SOURCES
      iq_encoder_decoder_test.cpp
    LINK_LIBRARIES
      hololink::operators::iq_dec
      hololink::operators::iq_enc
  )

  ConfigureTest(
      hololink_signal_generator_test
    SOURCES
      signal_generator_test.cpp
    LINK_LIBRARIES
      hololink::operators::sig_gen
  )

  ConfigureTest(
      hololink_roce_transmitter_test
    SOURCES
      roce_transmitter_test.cpp
    LINK_LIBRARIES
      hololink::operators::roce_transmitter
  )
  set_tests_properties(hololink_roce_transmitter_test PROPERTIES
    SKIP_REGULAR_EXPRESSION "Failed to find GID with IP based RoCE V2|No RoCE-capable InfiniBand device available"
  )

  if(HOLOLINK_BUILD_PYTHON)
    add_test(
      NAME hololink_python_basic_test
      COMMAND python3 -m pytest -vv -s
      WORKING_DIRECTORY ${hololink_SOURCE_DIR}
    )
  endif() # HOLOLINK_BUILD_PYTHON
endif() # HOLOLINK_BUILD_ROCE

# ##################################################################################################
# * hololink_module framework tests ---------------------------------------------------------------

# Stub module .so that publishes a TestServiceV1 wired to a host-published
# HostTestServiceV1 — proves the locator round-trips in both directions.
add_hololink_module(
  NAME hololink_module_stub_module
  NO_RECEIVER_CONSTRUCTS
  UUID stub
  SOURCES hololink_module_stub_module.cpp
)

# Stub .so that returns a deliberately-wrong ABI magic; built directly
# (without add_hololink_module) so its own abi_check overrides the one
# hololink::module_runtime would otherwise contribute.
add_library(hololink_module_abi_mismatch_module MODULE
  hololink_module_abi_mismatch_module.cpp)
set_target_properties(hololink_module_abi_mismatch_module PROPERTIES
  OUTPUT_NAME "hololink_module_abi_mismatch_module"
  PREFIX ""
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  POSITION_INDEPENDENT_CODE ON)
target_compile_features(hololink_module_abi_mismatch_module PRIVATE cxx_std_17)
target_link_libraries(hololink_module_abi_mismatch_module
  PRIVATE hololink::module_headers)

# Rename the good stub .so so the test can find it by a stable filename
# (add_hololink_module computes OUTPUT_NAME from UUID).
set_target_properties(hololink_module_stub_module PROPERTIES
  OUTPUT_NAME "hololink_module_stub_module")

ConfigureTest(
    hololink_module_framework_test
  SOURCES
    hololink_module_framework_test.cpp
  LINK_LIBRARIES
    hololink::module
)
target_compile_features(hololink_module_framework_test PRIVATE cxx_std_17)
# Reuse the host-side module's PCH — the test TU pulls in the same
# framework headers. Visibility is compatible (default on both: the
# test executable inherits CMAKE_CXX_VISIBILITY_PRESET, and
# hololink_module intentionally leaves visibility default).
if(TARGET hololink_module)
    target_precompile_headers(hololink_module_framework_test
        REUSE_FROM hololink_module)
endif()
add_dependencies(hololink_module_framework_test
  hololink_module_stub_module
  hololink_module_abi_mismatch_module)
set_property(TEST hololink_module_framework_test APPEND PROPERTY ENVIRONMENT
  "HOLOLINK_MODULE_TEST_MODULE_DIR=$<TARGET_FILE_DIR:hololink_module_stub_module>")

# ##################################################################################################
# * hololink_module Reactor + Logging singletons --------------------------------------------------

# Stub module .so that during init looks up the host's Reactor +
# LoggingInterfaceV1, schedules a Reactor callback, and emits an
# HSB_LOG_INFO line from inside it.
add_hololink_module(
  NAME hololink_module_singletons_stub_module
  NO_RECEIVER_CONSTRUCTS
  UUID singletons
  SOURCES hololink_module_singletons_stub_module.cpp
)
set_target_properties(hololink_module_singletons_stub_module PROPERTIES
  OUTPUT_NAME "hololink_module_singletons_stub_module")

ConfigureTest(
    hololink_module_singletons_test
  SOURCES
    hololink_module_singletons_test.cpp
  LINK_LIBRARIES
    hololink::module
    hololink::core
)
target_compile_features(hololink_module_singletons_test PRIVATE cxx_std_17)
add_dependencies(hololink_module_singletons_test
  hololink_module_singletons_stub_module)
set_property(TEST hololink_module_singletons_test APPEND PROPERTY ENVIRONMENT
  "HOLOLINK_MODULE_TEST_MODULE_DIR=$<TARGET_FILE_DIR:hololink_module_singletons_stub_module>")

# ##################################################################################################
# * hololink_module Enumeration -------------------------------------------------------------------

# Stub module .so claiming the reserved test UUID. NO_COMPAT_SUFFIX
# keeps the OUTPUT_NAME at the bare hololink_<uuid>.so form so
# Adapter::enumerate finds it via the no-compat-id fallback (the
# synthetic metadata the test feeds in carries no compat_id).
add_hololink_module(
  NAME hololink_module_enumeration_stub_module
  NO_RECEIVER_CONSTRUCTS
  UUID 01020304-0506-0708-090a-0b0c0d0e0f00
  NO_COMPAT_SUFFIX
  SOURCES hololink_module_enumeration_stub_module.cpp
)

ConfigureTest(
    hololink_module_enumeration_test
  SOURCES
    hololink_module_enumeration_test.cpp
  LINK_LIBRARIES
    hololink::module
)
target_compile_features(hololink_module_enumeration_test PRIVATE cxx_std_17)
add_dependencies(hololink_module_enumeration_test
  hololink_module_enumeration_stub_module)
set_property(TEST hololink_module_enumeration_test APPEND PROPERTY ENVIRONMENT
  "HOLOLINK_MODULE_TEST_MODULE_DIR=$<TARGET_FILE_DIR:hololink_module_enumeration_stub_module>")

# ##################################################################################################
# * hololink::module_core wrappers ----------------------------------------------------------------------

# Direct unit tests for the module/core/ default V1 wrappers that
# don't need live hardware (I2cLockImplV1, FrameMetadataV1).
ConfigureTest(
    hololink_module_core_test
  SOURCES
    hololink_module_core_test.cpp
  LINK_LIBRARIES
    hololink::module_core
)
target_compile_features(hololink_module_core_test PRIVATE cxx_std_17)

# ##################################################################################################
# * HSB-Lite module end-to-end ---------------------------------------------------------------------

# The hsb_lite module is built from module/hsb_lite/; its .so output
# name matches Adapter::enumerate's UUID-keyed lookup.
ConfigureTest(
    hololink_module_hsb_lite_test
  SOURCES
    hololink_module_hsb_lite_test.cpp
  LINK_LIBRARIES
    hololink::module
)
target_compile_features(hololink_module_hsb_lite_test PRIVATE cxx_std_17)
add_dependencies(hololink_module_hsb_lite_test hsb_lite hsb_lite_2510)
set_property(TEST hololink_module_hsb_lite_test APPEND PROPERTY ENVIRONMENT
  "HOLOLINK_MODULE_TEST_MODULE_DIR=$<TARGET_FILE_DIR:hsb_lite>")

# ##################################################################################################
# * Adapter bootp listener -------------------------------------------------------------------------

ConfigureTest(
    hololink_module_bootp_listener_test
  SOURCES
    hololink_module_bootp_listener_test.cpp
  LINK_LIBRARIES
    hololink::module
)
target_compile_features(hololink_module_bootp_listener_test PRIVATE cxx_std_17)

# Stub module .so + end-to-end gtest that crafts a synthetic bootp v2
# packet and verifies the listener's enumerate path delivers it to the
# stub's EnumerationInterfaceV1::update_metadata.
add_hololink_module(
    NAME hololink_module_bootp_stub_module
    NO_RECEIVER_CONSTRUCTS
    UUID 11223344-5566-7788-99aa-bbccddeeff00
    COMPAT 2603
    SOURCES hololink_module_bootp_stub_module.cpp
)
target_include_directories(hololink_module_bootp_stub_module
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

ConfigureTest(
    hololink_module_bootp_e2e_test
  SOURCES
    hololink_module_bootp_e2e_test.cpp
  LINK_LIBRARIES
    hololink::module
)
target_compile_features(hololink_module_bootp_e2e_test PRIVATE cxx_std_17)
target_include_directories(hololink_module_bootp_e2e_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
add_dependencies(hololink_module_bootp_e2e_test hololink_module_bootp_stub_module)
set_property(TEST hololink_module_bootp_e2e_test APPEND PROPERTY ENVIRONMENT
    "HOLOLINK_MODULE_TEST_MODULE_DIR=$<TARGET_FILE_DIR:hololink_module_bootp_stub_module>")

# ##################################################################################################
# * hololink_module Python bindings smoke test ----------------------------------------------------

if(HOLOLINK_BUILD_PYTHON)
    # The smoke test imports the in-build hololink_module package and
    # drives Adapter.enumerate against the hsb_lite .so.
    add_test(NAME hololink_module_python_smoke_test
        COMMAND python3 -m pytest -v
            --rootdir=${CMAKE_BINARY_DIR}/python
            ${CMAKE_CURRENT_SOURCE_DIR}/test_hololink_module_python.py)
    set_tests_properties(hololink_module_python_smoke_test PROPERTIES
        ENVIRONMENT
            "PYTHONPATH=${CMAKE_BINARY_DIR}/python;HOLOLINK_MODULE_TEST_MODULE_DIR=$<TARGET_FILE_DIR:hsb_lite>")

    # IMX274 sensor driver mock-based pytest. Verifies the driver
    # produces the expected I2C bus traffic for both expander outputs
    # without requiring real hardware.
    add_test(NAME hololink_module_imx274_python_test
        COMMAND python3 -m pytest -v --noconftest
            --rootdir=${CMAKE_BINARY_DIR}/python
            ${CMAKE_CURRENT_SOURCE_DIR}/test_hololink_module_imx274_python.py)
    set_tests_properties(hololink_module_imx274_python_test PROPERTIES
        ENVIRONMENT
            "PYTHONPATH=${CMAKE_BINARY_DIR}/python")

    # V1 service-interface pybind binding smoke test. Each abstract
    # interface exposes a Python alias and admits a Python subclass
    # whose pure-virtual overrides round-trip through the trampoline.
    add_test(NAME hololink_module_v1_bindings_python_test
        COMMAND python3 -m pytest -v --noconftest
            --rootdir=${CMAKE_BINARY_DIR}/python
            ${CMAKE_CURRENT_SOURCE_DIR}/test_hololink_module_v1_bindings_python.py)
    set_tests_properties(hololink_module_v1_bindings_python_test PROPERTIES
        ENVIRONMENT
            "PYTHONPATH=${CMAKE_BINARY_DIR}/python")
endif()

# ##################################################################################################
# * RoCE receiver operator (Holoscan-coupled) ------------------------------------------------------

# The C++ unit test links hololink::operators and exercises RoceReceiverOp
# directly, and the python smoke test below runs with --noconftest (so the
# accelerated_networking marker can't gate it); both therefore require the
# RoCE receiver operator to be built. The always-importable contract for
# `import hololink_module.operators` is verified instead by the full
# pytest sweep (conftest loaded, marker active) in RoCE-off configurations.
if(HOLOLINK_BUILD_ROCE AND HOLOLINK_MODULE_BUILD_OPERATORS)
    ConfigureTest(
        hololink_module_roce_receiver_op_test
      SOURCES
        hololink_module_roce_receiver_op_test.cpp
      LINK_LIBRARIES
        hololink::operators
        hololink::module
    )
    target_compile_features(hololink_module_roce_receiver_op_test PRIVATE cxx_std_17)

    if(HOLOLINK_BUILD_PYTHON)
        # Python smoke test verifying the operator's pybind class is
        # importable and the package layout is correct.
        add_test(NAME hololink_module_operators_python_smoke_test
            COMMAND python3 -m pytest -v --noconftest
                --rootdir=${CMAKE_BINARY_DIR}/python
                ${CMAKE_CURRENT_SOURCE_DIR}/test_hololink_module_operators_python.py)
        set_tests_properties(hololink_module_operators_python_smoke_test PROPERTIES
            ENVIRONMENT
                "PYTHONPATH=${CMAKE_BINARY_DIR}/python")
    endif()
endif()

# ##################################################################################################
# * IMX274 hardware smoke test ---------------------------------------------------------------------

# Runs the 4-channel module_quad_imx274_player.py example with
# --frame-limit=10 --headless against real HSB-Lite hardware (2 boards
# × 2 IMX274 cameras). The test is always registered when the build
# dependencies it needs are present (HOLOLINK_BUILD_PYTHON for the
# Python package, HOLOLINK_BUILD_ROCE + HOLOLINK_MODULE_BUILD_OPERATORS
# for the RoCE receiver operator). It is skipped at runtime unless the
# user opts in by setting HOLOLINK_TEST_IMX274=1 in the environment (the
# wrapper command exits 77 otherwise; SKIP_RETURN_CODE makes ctest report
# SKIPPED instead of FAILED). The "imx274" label also lets
# `ctest -L imx274` pick it out for explicit inclusion when running a
# focused hardware sweep.
if(HOLOLINK_BUILD_ROCE AND HOLOLINK_BUILD_PYTHON AND HOLOLINK_MODULE_BUILD_OPERATORS)
    # The example imports both the new `hololink_module` package
    # (staged under ${BUILD}/python/hololink_module/) and the legacy
    # `hololink` package (staged under
    # ${BUILD}/python/${CMAKE_INSTALL_LIBDIR}/hololink/ — the legacy
    # python/CMakeLists.txt copies its compiled .so + .py files
    # there). PYTHONPATH needs both directories so `import
    # hololink_module` and `import hololink` (for
    # `renesas_bajoran_lite_ts1` + the legacy operators the example
    # reuses) both resolve to the build tree. We derive the path
    # locally rather than reading the legacy build's cache variable
    # because tests/ is processed before python/ in the top-level
    # CMakeLists, so the cache variable isn't set yet at this point.
    set(_hololink_python_pkg_parent
        "${CMAKE_BINARY_DIR}/python/${CMAKE_INSTALL_LIBDIR}")

    add_test(NAME module_quad_imx274_player_test
        COMMAND bash -c
            "if [ \"\${HOLOLINK_TEST_IMX274:-0}\" = \"1\" ]; then \
                exec python3 ${hololink_SOURCE_DIR}/examples/module_quad_imx274_player.py \
                    --frame-limit=10 --headless \
                    --module-dir=${CMAKE_BINARY_DIR}/lib/hololink/modules; \
            else \
                echo \"SKIPPED: set HOLOLINK_TEST_IMX274=1 to run against real HSB-Lite hardware with IMX274 cameras attached\"; \
                exit 77; \
            fi"
        WORKING_DIRECTORY ${hololink_SOURCE_DIR})
    set_tests_properties(module_quad_imx274_player_test PROPERTIES
        SKIP_RETURN_CODE 77
        LABELS "imx274"
        ENVIRONMENT
            "PYTHONPATH=${CMAKE_BINARY_DIR}/python:${_hololink_python_pkg_parent}"
        TIMEOUT 120)
endif()

# ##################################################################################################
# * VB1940 hardware smoke test ---------------------------------------------------------------------

# Runs the single-camera examples/module_vb1940_player C++ binary with
# --frame-limit=10 --headless against a real Leopard VB1940 board. The
# test is registered when HOLOLINK_BUILD_ROCE + HOLOLINK_MODULE_BUILD_OPERATORS
# are set (the example links the module RoCE receiver operator, which
# needs both). It is skipped at runtime unless the user sets
# HOLOLINK_TEST_VB1940=1 (the wrapper exits 77; SKIP_RETURN_CODE makes
# ctest report SKIPPED rather than FAILED). The "vb1940" label lets
# `ctest -L vb1940` pick out the hardware-dependent slice for a focused sweep.
if(HOLOLINK_BUILD_ROCE AND HOLOLINK_MODULE_BUILD_OPERATORS)
    add_test(NAME module_vb1940_player_test
        COMMAND bash -c
            "if [ \"\${HOLOLINK_TEST_VB1940:-0}\" = \"1\" ]; then \
                exec $<TARGET_FILE:module_vb1940_player> \
                    --frame-limit=10 --headless \
                    --module-dir=${CMAKE_BINARY_DIR}/lib/hololink/modules; \
            else \
                echo \"SKIPPED: set HOLOLINK_TEST_VB1940=1 to run against a real Leopard VB1940 board\"; \
                exit 77; \
            fi"
        WORKING_DIRECTORY ${hololink_SOURCE_DIR})
    set_tests_properties(module_vb1940_player_test PROPERTIES
        SKIP_RETURN_CODE 77
        LABELS "vb1940"
        TIMEOUT 120)

    # Stereo smoke test: same gate, talks to the same HSB but exercises
    # both data planes simultaneously (left + right cameras share the
    # board, locator routes each channel to its own data-channel impl
    # via the metadata Vb1940Cam::use_sensor produces).
    add_test(NAME module_stereo_vb1940_player_test
        COMMAND bash -c
            "if [ \"\${HOLOLINK_TEST_VB1940:-0}\" = \"1\" ]; then \
                exec $<TARGET_FILE:module_stereo_vb1940_player> \
                    --frame-limit=10 --headless \
                    --module-dir=${CMAKE_BINARY_DIR}/lib/hololink/modules; \
            else \
                echo \"SKIPPED: set HOLOLINK_TEST_VB1940=1 to run against a real Leopard VB1940-AIO board\"; \
                exit 77; \
            fi"
        WORKING_DIRECTORY ${hololink_SOURCE_DIR})
    set_tests_properties(module_stereo_vb1940_player_test PROPERTIES
        SKIP_RETURN_CODE 77
        LABELS "vb1940"
        TIMEOUT 120)
endif()

# * Software (Linux) receiver hardware smoke tests --------------------------------------------------

# Run the single-camera module_linux_*_player C++ binaries with
# --frame-limit=10 --headless. Unlike the RoCE module players above these
# don't need ibverbs — they build and register in HOLOLINK_BUILD_ROCE=OFF
# configurations. Each test is guarded on its example target existing
# (the targets are created in examples/ under HOLOLINK_MODULE_BUILD_OPERATORS
# AND NOT HOLOLINK_BUILD_ONLY_NATIVE; examples/ is processed before
# tests/, so the target is visible here when it was built — and the
# guard cleanly drops the test when examples are disabled). Each is
# skipped at runtime unless its per-sensor switch is set
# (HOLOLINK_TEST_IMX274 / HOLOLINK_TEST_VB1940; the wrapper exits 77,
# SKIP_RETURN_CODE makes ctest report SKIPPED). The matching ctest label
# lets `ctest -L imx274` / `-L vb1940` select the hardware-dependent slice.
if(TARGET module_linux_imx274_player)
    add_test(NAME module_linux_imx274_player_test
        COMMAND bash -c
            "if [ \"\${HOLOLINK_TEST_IMX274:-0}\" = \"1\" ]; then \
                exec $<TARGET_FILE:module_linux_imx274_player> \
                    --frame-limit=10 --headless \
                    --module-dir=${CMAKE_BINARY_DIR}/lib/hololink/modules; \
            else \
                echo \"SKIPPED: set HOLOLINK_TEST_IMX274=1 to run against real HSB-Lite hardware with an IMX274 camera attached\"; \
                exit 77; \
            fi"
        WORKING_DIRECTORY ${hololink_SOURCE_DIR})
    set_tests_properties(module_linux_imx274_player_test PROPERTIES
        SKIP_RETURN_CODE 77
        LABELS "imx274"
        TIMEOUT 120)
endif()

if(TARGET module_linux_vb1940_player)
    add_test(NAME module_linux_vb1940_player_test
        COMMAND bash -c
            "if [ \"\${HOLOLINK_TEST_VB1940:-0}\" = \"1\" ]; then \
                exec $<TARGET_FILE:module_linux_vb1940_player> \
                    --frame-limit=10 --headless \
                    --module-dir=${CMAKE_BINARY_DIR}/lib/hololink/modules; \
            else \
                echo \"SKIPPED: set HOLOLINK_TEST_VB1940=1 to run against a real Leopard VB1940 board\"; \
                exit 77; \
            fi"
        WORKING_DIRECTORY ${hololink_SOURCE_DIR})
    set_tests_properties(module_linux_vb1940_player_test PROPERTIES
        SKIP_RETURN_CODE 77
        LABELS "vb1940"
        TIMEOUT 120)
endif()

# ##################################################################################################
# * emulator tests ---------------------------------------------------------------------------------
# Unit tests for the emulator components; gated on HOLOLINK_BUILD_EMULATOR so they
# build even when the ROCE transport layer is disabled (e.g. emulator-only builds
# consumed by external simulators like Isaac Sim).
if(HOLOLINK_BUILD_EMULATOR)
  ConfigureTest(
      hololink_vb1940_emulator_test
    SOURCES
      vb1940_emulator_test.cpp
    LINK_LIBRARIES
      hololink::emulation::sensors
  )

  ConfigureTest(
      hololink_hsb_emulator_vsync_stub_test
    SOURCES
      hsb_emulator_vsync_stub_test.cpp
    LINK_LIBRARIES
      hololink::emulation
  )
endif() # HOLOLINK_BUILD_EMULATOR
