# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Holoscan-coupled operators built against the module V1 interfaces.
# Gated by HOLOLINK_MODULE_BUILD_OPERATORS so framework-only build
# environments without Holoscan can configure the module without
# pulling Holoscan in.

if(NOT HOLOLINK_MODULE_BUILD_OPERATORS)
    return()
endif()

# Holoscan SDK is required.
if(NOT TARGET holoscan::core)
    find_package(holoscan 4.0 REQUIRED CONFIG PATHS "/opt/nvidia/holoscan")
endif()

# C++ operator implementations. Each operator is enabled per
# environment capability. The Linux software receiver operator, the
# CSI->Bayer converter operator, the image-processing operator, and the
# packed-format converter operator need no ibverbs and so are built
# whenever the operators tree is; the RoCE receiver and FUSA CoE capture
# operators are added only when their capability is available. The
# library is therefore created unconditionally here (it always has at
# least the Linux receiver, CSI->Bayer, image-processing, and
# packed-format converter operators).
add_library(hololink_operators STATIC
    csi_to_bayer_op.cpp
    cuda_function_launcher.cpp
    hsb_controller.cpp
    hsb_controller_op.cpp
    image_processor_op.cpp
    linux_network_receiver.cpp
    linux_receiver_op.cpp
    packed_format_converter_op.cpp
    sensor_factory.cpp)
add_library(hololink::operators ALIAS hololink_operators)

# The RoCE receiver operator's TU pulls in the legacy ibverbs receiver
# (via hololink::module) and is only meaningful with RoCE. The RoCE
# NetworkReceiver (the reconnection controller's RoCE data plane) shares
# that dependency.
if(HOLOLINK_BUILD_ROCE)
    target_sources(hololink_operators PRIVATE
        roce_network_receiver.cpp
        roce_receiver_op.cpp)
endif()
if(HOLOLINK_BUILD_FUSA)
    target_sources(hololink_operators PRIVATE fusa_coe_capture_op.cpp)
endif()

# CRC operators (ComputeCrcOp / CheckCrcOp) compute a frame CRC32 via
# nvcomp. Built when nvcomp is available; cleanly disabled otherwise so
# environments without nvcomp can still build the rest of the operators.
option(HOLOLINK_MODULE_BUILD_CRC
    "Build the module CRC operators (requires nvcomp)" ON)
if(HOLOLINK_MODULE_BUILD_CRC)
    find_package(nvcomp QUIET PATHS /usr/lib/sbsa-linux-gnu/cmake/nvcomp)
    if(nvcomp_FOUND)
        target_sources(hololink_operators PRIVATE compute_crc_op.cpp)
    else()
        message(STATUS "nvcomp not found; disabling HOLOLINK_MODULE_BUILD_CRC")
        set(HOLOLINK_MODULE_BUILD_CRC OFF)
    endif()
endif()

target_compile_features(hololink_operators PRIVATE cxx_std_17)
target_include_directories(hololink_operators
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(hololink_operators
    PUBLIC
        hololink::module_headers
        holoscan::core
        # hololink::core (legacy) is needed only by the FUSA operator, whose
        # header pulls in hololink/core/csi_controller.hpp. The always-built
        # CsiToBayerOp no longer needs it: its CUDA helpers
        # (CudaFunctionLauncher / CudaContextScopedPush) are now module-owned
        # in cuda_function_launcher.cpp.
        $<$<BOOL:${HOLOLINK_BUILD_FUSA}>:hololink::core>
        $<$<BOOL:${HOLOLINK_BUILD_FUSA}>:hololink::operators::fusa_coe_capture_core>
    PRIVATE
        # CsiToBayerOp's vendored CUDA helpers compile CUDA source with NVRTC
        # and drive it through the CUDA driver API.
        CUDA::cuda_driver
        CUDA::nvrtc
        hololink::module)
set_target_properties(hololink_operators PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON)

if(HOLOLINK_BUILD_FUSA)
    target_compile_definitions(hololink_operators PUBLIC HOLOLINK_BUILD_FUSA)
endif()

if(HOLOLINK_BUILD_ROCE)
    target_compile_definitions(hololink_operators PUBLIC HOLOLINK_BUILD_ROCE)
endif()

# nvcomp + CUDA runtime for the CRC operators. Public so the nvcomp
# include dir (compute_crc_op.hpp uses nvcompBatchedCRC32Opts_t) and the
# HOLOLINK_MODULE_BUILD_CRC definition propagate to the pybind module.
if(HOLOLINK_MODULE_BUILD_CRC)
    target_link_libraries(hololink_operators
        PUBLIC
            nvcomp::nvcomp
            nvcomp::nvcomp_cpu
        PRIVATE
            CUDA::cudart)
    target_include_directories(hololink_operators PUBLIC ${nvcomp_INCLUDE_DIR})
    target_compile_definitions(hololink_operators PUBLIC HOLOLINK_MODULE_BUILD_CRC)
endif()

# pybind11 extension exposing the operators to Python. Always built when
# Python is enabled (independent of any single operator's capability) so
# `import hololink_module.operators` always succeeds; operators_py.cpp
# registers only the operators this configuration compiled in.
if(HOLOLINK_BUILD_PYTHON)
    include(hololink_deps/pybind11)

    pybind11_add_module(_hololink_module_operators MODULE
        python/operators_py.cpp)
    # holoscan::pybind11 makes this compile against the same pybind11
    # (incl. ABI tag and internals capsule) the sibling _hololink_py_module
    # extension uses; see the matching link in
    # hololink_module/host/CMakeLists.txt. Without it, holoscan::Operator
    # (registered by holoscan's own .so) can't be referenced as a base
    # type at module load time.
    #
    # hololink::operators is linked by existence rather than by any single
    # operator's capability: it is created only if at least one operator
    # was built, and TARGET_NAME_IF_EXISTS resolves to empty when no
    # operator was built so the link drops out cleanly.
    # operators_py.cpp registers only the operators this build compiled in.
    # hololink::core (legacy) is linked only for FUSA: its operator header
    # needs hololink/core/csi_controller.hpp. The always-built CsiToBayerOp is
    # self-contained now that its CUDA helpers are module-owned, so the
    # framework-only configuration links no legacy core here.
    target_link_libraries(_hololink_module_operators
        PRIVATE
            hololink::module
            holoscan::core
            holoscan::pybind11
            $<$<BOOL:${HOLOLINK_BUILD_FUSA}>:hololink::core>
            $<TARGET_NAME_IF_EXISTS:hololink::operators>)
    target_compile_features(_hololink_module_operators PRIVATE cxx_std_17)

    if(HOLOLINK_BUILD_ROCE)
        target_compile_definitions(_hololink_module_operators PRIVATE HOLOLINK_BUILD_ROCE)
    endif()
    if(HOLOLINK_BUILD_FUSA)
        target_compile_definitions(_hololink_module_operators PRIVATE HOLOLINK_BUILD_FUSA)
    endif()
    if(HOLOLINK_MODULE_BUILD_CRC)
        target_compile_definitions(_hololink_module_operators PRIVATE HOLOLINK_MODULE_BUILD_CRC)
    endif()

    set_target_properties(_hololink_module_operators PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
            "${CMAKE_BINARY_DIR}/python/hololink_module/operators")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/python/__init__.py"
        "${CMAKE_BINARY_DIR}/python/hololink_module/operators/__init__.py"
        COPYONLY)
endif()
