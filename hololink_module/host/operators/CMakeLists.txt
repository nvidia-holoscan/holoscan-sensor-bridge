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

# Holoscan SDK is required. All Holoscan-coupled operators in this directory have now been
# ported to Holoscan SDK 5.0's Operator<Params> API (typed tensor/metadata ports,
# Contract/NotificationSource, TensorOutputSpec runtime-pool tensors) -- the HSDK4-targeting
# hololink_operators target this directory used to also carry (with its own
# find_package(holoscan 4.0 ...) call, dead in the real combined build -- the top-level
# CMakeLists.txt resolves 5.0 unconditionally before add_subdirectory(hololink_module) runs)
# is retired; only this single 5.0 resolution remains.
if(NOT TARGET holoscan::core AND NOT holoscan_SOURCE_DIR)
    find_package(holoscan 5.0 REQUIRED CONFIG PATHS "/opt/nvidia/holoscan")
endif()

# CRC operators (ComputeCrcOp / CheckCrcOp) compute a frame CRC32 via nvcomp. Built when
# nvcomp is available; cleanly disabled otherwise so environments without nvcomp can still
# build the rest of the operators.
option(HOLOLINK_MODULE_BUILD_CRC
    "Build the module CRC operators (requires nvcomp)" ON)
if(HOLOLINK_MODULE_BUILD_CRC)
    find_package(nvcomp QUIET PATHS /usr/lib/sbsa-linux-gnu/cmake/nvcomp)
    if(NOT nvcomp_FOUND)
        message(STATUS "nvcomp not found; disabling HOLOLINK_MODULE_BUILD_CRC")
        set(HOLOLINK_MODULE_BUILD_CRC OFF)
    endif()
endif()

add_library(hololink_module_operators_hsdk5 STATIC
    csi_to_bayer_op.cpp
    cuda_function_launcher.cpp
    hsb_controller.cpp
    hsb_controller_op.cpp
    image_processor_op.cpp
    linux_network_receiver.cpp
    linux_receiver_op.cpp
    module_operator_catalog.cpp
    packed_format_converter_op.cpp
    sensor_factory.cpp)
add_library(hololink::module::operators::hsdk5 ALIAS hololink_module_operators_hsdk5)

if(HOLOLINK_BUILD_ROCE)
    target_sources(hololink_module_operators_hsdk5 PRIVATE
        roce_network_receiver.cpp
        roce_receiver_op.cpp)
endif()
if(HOLOLINK_BUILD_FUSA)
    target_sources(hololink_module_operators_hsdk5 PRIVATE fusa_coe_capture_op.cpp)
endif()
if(HOLOLINK_MODULE_BUILD_CRC)
    target_sources(hololink_module_operators_hsdk5 PRIVATE compute_crc_op.cpp)
endif()

target_compile_features(hololink_module_operators_hsdk5 PRIVATE cxx_std_17)
target_include_directories(hololink_module_operators_hsdk5
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(hololink_module_operators_hsdk5
    PUBLIC
        hololink::module_headers
        # Pulls in holoscan::core, holoscan::schema, hololink, hololink::sensor,
        # CUDA::cuda_driver, CUDA::cudart, and (critically) the
        # capture_metadata_traits.hpp generated-header include path this
        # target's base_receiver_op.hpp/CaptureMetadataT usage needs -- the
        # same INTERFACE library the already-ported native receivers use, so
        # both trees agree on the CaptureMetadataT schema.
        hololink::operators::base_receiver_op
        # hololink/core/metadata.hpp and ingress_stall.hpp (BaseReceiverOp-derived receivers)
        # and hololink.hpp's ExtendedCounter (the RoCE receivers' frame_number).
        hololink::core
        # CsiToBayerOp's/ImageProcessorOp's/PackedFormatConverterOp's output port is
        # schema::ImageT (the image-domain payload) -- holoscan::schema (pulled in
        # transitively via base_receiver_op above) does not carry
        # holoscan/sensor_io/sensor_schema_package.hpp's own usage requirements to an
        # installed consumer (specifically, holoscan::detail::runtime_port_type_ops<ImageT>'s
        # definition); link it explicitly, matching native's identical requirement and
        # comment in src/hololink/operators/csi_to_bayer/CMakeLists.txt. Latent until
        # module_operator_catalog.cpp's registry rows became the first thing to actually link
        # this library into a full executable (the still-broken pybind11 extension never got
        # that far) -- not introduced by that file, just newly exposed by it.
        holoscan::sensor_io
        $<$<BOOL:${HOLOLINK_BUILD_FUSA}>:hololink::operators::fusa_coe_capture_core>
    PRIVATE
        # CsiToBayerOp's / PackedFormatConverterOp's / ImageProcessorOp's vendored CUDA
        # helpers (cuda_function_launcher.cpp) compile CUDA source with NVRTC and drive it
        # through the CUDA driver API.
        CUDA::cuda_driver
        CUDA::nvrtc
        hololink::module)
set_target_properties(hololink_module_operators_hsdk5 PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON)

if(HOLOLINK_BUILD_FUSA)
    target_compile_definitions(hololink_module_operators_hsdk5 PUBLIC HOLOLINK_BUILD_FUSA)
endif()
if(HOLOLINK_BUILD_ROCE)
    target_compile_definitions(hololink_module_operators_hsdk5 PUBLIC HOLOLINK_BUILD_ROCE)
endif()

# nvcomp + CUDA runtime for the CRC operators. Public so the nvcomp include dir
# (compute_crc_op.hpp uses nvcompBatchedCRC32Opts_t) and the HOLOLINK_MODULE_BUILD_CRC
# definition propagate to the pybind module.
if(HOLOLINK_MODULE_BUILD_CRC)
    target_link_libraries(hololink_module_operators_hsdk5
        PUBLIC
            nvcomp::nvcomp
            nvcomp::nvcomp_cpu
        PRIVATE
            CUDA::cudart)
    target_include_directories(hololink_module_operators_hsdk5 PUBLIC ${nvcomp_INCLUDE_DIR})
    target_compile_definitions(hololink_module_operators_hsdk5 PUBLIC HOLOLINK_MODULE_BUILD_CRC)
endif()

# Python bindings: SensorFactory/SensorDevice/NetworkReceiverFactory (the only pieces of
# operators_py.cpp still compatible with real bindings -- RoceReceiverOp/LinuxReceiverOp/
# FusaCoeCaptureOp/CsiToBayerOp/ImageProcessorOp/PackedFormatConverterOp/HsbControllerOp/
# ComputeCrcOp/CheckCrcOp's ported Operator<Params> shape has no holoscan::ArgList/Fragment*
# construction path for a pybind11-style trampoline to forward through any more) are now bound
# in python/hololink_module/_hololink_module.cpp, consolidated with the other hololink_module
# pybind11 extensions into one nanobind translation unit -- see
# hololink_module/host/CMakeLists.txt's own Python comment for where that target is defined
# and why. The operator classes themselves (CsiToBayerOp etc.) need no nb::class_ binding at
# all: they're resolved through module_operator_catalog.cpp's authoring_registry() rows, the
# same way hololink.operators.* is for the native family -- see
# python/hololink/operators/__init__.py's registry-discovery pattern, reused (with
# capture_metadata's own payload-identity gotcha -- see that file's own comment) by this
# directory's own python/__init__.py for hololink_module.operators.
if(HOLOLINK_BUILD_PYTHON)
    # The compiled extension itself (hololink_module_python target, linking this directory's
    # hololink_module_operators_hsdk5 library) is defined in hololink_module/host/CMakeLists.txt;
    # this just stages the operators sub-package's own __init__.py next to it, preserving the
    # old per-extension output layout exactly.
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/python/__init__.py"
        "${CMAKE_BINARY_DIR}/python/hololink_module/operators/__init__.py"
        COPYONLY)
endif()
