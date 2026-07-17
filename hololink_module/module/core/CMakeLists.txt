# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# hololink::module_core — V1-shaped wrappers over the existing
# src/hololink/core/ backend. Every module .so absorbs its own
# private copy via add_hololink_module() + RTLD_LOCAL.
add_library(hololink_module_core STATIC
    frame_metadata_default.cpp
    hololink_default.cpp)
add_library(hololink::module_core ALIAS hololink_module_core)

target_compile_features(hololink_module_core PRIVATE cxx_std_17)
target_include_directories(hololink_module_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(hololink_module_core
    PUBLIC
        hololink::module_headers
        hololink::module_runtime
        hololink::core)

# The publisher headers declare construct_roce_receiver (always called
# from the construct_service chain) but define it out-of-line in
# roce_receiver_construct.cpp / roce_receiver_construct_2510.cpp. Defining
# those methods needs the full publisher class, which transitively
# includes a board supplement header (hsb_lite.hpp), so they are compiled
# into each per-board module (which carries the supplement include path)
# rather than here in core. add_hololink_module() injects the base pair
# automatically (see HOLOLINK_MODULE_RECEIVER_CONSTRUCT_SOURCES below), so
# modules no longer name them; boards that need version-specific overrides
# (e.g. hsb_lite_2510) still add their *_2510 variants explicitly.
#
# The construct body is selected by HOLOLINK_BUILD_ROCE. Rather than
# stamp that macro per-source, propagate it as a PUBLIC define on
# hololink::module_core (mirroring hololink::module on the host): every
# per-board module inherits it through the link, and because the macro
# is baked into this target's PCH too, the PCH the modules reuse stays
# flag-consistent. hololink::module_core also propagates the legacy ibverbs
# receiver so the construct TU links it when RoCE is on.
if(HOLOLINK_BUILD_ROCE)
    target_compile_definitions(hololink_module_core PUBLIC HOLOLINK_BUILD_ROCE)
    target_link_libraries(hololink_module_core PUBLIC hololink::operators::roce_receiver)
endif()

# The software (Linux) receiver class. Compiled directly into
# hololink::module_core — rather than linking the legacy
# hololink::operators::linux_receiver target — so every module .so
# absorbs the receiver WITHOUT that target's spurious
# base_receiver_op -> Holoscan interface coupling. The software receiver
# is meant to run on hosts with no Holoscan and no RoCE, so it must not
# drag Holoscan into the module link. Unconditional: the
# construct_linux_receiver branch (linux_receiver_construct.cpp, compiled
# per-board like the RoCE construct TU) always publishes a
# LinuxReceiverV1. CUDA::cuda_driver backs the receiver's host->device
# frame copy.
target_sources(hololink_module_core
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../src/hololink/operators/linux_receiver/linux_receiver.cpp)
target_link_libraries(hololink_module_core PUBLIC CUDA::cuda_driver)

# The out-of-line receiver-construct TUs every HsbLite-based module
# compiles. add_hololink_module() appends these to each module's sources
# automatically (unless NO_RECEIVER_CONSTRUCTS is passed), so modules don't
# name them. CACHE INTERNAL so the paths are visible from the sibling
# module/ subdirectories that call the helper.
set(HOLOLINK_MODULE_RECEIVER_CONSTRUCT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/roce_receiver_construct.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/linux_receiver_construct.cpp"
    CACHE INTERNAL "Receiver-construct TUs injected into every hololink module")

set_target_properties(hololink_module_core PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON)

# Precompile the framework + module_core headers every module-side TU
# transitively pulls in. add_hololink_module() wires
# `target_precompile_headers(<module> REUSE_FROM hololink_module_core)` so
# every supplement + test-stub .so picks this up automatically.
target_precompile_headers(hololink_module_core
    PRIVATE
        <cstdint>
        <memory>
        <optional>
        <stdexcept>
        <string>
        <utility>
        <vector>
        <hololink/module/data_channel.hpp>
        <hololink/module/enumeration_metadata.hpp>
        <hololink/module/hololink.hpp>
        <hololink/module/i2c_lock.hpp>
        <hololink/module/logging.hpp>
        <hololink/module/module.hpp>
        <hololink/module/publisher.hpp>
        <hololink/module/service.hpp>
        <hololink/module/status.h>)

# Head-of-tree default compat-id. add_hololink_module() reads this
# property when a per-board module's CMake invocation does not pass
# its own COMPAT override, so the resulting .so file name uses the
# default unless the board explicitly overrides.
set_target_properties(hololink_module_core PROPERTIES
    DEFAULT_COMPAT 2603)
