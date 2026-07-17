# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Public headers consumed by module callers and by every loaded module.
# The fmt header-only target is required so HSB_LOG_* macros in
# logging.hpp compile in any TU that includes them.
add_library(hololink_module_headers INTERFACE)
add_library(hololink::module_headers ALIAS hololink_module_headers)
target_include_directories(hololink_module_headers
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(hololink_module_headers INTERFACE cxx_std_17)
target_link_libraries(hololink_module_headers
    INTERFACE
        fmt::fmt-header-only)

# Diagnostics carry project-root-relative source paths, not the absolute or
# build-dir paths __FILE__ would otherwise render. -fmacro-prefix-map
# rewrites the project-root prefix out of __FILE__ (and therefore
# HOLOLINK_MODULE_FILE, used by logging.hpp / cuda_unique.hpp). It is added
# as an INTERFACE compile option on adapter_headers so every consumer picks
# it up uniformly — applied identically to each target's precompiled header
# and its sources, which keeps the PCH valid (a per-source -D of the path
# cannot: it diverges from the PCH's preprocessor state and is dropped with
# -Winvalid-pch). Requires GCC >= 8 / Clang >= 10.
target_compile_options(hololink_module_headers
    INTERFACE -fmacro-prefix-map=${CMAKE_SOURCE_DIR}/=)

# Export the public module header surface for external applications. The
# sensor drivers (hololink::module::sensors::*) carry adapter_headers as a
# PUBLIC usage requirement, so it must join the same HololinkTargets export
# set for their generated imported targets to resolve. Like the legacy
# hololink::common target it leaves fmt::fmt-header-only in its interface;
# HololinkConfig.cmake resolves it the same way for both. (The baked
# -fmacro-prefix-map path is the build machine's CMAKE_SOURCE_DIR — a no-op
# prefix for an installed consumer, harmless.)
include(GNUInstallDirs)
set_target_properties(hololink_module_headers PROPERTIES
    EXPORT_NAME module_headers)
install(TARGETS hololink_module_headers
    EXPORT HololinkTargets
    COMPONENT hololink-module)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/hololink
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT hololink-module)

# Module-side framework glue: Module + Publisher classes, the
# hololink_module_get_abi_check symbol, the ServicePublisher helper,
# and the per-binary HSB_LOG cache pointer. Every loaded module .so
# absorbs its own private copy via hololink::module_runtime +
# RTLD_LOCAL — so the cache pointer (and the Publisher's per-binary
# singleton) are unique to that .so.
add_library(hololink_module_runtime STATIC
    src_module/module_base.cpp
    src_module/logging_cache.cpp)
add_library(hololink::module_runtime ALIAS hololink_module_runtime)
target_compile_features(hololink_module_runtime PRIVATE cxx_std_17)
target_link_libraries(hololink_module_runtime
    PUBLIC
        hololink::module_headers)
set_target_properties(hololink_module_runtime PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON)

# Export the module-side framework glue. It defines hsb_logger_cache /
# set_hsb_logger_cache (logging_cache.cpp), the symbols the HSB_LOG_* macros
# in any consumer TU resolve against — so the sensor archives are not
# independently linkable without it. hololink::module carries it as a
# (static-propagated) link dependency, so it must join the same export set.
set_target_properties(hololink_module_runtime PROPERTIES
    EXPORT_NAME module_runtime)
install(TARGETS hololink_module_runtime
    EXPORT HololinkTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT hololink-module)
# Precompile the framework headers every module-side TU pulls in.
# Used directly by this archive's own .cpp files; not REUSE_FROM'd
# (downstream module .so targets reuse hololink_module_core's PCH instead,
# which is a superset).
target_precompile_headers(hololink_module_runtime
    PRIVATE
        <memory>
        <mutex>
        <stdexcept>
        <string>
        <unordered_map>
        <hololink/module/logging.hpp>
        <hololink/module/module.hpp>
        <hololink/module/publisher.hpp>
        <hololink/module/service.hpp>
        <hololink/module/status.h>)

# Host-side module implementation: Adapter singleton, dlopen + ABI
# check, host-side Module wiring, plus the host's own Logging sink,
# Reactor poll loop, and bootp-v2 listener (all in src/). This
# library deliberately links no legacy hololink::core:
# the infrastructure it needs (logging, reactor, bootp parsing, route
# lookup, the ReceiverMemoryDescriptor CUDA helpers) is module-owned, so
# applications that link the host never pull in src/hololink/core — that
# code reaches the process only inside loaded module .so files. (PAGE_SIZE
# and round_up are header-only in hololink/module/page_size.hpp so loaded
# modules can use them without linking the module host library.)
find_package(CUDAToolkit REQUIRED)
add_library(hololink_module STATIC
    src/adapter.cpp
    src/bootp.cpp
    src/hololink.cpp
    src/logging_impl.cpp
    src/networking.cpp
    src/oscillator.cpp
    src/reactor_impl.cpp
    src/receiver_memory_descriptor.cpp
    src/service.cpp)
# ibv_device.cpp (ibv_device_for_peer) is the only RoCE-coupled piece of
# the host library; its sole caller is the RoCE receiver operator, which
# is itself built only with RoCE. HOLOLINK_BUILD_ROCE is exposed PUBLIC
# so module consumers can gate their RoCE-conditional binding code on
# it: the core/operators pybind extensions (and the framework test) link
# hololink::module and reuse its precompiled header, so the macro stays
# consistent across all of them with this one definition.
if(HOLOLINK_BUILD_ROCE)
    find_package(ibverbs REQUIRED)
    target_sources(hololink_module PRIVATE src/ibv_device.cpp)
    target_link_libraries(hololink_module PRIVATE ${ibverbs_LIBRARIES})
    target_compile_definitions(hololink_module PUBLIC HOLOLINK_BUILD_ROCE)
endif()
add_library(hololink::module ALIAS hololink_module)
target_compile_features(hololink_module PRIVATE cxx_std_17)
target_link_libraries(hololink_module
    PUBLIC
        hololink::module_headers
    PRIVATE
        CUDA::cuda_driver
        hololink::module_runtime
        ${CMAKE_DL_LIBS})
# Intentionally no CXX_VISIBILITY_PRESET hidden — this static lib is
# linked into the Python extension (_hololink_py_module.so), which
# exports its C++ typeinfo so sibling pybind11 extensions can match
# std::type_index across modules. Hidden visibility here would emit
# hidden weak typeinfo that wins the linker merge over the binding
# TU's default-visibility copy. The legacy hololink::core static lib
# (src/hololink/common/CMakeLists.txt) follows the same pattern.
set_target_properties(hololink_module PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

# Export the host-side runtime so external applications can obtain the V1
# service interfaces the sensor drivers consume (Adapter, enumeration, the
# HSB logger). As a static archive its PRIVATE deps propagate to consumers as
# $<LINK_ONLY:...>: CUDA::cuda_driver (imported — consumer find_package's
# CUDAToolkit, per the project's no-find_dependency config convention) and
# hololink::module_runtime (exported above). ${CMAKE_DL_LIBS} / ibverbs are
# plain system libs.
set_target_properties(hololink_module PROPERTIES
    EXPORT_NAME module)
install(TARGETS hololink_module
    EXPORT HololinkTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT hololink-module)
# Precompile the framework headers every host-side module TU pulls
# in. Downstream targets that share the same visibility profile
# (pybind extensions + the framework test executable) reuse this PCH
# via REUSE_FROM.
target_precompile_headers(hololink_module
    PRIVATE
        <memory>
        <stdexcept>
        <string>
        <vector>
        <hololink/module/adapter.hpp>
        <hololink/module/enumeration_metadata.hpp>
        <hololink/module/logging.hpp>
        <hololink/module/module.hpp>
        <hololink/module/publisher.hpp>
        <hololink/module/service.hpp>
        <hololink/module/status.h>)

# pybind11 extension exposing the core module surface to Python.
# Disabled when HOLOLINK_BUILD_PYTHON=OFF (the same gate the parent
# project uses for its own Python bindings). Uses the parent project's
# hololink_deps/pybind11 helper, which finds an installed pybind11 if
# present and falls back to FetchContent otherwise.
if(HOLOLINK_BUILD_PYTHON)
    include(hololink_deps/pybind11)

    pybind11_add_module(_hololink_py_module MODULE
        python/hololink_module_py.cpp)
    # Link holoscan::pybind11 so this extension compiles against the
    # same pybind11 headers / internals capsule that Holoscan's own
    # extensions (and the sibling _hololink_module_operators) use.
    # Without it the operators extension can't see hololink_module's
    # registered types — they end up in a different pybind11 internals
    # capsule than holoscan::Operator and friends, so cross-module
    # std::type_index lookup fails with "incompatible constructor
    # arguments" naming the C++ types as raw "hololink::module::...".
    target_link_libraries(_hololink_py_module
        PRIVATE
            hololink::module
            holoscan::pybind11)
    target_compile_features(_hololink_py_module PRIVATE cxx_std_17)
    # Reuse the host-side module's PCH — the binding TU pulls in the
    # same framework headers it precompiles. Visibility is compatible
    # (both leave default visibility for the typeinfo-export reason
    # documented above).
    target_precompile_headers(_hololink_py_module REUSE_FROM hololink_module)

    # Stage __init__.py next to the .so so
    # PYTHONPATH=$<TARGET_FILE_DIR>/.. imports the package as
    # `hololink_module`. The C++ example binaries land at
    # <site-packages>/bin/<name> via examples/CMakeLists.txt's
    # install(TARGETS …) rule; the demo container's Dockerfile
    # symlinks that dir onto /usr/local/bin/ so they appear on PATH
    # without per-binary plumbing here.
    set_target_properties(_hololink_py_module PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
            "${CMAKE_BINARY_DIR}/python/hololink_module")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/python/__init__.py"
        "${CMAKE_BINARY_DIR}/python/hololink_module/__init__.py" COPYONLY)

    # Export typeinfo from this extension so the sibling extensions
    # (_hololink_module_operators, _hololink_module_<module>) can
    # find a matching std::type_index when they cast user-supplied
    # objects to shared_ptr<RoceDataChannelInterface> etc. pybind11
    # defaults to -fvisibility=hidden which hides the weak typeinfo
    # symbols and breaks cross-module type lookup; the legacy
    # _hololink_core module uses the same override for the same reason
    # (python/hololink/hololink_core/CMakeLists.txt:33-39).
    set_property(TARGET _hololink_py_module PROPERTY CXX_VISIBILITY_PRESET default)
    set_property(TARGET _hololink_py_module PROPERTY VISIBILITY_INLINES_HIDDEN OFF)
    target_link_options(_hololink_py_module PRIVATE "LINKER:--export-dynamic")
endif()
