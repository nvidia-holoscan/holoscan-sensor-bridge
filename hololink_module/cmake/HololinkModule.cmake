# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# add_hololink_module(
#     NAME <name> UUID <uuid> [COMPAT <compat-id> | NO_COMPAT_SUFFIX]
#     [NO_RECEIVER_CONSTRUCTS] SOURCES ...)
#
# Every HsbLite-based supplement module compiles the out-of-line
# HsbLitePublisher::construct_roce_receiver / construct_linux_receiver
# definitions. Rather than have each module name those TUs, the helper
# appends them (module/core's HOLOLINK_MODULE_RECEIVER_CONSTRUCT_SOURCES)
# automatically. A minimal, non-supplement module — e.g. a test stub that
# lacks the HSB-Lite supplement include path — passes NO_RECEIVER_CONSTRUCTS
# to skip them.
#
# Creates a module .so whose OUTPUT_NAME is built from UUID and a
# resolved compat-id:
#
#   hololink_<uuid>_<compat>.so        (default; compat from arg or DEFAULT_COMPAT)
#   hololink_<uuid>.so                 (NO_COMPAT_SUFFIX)
#
# Compat-id resolution order:
#   1. NO_COMPAT_SUFFIX option → no compat suffix is emitted. The
#      resulting `.so` is the Adapter's no-compat-id fallback when
#      enumeration metadata doesn't carry a compat_id and as a
#      catch-all for any compat_id whose dedicated `.so` is absent.
#   2. The COMPAT argument, if supplied.
#   3. The DEFAULT_COMPAT target property on hololink::module_core
#      (set in module/core/CMakeLists.txt).
# COMPAT and NO_COMPAT_SUFFIX are mutually exclusive; specifying
# neither (and having no DEFAULT_COMPAT property) is a configure error.
#
# The target privately absorbs hololink::module_headers +
# hololink::module_runtime. Module sources compile on top of those.
function(add_hololink_module)
    set(_options NO_COMPAT_SUFFIX NO_RECEIVER_CONSTRUCTS)
    set(_oneval NAME UUID COMPAT)
    set(_multival SOURCES)
    cmake_parse_arguments(_HM "${_options}" "${_oneval}" "${_multival}" ${ARGN})

    if(NOT _HM_NAME)
        message(FATAL_ERROR "add_hololink_module: NAME is required")
    endif()
    if(NOT _HM_UUID)
        message(FATAL_ERROR "add_hololink_module: UUID is required")
    endif()
    if(NOT _HM_SOURCES)
        message(FATAL_ERROR "add_hololink_module: SOURCES is required")
    endif()
    if(_HM_NO_COMPAT_SUFFIX AND _HM_COMPAT)
        message(FATAL_ERROR
            "add_hololink_module(${_HM_NAME}): NO_COMPAT_SUFFIX and COMPAT "
            "are mutually exclusive — NO_COMPAT_SUFFIX suppresses the "
            "compat-id suffix entirely, so a COMPAT value would be ignored.")
    endif()

    # Inject the shared receiver-construct TUs unless the caller opts out.
    set(_module_sources ${_HM_SOURCES})
    if(NOT _HM_NO_RECEIVER_CONSTRUCTS)
        if(NOT DEFINED HOLOLINK_MODULE_RECEIVER_CONSTRUCT_SOURCES)
            message(FATAL_ERROR
                "add_hololink_module(${_HM_NAME}): the receiver-construct sources "
                "are not defined yet — add_subdirectory(module/core) must run before "
                "declaring modules, or pass NO_RECEIVER_CONSTRUCTS for a minimal "
                "(non-supplement) module.")
        endif()
        list(APPEND _module_sources ${HOLOLINK_MODULE_RECEIVER_CONSTRUCT_SOURCES})
    endif()

    add_library(${_HM_NAME} MODULE ${_module_sources})

    if(_HM_NO_COMPAT_SUFFIX)
        set_target_properties(${_HM_NAME} PROPERTIES
            OUTPUT_NAME "hololink_${_HM_UUID}"
            PREFIX "")
    else()
        if(_HM_COMPAT)
            set(_resolved_compat "${_HM_COMPAT}")
        else()
            if(NOT TARGET hololink_module_core)
                message(FATAL_ERROR
                    "add_hololink_module(${_HM_NAME}): no COMPAT supplied and "
                    "the hololink_module_core target is not defined yet — call "
                    "add_subdirectory(module/core) before declaring per-board "
                    "modules, or pass COMPAT explicitly.")
            endif()
            get_target_property(_resolved_compat hololink_module_core DEFAULT_COMPAT)
            if(NOT _resolved_compat OR _resolved_compat STREQUAL "_resolved_compat-NOTFOUND")
                message(FATAL_ERROR
                    "add_hololink_module(${_HM_NAME}): no COMPAT supplied and "
                    "hololink_module_core's DEFAULT_COMPAT property is not set.")
            endif()
        endif()
        set_target_properties(${_HM_NAME} PROPERTIES
            OUTPUT_NAME "hololink_${_HM_UUID}_${_resolved_compat}"
            PREFIX "")
    endif()

    target_compile_features(${_HM_NAME} PRIVATE cxx_std_17)
    set_target_properties(${_HM_NAME} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        POSITION_INDEPENDENT_CODE ON
        # Canonical staging dir: every hololink_<UUID>.so lands in
        # one place under the build tree, matching the install path
        # (lib/hololink/modules) so applications can reach them via
        # `--module-dir $BUILD/lib/hololink/modules` without knowing
        # which sub-directory under hololink_module/module/ owned
        # the source.
        LIBRARY_OUTPUT_DIRECTORY
            "${CMAKE_BINARY_DIR}/lib/hololink/modules")

    target_link_libraries(${_HM_NAME}
        PRIVATE
            hololink::module_headers
            hololink::module_runtime)

    # Reuse hololink::module_core's precompiled headers — every supplement +
    # test-stub .so transitively pulls in the same framework +
    # module_core headers, so skipping the re-parse pays back across
    # every per-board .so the loader builds.
    if(TARGET hololink_module_core)
        target_precompile_headers(${_HM_NAME} REUSE_FROM hololink_module_core)
    endif()
endfunction()
