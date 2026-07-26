# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# HSB-Lite module pinned to compat-id 2510. Shares the FPGA UUID
# with module/hsb_lite/ but ships under the bare `hololink_<UUID>.so`
# filename (via NO_COMPAT_SUFFIX) so Adapter::load_module_for picks
# it up via the no-compat-id fallback path: boards reporting
# compat_id=2603 hit the compat-suffixed `hololink_<UUID>_2603.so`
# and load the sibling, while everything else (compat_id=2510, any
# other compat-id without a dedicated .so, or no compat_id at all)
# falls through to this module. Both modules publish the same
# HsbLiteInterfaceV1 supplement type.

add_hololink_module(
    NAME hsb_lite_2510
    UUID 889b7ce3-65a5-4247-8b05-4ff1904c3359
    NO_COMPAT_SUFFIX
    SOURCES
        module_entry.cpp
        # The base construct_roce_receiver / construct_linux_receiver TUs are
        # injected by add_hololink_module(); HsbLite2510Publisher adds only
        # its 2510 overrides (the pre-0x2603 software-receiver page-decode)
        # out-of-line here.
        ../core/roce_receiver_construct_2510.cpp
        ../core/linux_receiver_construct_2510.cpp
)

target_include_directories(hsb_lite_2510
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        $<TARGET_PROPERTY:hololink::hsb_lite::headers,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:hololink::module_core,INCLUDE_DIRECTORIES>)

target_link_libraries(hsb_lite_2510
    PRIVATE
        hololink::module_core)
