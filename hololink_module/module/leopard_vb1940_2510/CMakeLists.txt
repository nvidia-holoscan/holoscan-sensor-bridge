# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Leopard VB1940 module for the pre-0x2602 FPGA. Shares the FPGA UUID
# with module/leopard_vb1940/ but ships under the bare
# `hololink_<UUID>.so` filename (via NO_COMPAT_SUFFIX) so
# Adapter::load_module_for picks it up via the no-compat-id fallback
# path: boards reporting compat_id=2603 hit the compat-suffixed
# `hololink_<UUID>_2603.so` and load the sibling, while everything else
# (the 2510 boards, which report compat_id=0, plus any other compat-id
# without a dedicated .so) falls through to this module. Both modules
# publish the same LeopardVb1940InterfaceV1 supplement type, so
# Vb1940Cam reaches either unchanged.
#
# Documented exception to the "modules should not use NO_COMPAT_SUFFIX"
# rule; `module/hsb_lite_2510/` is the other.

add_hololink_module(
    NAME leopard_vb1940_2510
    UUID f1627640-b4dc-48af-a360-c55b09b3d230
    NO_COMPAT_SUFFIX
    SOURCES
        module_entry.cpp
        # The base construct_roce_receiver / construct_linux_receiver TUs are
        # injected by add_hololink_module(); LeopardVb1940_2510_Publisher adds
        # only its 2510 overrides out-of-line here.
        roce_receiver_construct_2510.cpp
        linux_receiver_construct_2510.cpp
)

target_include_directories(leopard_vb1940_2510
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../leopard_vb1940
        ${CMAKE_CURRENT_SOURCE_DIR}/../leopard_vb1940/include
        $<TARGET_PROPERTY:hololink::module_core,INCLUDE_DIRECTORIES>)

target_link_libraries(leopard_vb1940_2510
    PRIVATE
        hololink::module_core)
