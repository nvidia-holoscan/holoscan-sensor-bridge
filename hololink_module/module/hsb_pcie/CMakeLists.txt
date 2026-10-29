# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# hsb_pcie module: the in-process PCIe transport for the hololink_module
# framework. Everything PCIe lives in this directory: the control plane
# (ECB bridge over BAR0), the data channel, the receiver and its device
# access, the sysfs enumerator, and the kernel driver's ABI mirrors.
#
# Targets:
#   hololink::hsb_pcie::headers
#                             INTERFACE target for the module-owned public
#                             headers under include/hololink/module/hsb_pcie/:
#                             pcie_data_channel.hpp (PcieDataChannelInterfaceV1
#                             + PcieReceiverAttachmentV1) and pcie_receiver.hpp
#                             (PcieReceiverInterfaceV1), the pattern of
#                             hololink::hsb_lite::headers. Host code --
#                             PcieReceiverOp, the examples -- links this and
#                             never the module .so.
#   hololink::hsb_pcie::abi_headers
#                             INTERFACE target for this directory's header-only
#                             pieces that are not part of the service surface:
#                             the kernel driver's uapi / bootp mirrors
#                             (hololink_pcie_uapi.h, hololink_pcie_bootp.h,
#                             checked byte-identical to src/ehsb/kmod/include at
#                             configure time), the BAR0 map (pcie_regs.hpp) and
#                             the sysfs helpers (pcie_sysfs.hpp). For the tests
#                             and the identity tool; not installed.
#   hololink::hsb_pcie::core  STATIC archive with everything that does
#                             not need the module framework's dlopen
#                             path -- ECB codec, address policy, software
#                             APB sequencer, endpoint interface, the
#                             driver-backed PcieEndpoint, the loopback
#                             endpoint, PcieHololink, and the receive path's
#                             backing (PcieDeviceAccess, PcieReceiver) -- so
#                             the codec / hololink / receiver tests link it
#                             directly.
#   hsb_pcie                  hololink_<UUID>.so via add_hololink_module():
#                             module entry, PcieHololinkV1, PcieDataChannelV1,
#                             PcieReceiverV1 and the publisher on top of
#                             hololink::module_core.

include(GNUInstallDirs)
find_package(CUDAToolkit REQUIRED)

# ---- public service headers ------------------------------------------------
# Header-only: linking it drags in no module .so, so hololink::operators and
# out-of-tree applications can call PcieReceiverInterfaceV1::get_service /
# PcieDataChannelInterfaceV1::get_service while staying free of the module
# and of the legacy core. Installed alongside the framework headers.
add_library(hololink_hsb_pcie_headers INTERFACE)
add_library(hololink::hsb_pcie::headers ALIAS hololink_hsb_pcie_headers)
target_include_directories(hololink_hsb_pcie_headers
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(hololink_hsb_pcie_headers
    INTERFACE
        hololink::module_headers)
target_compile_features(hololink_hsb_pcie_headers INTERFACE cxx_std_17)
set_target_properties(hololink_hsb_pcie_headers PROPERTIES
    EXPORT_NAME hsb_pcie::headers)
install(TARGETS hololink_hsb_pcie_headers
    EXPORT HololinkTargets
    COMPONENT hololink-module)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/hololink
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT hololink-module)

# ---- kernel ABI mirrors, BAR0 map, sysfs helpers (header-only) --------------
add_library(hololink_hsb_pcie_abi_headers INTERFACE)
add_library(hololink::hsb_pcie::abi_headers ALIAS hololink_hsb_pcie_abi_headers)
target_include_directories(hololink_hsb_pcie_abi_headers
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(hololink_hsb_pcie_abi_headers
    INTERFACE
        hololink::module_headers)
target_compile_features(hololink_hsb_pcie_abi_headers INTERFACE cxx_std_17)

# ---- control plane + receive-path backing ------------------------------------
add_library(hololink_hsb_pcie_core STATIC
    address_policy.cpp
    apb_sequencer.cpp
    loopback_pcie_endpoint.cpp
    pcie_device_access.cpp
    pcie_endpoint.cpp
    pcie_hololink.cpp
    pcie_receiver.cpp)
add_library(hololink::hsb_pcie::core ALIAS hololink_hsb_pcie_core)

target_compile_features(hololink_hsb_pcie_core PUBLIC cxx_std_17)

# ${hololink_SOURCE_DIR}/src provides <hololink/core/*.hpp> (the Hololink
# base class and the RESPONSE_* codes the policy reports). ../core is needed
# for the header-only module_core::LegacyHololinkAccess (hololink_default.hpp)
# that PcieHololink derives from -- header only, so the archive does not
# link hololink::module_core and the policy test stays free of the
# dependencies module_core drags in.
target_include_directories(hololink_hsb_pcie_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../core>
        $<BUILD_INTERFACE:${hololink_SOURCE_DIR}/src>)

# CUDA::cuda_driver: the receiver dma-buf exports the frame buffer and reads
# slot metadata out of VRAM through the driver API.
target_link_libraries(hololink_hsb_pcie_core
    PUBLIC
        hololink::module_headers
        hololink::hsb_pcie::headers
        hololink::hsb_pcie::abi_headers
        hololink::module_runtime
        hololink::core
        CUDA::cuda_driver)

set_target_properties(hololink_hsb_pcie_core PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON)

# ---- the module .so ------------------------------------------------------------
# NO_COMPAT_SUFFIX: every hsb_pcie board is provisioned with this one UUID
# and no compat_id is stamped, so Adapter::enumerate resolves the bare
# hololink_<UUID>.so. The receiver-construct TUs are deliberately kept (no
# NO_RECEIVER_CONSTRUCTS): HsbIpPublisher declares construct_roce_receiver /
# construct_linux_receiver and dlopen would fail on the missing vtable
# entries without them.
add_hololink_module(
    NAME hsb_pcie
    UUID 58c1419b-1e7a-4120-95fd-4efc156187c1
    NO_COMPAT_SUFFIX
    SOURCES
        module_entry.cpp
        pcie_data_channel_v1.cpp
        pcie_hololink_v1.cpp
        pcie_receiver_v1.cpp
)

target_include_directories(hsb_pcie
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        $<TARGET_PROPERTY:hololink::module_core,INCLUDE_DIRECTORIES>)

target_link_libraries(hsb_pcie
    PRIVATE
        hololink::module_core
        hololink::hsb_pcie::core
        hololink::hsb_pcie::headers)
