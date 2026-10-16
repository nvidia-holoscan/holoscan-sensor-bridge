# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Module-native PCIe receive path: hololink::module::pcie_receiver.
#
# Core-free port of the reference bring-up receiver (geometry, device
# access, receiver) for the hololink_module framework, plus the PCIe
# endpoint helpers (pcie_enumerator.cpp: the /sys/class/hololink_pcie walk
# and the identity ioctls the hololink_pcie_identity tool and the PCIe
# example use; the header installs as <hololink/module/pcie_enumerator.hpp>
# as before). Everything PCIe-specific on the host side is in this target,
# so the hololink::module host library itself carries no PCIe code. The
# receiver owns
# memory (dma-buf import / host staging) and the MSI-X completion wait
# through its own fd (<hololink/module/pcie/hololink_pcie_uapi.h>) and
# touches no register: it implements
# hololink::module::hsb_pcie::PcieReceiverAttachmentV1 (from the
# header-only hololink::hsb_pcie::headers), and the hsb_pcie module's
# PcieDataChannelInterfaceV1 programs the endpoint for it. Deliberately
# links no hololink::core, no legacy core and no control-plane relay --
# PcieReceiverOp (host/operators) links this so the operators archive
# stays free of the legacy core.
include(GNUInstallDirs)
find_package(CUDAToolkit REQUIRED)

# Header-only target for the PCIe headers under include/hololink/module/pcie/:
# the kernel ABI mirrors (hololink_pcie_uapi.h, hololink_pcie_bootp.h; the
# kernel module's CMake checks them against src/ehsb/kmod/include), the BAR0
# register map (pcie_regs.hpp) and the sysfs helpers (pcie_sysfs.hpp). The
# hsb_pcie module and its header target link this, so it is defined before
# module/hsb_pcie is processed (see hololink_module/CMakeLists.txt). The
# hololink::module host library carries none of these.
add_library(hololink_module_pcie_headers INTERFACE)
add_library(hololink::module::pcie_headers ALIAS hololink_module_pcie_headers)
target_include_directories(hololink_module_pcie_headers
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(hololink_module_pcie_headers
    INTERFACE
        hololink::module_headers)
target_compile_features(hololink_module_pcie_headers INTERFACE cxx_std_17)
set_target_properties(hololink_module_pcie_headers PROPERTIES
    EXPORT_NAME module::pcie_headers)
install(TARGETS hololink_module_pcie_headers
    EXPORT HololinkTargets
    COMPONENT hololink-module)

add_library(hololink_module_pcie_receiver STATIC
    pcie_device_access.cpp
    pcie_enumerator.cpp
    pcie_receiver.cpp)
add_library(hololink::module::pcie_receiver ALIAS hololink_module_pcie_receiver)

target_compile_features(hololink_module_pcie_receiver PUBLIC cxx_std_17)
target_include_directories(hololink_module_pcie_receiver
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# hololink::module_runtime supplies hsb_logger_cache, which the HSB_LOG_*
# macros from <hololink/module/logging.hpp> resolve against; carrying it
# here keeps the archive independently linkable (the receiver test links
# it next to hololink::module, which brings the same static-propagated
# dependency).
target_link_libraries(hololink_module_pcie_receiver
    PUBLIC
        hololink::module_headers
        hololink::module::pcie_headers
        hololink::hsb_pcie::headers
        hololink::module_runtime
        CUDA::cuda_driver)

# Header-only use of hololink::core::Deserializer (the bounds-checked reader
# for the slot-metadata blocks); no hololink::core symbol is linked, so the
# receiver stays free of the legacy core library.
target_include_directories(hololink_module_pcie_receiver
    PRIVATE
        ${hololink_SOURCE_DIR}/src)
set_target_properties(hololink_module_pcie_receiver PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME module::pcie_receiver)

install(TARGETS hololink_module_pcie_receiver
    EXPORT HololinkTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT hololink-module)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/hololink
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT hololink-module)
