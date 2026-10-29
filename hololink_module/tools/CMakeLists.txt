# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Command-line tools that ship with the hololink_module framework.
# Unlike examples (which exercise a specific board / sensor /
# operator pipeline), tools here are board-agnostic and only depend
# on hololink::module.

add_executable(adapter_enumerator adapter_enumerator.cpp)
target_link_libraries(adapter_enumerator
    PRIVATE
        hololink::module
)

install(TARGETS adapter_enumerator
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT hololink-module-tools)

# Read-only PCIe endpoint identity tool: shows the board identity the
# hololink_pcie kernel driver reports (BDF, UUID, HSB version/date,
# responding or not) and offers raw BAR0 peek/poke for bring-up. The
# board UUID is read-only in the FPGA, so nothing here writes it.
# Part of the PCIe transport (HOLOLINK_BUILD_PCIE). The sysfs / ioctl reads
# it drives (PcieDeviceInfo, pcie_device_info.{hpp,cpp}) are private to this
# tool -- discovery on the application path is the driver's bootp announcement
# -- and compile straight into it; the test reuses the same source file.
if(HOLOLINK_BUILD_PCIE)
add_executable(hololink_pcie_identity
    hololink_pcie_identity.cpp
    pcie_device_info.cpp)
target_link_libraries(hololink_pcie_identity
    PRIVATE
        hololink::module
        hololink::hsb_pcie::abi_headers
)

install(TARGETS hololink_pcie_identity
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT hololink-module-tools)
endif() # HOLOLINK_BUILD_PCIE
