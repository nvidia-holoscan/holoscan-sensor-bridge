# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# See README.md for detailed information.

add_subdirectory(common)
add_subdirectory(${HSB_EMULATOR_PLATFORM})

# INTERFACE library: consumers link to emulation_host and emulation_common (both STATIC).
# Order: host then common so the linker resolves host->common refs (e.g. validate_configuration).
# Use --whole-archive for emulation_host so all objects (e.g. stm32f7xx_it.c interrupt handlers)
# and strong referenced symbols are linked without explicit need for user application level; 
# they are only referenced from the vector table in the same static lib.
add_library(emulation INTERFACE)

target_link_libraries(emulation INTERFACE
    -Wl,--whole-archive
    emulation_host
    emulation_common
    -Wl,--no-whole-archive
    dlpack::dlpack
)

# embedded targets will have a lower level hal driver that needs to get propagated
if (TARGET embedded_hal_driver)
    target_link_libraries(emulation INTERFACE embedded_hal_driver)
endif()

add_library(hololink::emulation ALIAS emulation)

# INTERFACE has no artifact; DESTINATION is ignored for it. Export all three targets.
install(TARGETS emulation emulation_host emulation_common
    EXPORT HololinkTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT hololink-emulation)
