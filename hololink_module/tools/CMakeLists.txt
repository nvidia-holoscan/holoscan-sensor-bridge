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
