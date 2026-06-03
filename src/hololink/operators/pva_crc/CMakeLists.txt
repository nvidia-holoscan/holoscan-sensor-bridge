# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# PVA CRC C++ API: header and precompiled static library (libpva_2.9_crc.a).
# The library is downloaded during the Python extension build; see
# python/hololink/operators/pva_crc/cpp/CMakeLists.txt.

# Install the public header so other components can include it
install(FILES pva_crc.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/operators/pva_crc
        COMPONENT hololink-operators)
