# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# TEMPLATE platform — empty-stub starting point for porting the HSB Emulator to a new
# target. Every platform-side function compiles but does nothing useful; the user
# fills in the bodies in src/Template/HSBTemplate.cpp (and adjusts the struct
# extensions in src/Template/HSBTemplate.hpp) to wire the emulator up to their
# board / network stack / I/O peripherals.
#
# Configure with:
#   cmake -S src/hololink/emulation -B build -DHSB_EMULATOR_TARGET=TEMPLATE
#   cmake --build build -j
#
# The build will succeed but the resulting libraries cannot run as-is.

message(STATUS "configuring for TEMPLATE (empty-stub target)")

set(HSB_EMULATOR_PLATFORM "Template")

# HAL defines — none required for TEMPLATE. The user may set MAX_DATA_PLANES,
# MAX_SENSORS, or other compile-time policy here when porting; defaults in
# hsb_config.hpp (MAX_DATA_PLANES=256, MAX_SENSORS=32) are used otherwise.
set(HSB_HAL_DEFINES "")

# AddressMap capacities for HSBEmulatorCtxt::cp_{write,read}_map. Defaults to the
# std::vector-backed dynamic specialization (capacity 0). Override to a fixed
# capacity for no-heap targets — see cmake/targets/STM32F767ZI.cmake for an example.
# add_compile_definitions(HSB_CP_WRITE_MAP_SIZE=20 HSB_CP_READ_MAP_SIZE=20)

# FrameMetadata transport policy:
#   0 = embed FrameMetadata in the final payload packet by padding with zeros
#       (the host RDMA receiver lands it at virtual_address + metadata_offset)
#   1 = send FrameMetadata as its own packet (no zero-padding the payload packet)
# Linux and STM32 default to 0. Override at configure time with
# -DSEPARATE_FRAMEMETADATA_PACKET=<0|1>. Note that the tests/test_emulator_serve_file_receiver_sequences will currently fail if set to 1 as it is specifically testing the -DSEPARATE_FRAMEMETADATA_PACKET=0 case
set(SEPARATE_FRAMEMETADATA_PACKET 0 CACHE STRING
    "0=embed FrameMetadata in last payload packet (zero-pad); 1=send FrameMetadata as a separate packet")
add_compile_definitions(SEPARATE_FRAMEMETADATA_PACKET=${SEPARATE_FRAMEMETADATA_PACKET})

# Pull in the shared dlpack dependency the same way the other targets do.
include("${HOLOLINK_REL_PATH}/cmake/hololink_deps/dlpack.cmake")

# RoCEv2 iCRC needs zlib's crc32 unless the platform provides a CRC peripheral.
# `cmake/deps/zlib_crc32.cmake` downloads crc32.c + crc32.h + zlib.h + zutil.h +
# zconf.h into the build tree and sets ZLIB_PATH. The transport library setup in
# src/CMakeLists.txt + src/Template/CMakeLists.txt then publishes ZLIB_PATH on
# hsb_platform so common/rocev2_transmitter.cpp can `#include <zlib.h>`.
#
# If your target has a hardware CRC unit (Cortex-M7 STM32 / many vendor MCUs),
# you can use it instead by setting CRC_OFFLOAD=ON and providing the matching
# HAL_CRC_* shims in HSBTemplate.cpp — see common/rocev2_transmitter.cpp for the
# expected calls inside the `#ifdef CRC_OFFLOAD` branch.
include("${CMAKE_SOURCE_DIR}/cmake/deps/zlib_crc32.cmake")
