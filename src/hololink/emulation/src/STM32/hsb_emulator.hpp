/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * See README.md for detailed information.
 */

#ifndef STM32_HSB_EMULATOR_HPP
#define STM32_HSB_EMULATOR_HPP

#include "../../hsb_emulator.hpp"
#include <cstdint>

#include "../common/apb_events.hpp"
#include "STM32/i2c.hpp"
#include "STM32/net.hpp"
#include "STM32/spi.hpp"
#include "address_map.hpp"

#define CONTROL_PACKET_SIZE TX_BUFFER_SIZE

// STM32 uses fixed-capacity AddressMaps (no heap). The HSB_CP_*_MAP_SIZE macros that
// drive HSBEmulatorCtxt's cp_*_map types are set as build-wide compile definitions in
// cmake/targets/STM32F767ZI.cmake so every translation unit sees the same layout.

namespace hololink::emulation {

void handle_control_packet(HSBEmulatorCtxt* ctxt, ETH_BufferTypeDef* buffer);

/**
 * @brief STM32-specific HSBEmulatorCtxt extension. `base` (the common HSBEmulatorCtxt) is
 * the first member; the rest is STM32-only state (HAL handles, SPI ctxt, BootP scheduling,
 * and the file-scope data-plane array). No mutex — STM32 is single-threaded.
 */
struct STM32HSBEmulatorCtxt {
    HSBEmulatorCtxt base;

    DataPlane* data_plane_list[MAX_DATA_PLANES];
    unsigned short data_plane_count;

    ETH_HandleTypeDef eth_handle;
    SpiControllerCtxt* spi_ctxt;

    // TODO: generalize to an event loop mechanism
    uint32_t up_time_msec;
    uint32_t next_bootp_time_msec;
};

}

#endif