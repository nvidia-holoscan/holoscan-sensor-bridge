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

#ifndef STM32_DATA_PLANE_HPP
#define STM32_DATA_PLANE_HPP

#include "../../data_plane.hpp"
#include "STM32/net.hpp"
#include "STM32/tim.h"

namespace hololink::emulation {

/**
 * @brief STM32-specific DataPlane context extension.
 *
 * `base` MUST be the first member: the DataPlane base class stores its `data_plane_ctxt_`
 * as a `DataPlaneCtxt*` pointing into this `base`. Standard-layout C++ guarantees
 * `&this->base == reinterpret_cast<STM32DataPlaneCtxt*>(this)`, so STM32 sources downcast
 * via `reinterpret_cast<STM32DataPlaneCtxt*>(data_plane_ctxt_.get())`.
 *
 * Allocation policy: STM32 builds use a file-scope static pool of these (see
 * `DATA_PLANE_CTXT[]` in STM32/data_plane.cpp) — no `new`/`malloc`. When the DataPlane
 * constructor is called with `ctxt == nullptr`, it claims `&DATA_PLANE_CTXT[data_plane_id].base`.
 */
struct STM32DataPlaneCtxt {
    DataPlaneCtxt base;
    // STM32 HAL Ethernet handle, used by HAL_ETH_Transmit() for BootP broadcasts.
    ETH_HandleTypeDef* eth_handle;
    // BootP TX buffer + HAL TX descriptor, pre-built in the DataPlane constructor
    // and reused on every broadcast.
    ETH_BufferTypeDef tx_buffers;
    ETH_TxPacketConfigTypeDef tx_config;
    alignas(uint32_t) uint8_t reserved[2]; // for DMA alignment of bootp_buffer below. should already be 32-bit aligned, but force it
    uint8_t bootp_buffer[sizeof(BootpPacket) + ETHER_HDR_LEN + IP_HDR_LEN + UDP_HDR_LEN];
};

// File-scope static pool of STM32 contexts; allocated once at program startup, indexed by
// data_plane_id when the DataPlane constructor is given `ctxt == nullptr`.
extern struct STM32DataPlaneCtxt DATA_PLANE_CTXT[MAX_DATA_PLANES];

}

#endif
