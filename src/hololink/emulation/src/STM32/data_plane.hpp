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
#include "time.h"

namespace hololink::emulation {

struct DataPlaneCtxt {
    // metadata protection
    uint8_t bootp_buffer[sizeof(BootpPacket) + ETHER_HDR_LEN + IP_HDR_LEN + UDP_HDR_LEN];
    ETH_BufferTypeDef tx_buffers;
    ETH_TxPacketConfigTypeDef tx_config;
    ETH_HandleTypeDef* eth_handle;
    struct timespec start_time;
    uint32_t sif_address;
    uint32_t packetizer_mode;
    bool running;
    bool new_frame;
};

}

#endif
