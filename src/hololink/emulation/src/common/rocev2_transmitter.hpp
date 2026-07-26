/**
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef COMMON_ROCEV2_TRANSMITTER_HPP
#define COMMON_ROCEV2_TRANSMITTER_HPP

#include <cstdint>
#include <cstring>

#include "dlpack/dlpack.h"

#include "base_transmitter.hpp"
#include "hsb_config.hpp"
#include "net.hpp"

namespace hololink::emulation {

struct RoCEv2Ctxt;

void set_opcode(RoCEv2Ctxt* rocev2_ctxt, uint8_t opcode);
void update_frame_metadata_end(RoCEv2Ctxt* rocev2_ctxt, uint16_t packet_bytes);
uint32_t condition_packet_crc(RoCEv2Ctxt* metadata, const ETH_BufferTypeDef* tx_buffer_base);

/**
 * @brief The RoCEv2Transmitter implements the BaseTransmitter interface and encapsulates the transport over RoCEv2
 */
class RoCEv2Transmitter : public BaseTransmitter {
public:
    /**
     * @brief Send a host buffer over RoCEv2. The ctxt parameter is downcast to
     * RoCEv2Ctxt* (RoCEv2Ctxt embeds the platform DataPlaneCtxt extension as its first
     * member). Implementation of BaseTransmitter::send interface method.
     */
    int64_t send(DataPlaneCtxt* ctxt, const uint8_t* buffer, size_t buffer_size, FrameMetadata* frame_metadata = nullptr) override;

    void init_metadata(RoCEv2Ctxt* metadata, const IPAddress& source_ip);
};

} // namespace hololink::emulation

#endif
