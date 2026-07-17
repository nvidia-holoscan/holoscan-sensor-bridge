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

#ifndef COMMON_COE_TRANSMITTER_HPP
#define COMMON_COE_TRANSMITTER_HPP

#include <cstdint>
#include <cstring>

#include "dlpack/dlpack.h"

#include "../../hsb_config.hpp"
#include "base_transmitter.hpp"
#include "net.hpp"

namespace hololink::emulation {

struct COECtxt;

void update_packet_headers(COECtxt* coe_ctxt);
void update_frame_metadata_start(COECtxt* coe_ctxt);
void update_frame_metadata_end(COECtxt* coe_ctxt, uint16_t packet_bytes);
// this is update of packet headers in the middle of sending a frame
void update_packet_headers_midframe(COECtxt* coe_ctxt, uint32_t to_consume, bool frame_end);

/**
 * @brief The COETransmitter implements the BaseTransmitter interface and encapsulates the transport over COE
 */
class COETransmitter : public BaseTransmitter {
public:
    /**
     * @brief Send a host buffer over COE. The ctxt parameter is downcast to COECtxt*
     * (COECtxt embeds the platform DataPlaneCtxt extension as its first member).
     * Implementation of BaseTransmitter::send interface method.
     */
    int64_t send(DataPlaneCtxt* ctxt, const uint8_t* buffer, size_t buffer_size, FrameMetadata* frame_metadata = nullptr) override;

    void init_metadata(COECtxt* metadata, const IPAddress& source_ip, uint8_t sensor_id);
};

} // namespace hololink::emulation

#endif
