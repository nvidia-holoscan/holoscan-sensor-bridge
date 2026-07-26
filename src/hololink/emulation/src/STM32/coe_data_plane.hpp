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

#ifndef STM32_COE_DATA_PLANE_HPP
#define STM32_COE_DATA_PLANE_HPP

#include <cstdint>
#include <cstring>

#include "dlpack/dlpack.h"

#include "../../coe_data_plane.hpp"
#include "../../hsb_config.hpp"
#include "base_transmitter.hpp"
#include "data_plane.hpp"
#include "net.hpp"

#ifndef MAX_TRANSMITTERS
#define MAX_TRANSMITTERS MAX_DATA_PLANES
#endif

// offset from the start of the packet buffer to the start of the data buffer to ensure 64-bit alignment of headers
#define COE_PACKET_OFFSET_RESET 2u

#define COE_HDR_LEN (ETHER_HDR_LEN + NTSCF_HDR_LEN + ACFUSER0C_HDR_LEN)
#define COECtxt_get_buffer_base(metadatap) (&((metadatap)->packet[COE_PACKET_OFFSET_RESET]))
#define COECtxt_get_buffer(metadatap) (&((metadatap)->packet[(metadatap)->packet_offset]))
#define COECtxt_get_buffer_size(metadatap) ((metadatap)->packet_offset - COE_PACKET_OFFSET_RESET)
#define COECtxt_buffer_clear(metadatap) ((metadatap)->packet_offset = COE_PACKET_OFFSET_RESET + COE_HDR_LEN)
// static packet size (MTU + COE_PACKET_OFFSET_RESET) - COE_PACKET_OFFSET_RESET - iCRC data (4 bytes)
#define COECtxt_get_max_size(metadatap) (sizeof(metadatap->packet) - COE_PACKET_OFFSET_RESET - sizeof(uint32_t))
#define COECtxt_mark_in_use(metadatap) ((metadatap)->in_use = true)
#define COECtxt_mark_available(metadatap) ((metadatap)->in_use = false)
#define COECtxt_is_in_use(metadatap) ((metadatap)->in_use == true)

namespace hololink::emulation {

// offset from the start of the packet buffer to the start of the data buffer to ensure 64-bit alignment of headers
#define COE_PACKET_OFFSET_RESET 2u

/**
 * Per-transmission context for COETransmitter (`COECtxt`).
 *
 * Holds COE state that can change every frame sent.
 */
struct COECtxt {
    // First member is the STM32 DataPlaneCtxt extension, which itself first-members the
    // common DataPlaneCtxt. &coe_ctxt == &coe_ctxt->base == &coe_ctxt->base.base ==
    // DataPlaneCtxt*; a COECtxt* can be handed to the protected DataPlane ctor wrapped
    // in a unique_ptr<DataPlaneCtxt, ...>. Per-data-plane (hif) and per-sensor (vp)
    // register slices are reachable via base.base.dp_registers /
    // base.base.dp_sensor_registers.
    STM32DataPlaneCtxt base;
    // this shall be 64-bit aligned
    uint32_t frame_size;
    uint32_t line_threshold;
    // frame_metadata also has these values, but they are in network byte order. Use these for actual tracking
    uint32_t frame_number;
    uint32_t psn;
    uint32_t line_offset; // number of bytes sent on the COE line
    uint32_t address; // this is the address offset where the next packet will be written to
    // Page-aligned offset (from start of frame buffer) at which FrameMetadata is
    // written. Computed in COEDataPlane::update_metadata() from frame_size, then
    // assigned to `address` in COETransmitter::send() once the payload is done so
    // the trailing metadata packet lands in the receiver-expected slot.
    uint32_t metadata_offset;
    uint16_t payload_size;
    uint16_t packet_offset;
    // frame_metadata, packet, and packet_offset are possibly not in all implementations
    // FrameMetadata requires 32-bit alignment
    FrameMetadata frame_metadata;
    // should already be 32-bit aligned
    alignas(uint32_t) uint8_t packet[TX_BUFFER_SIZE + COE_PACKET_OFFSET_RESET];
    uint8_t channel;
    // final due to alignment requirements
    bool in_use;
    bool enable_1722b;
};

} // namespace hololink::emulation

#endif
