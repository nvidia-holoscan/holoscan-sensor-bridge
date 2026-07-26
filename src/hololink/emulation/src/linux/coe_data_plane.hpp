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

#ifndef LINUX_COE_DATA_PLANE_HPP
#define LINUX_COE_DATA_PLANE_HPP

#include <cstdint>
#include <cstring>

#include "dlpack/dlpack.h"

#include "../../coe_data_plane.hpp"
#include "../../hsb_config.hpp"
#include "base_transmitter.hpp"
#include "data_plane.hpp"
#include "net.hpp"

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
 * First member is the platform's DataPlaneCtxt extension (LinuxDataPlaneCtxt), which
 * itself first-members the common DataPlaneCtxt. Standard-layout C++ guarantees that
 * `&coe_ctxt == &coe_ctxt->base == &coe_ctxt->base.base == DataPlaneCtxt*`, so a
 * COECtxt* can be passed everywhere a DataPlaneCtxt* is expected (e.g. into the
 * protected DataPlane constructor) and DataPlane::data_plane_ctxt_ correctly points
 * at the embedded chain. Holds COE state that can change every frame sent.
 *
 * Per-data-plane (hif) and per-sensor (vp) register slices are reachable via
 * base.base.dp_registers / base.base.dp_sensor_registers.
 */
struct COECtxt {
    LinuxDataPlaneCtxt base;
    // this shall be 64-bit aligned
    uint32_t frame_size;

    uint32_t line_threshold;
    // frame_metadata also has these values, but they are in network byte order. Use these for actual tracking
    uint32_t frame_number;
    uint32_t psn;
    uint32_t line_offset; // number of bytes sent on the COE line
    uint32_t address; // this is the address offset where the next packet will be written to
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
    int data_socket_fd;
    struct sockaddr_ll dest_addr;
};

} // namespace hololink::emulation

#endif
