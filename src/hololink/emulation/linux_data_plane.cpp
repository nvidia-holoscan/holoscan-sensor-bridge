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

#include "linux_data_plane.hpp"
#include "linux_transmitter.hpp"

namespace hololink::emulation {

LinuxDataPlane::LinuxDataPlane(HSBEmulator& hsb_emulator, const IPAddress& source_ip, uint8_t data_plane_id, uint8_t sensor_id)
    : DataPlane(hsb_emulator, source_ip, data_plane_id, sensor_id)
{
    // allocate transmitter and its metadata
    transmitter_ = new LinuxTransmitter(source_ip);
    metadata_ = (TransmissionMetadata*)calloc(1, sizeof(LinuxTransmissionMetadata));
}

LinuxDataPlane::~LinuxDataPlane()
{
    delete transmitter_;
    free(metadata_);
}

void LinuxDataPlane::update_metadata()
{
    std::vector<uint32_t> addresses = {
        // TransmissionMetadata
        vp_address_ + hololink::DP_HOST_MAC_LOW, // dest_mac_low
        vp_address_ + hololink::DP_HOST_MAC_HIGH, // dest_mac_high
        vp_address_ + hololink::DP_HOST_IP, // dest_ip_address
        vp_address_ + hololink::DP_BUFFER_LENGTH, // frame_size
        hif_address_ + hololink::DP_PACKET_SIZE, // payload_size
        vp_address_ + hololink::DP_HOST_UDP_PORT, // dest_port
        hif_address_ + hololink::DP_PACKET_UDP_PORT, // src_port

        /* RoCE specific registers*/
        vp_address_ + hololink::DP_QP, // qp
        vp_address_ + hololink::DP_RKEY, // rkey
        vp_address_ + hololink::DP_BUFFER_MASK, // page mask
        /* address must be figured out later */
    };

    // read many will lock the registers for reading while update_metadata will lock the TransmissionMetadata for writing
    std::vector<uint32_t> values = registers_->read_many(addresses);

    // assign common TransmissionMetadata fields
    metadata_->dest_mac_low = values[0];
    metadata_->dest_mac_high = values[1];
    metadata_->dest_ip_address = values[2];
    metadata_->frame_size = values[3];
    metadata_->payload_size = values[4] * hololink::core::PAGE_SIZE;
    if (values[5] > UINT16_MAX) {
        throw std::runtime_error("dest_port is too large for uint16_t");
    }
    metadata_->dest_port = (uint16_t)values[5];
    if (values[6] > UINT16_MAX) {
        throw std::runtime_error("src_port is too large for uint16_t");
    }
    metadata_->src_port = (uint16_t)values[6];

    // assign RoCE specific fields
    LinuxTransmissionMetadata* linux_metadata = (LinuxTransmissionMetadata*)metadata_;

    linux_metadata->qp = values[7];
    linux_metadata->rkey = values[8];
    linux_metadata->page_mask = values[9];
    linux_metadata->metadata_offset = (metadata_->frame_size + hololink::core::PAGE_SIZE - 1) & ~(hololink::core::PAGE_SIZE - 1);

    uint32_t next_page = next_page_;
    uint32_t page = next_page;
    if (linux_metadata->page_mask) {
        for (uint32_t i = 0; i < 0x20; i++) {
            page = next_page;
            next_page = (next_page + 1) & 0x1F; // % 32
            if (linux_metadata->page_mask & (1 << page)) {
                break;
            }
        }
    }
    // NOTE that if page_mask is 0, the page will be reused
    next_page_ = next_page; // save for next frame
    linux_metadata->page = page;
    linux_metadata->address = ((uint64_t)registers_->read(vp_address_ + ADDRESS_MAP[page])) << 7;
}

} // namespace hololink::emulation
