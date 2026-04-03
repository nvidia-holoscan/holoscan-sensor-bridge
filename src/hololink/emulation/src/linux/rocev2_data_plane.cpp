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

#include <stdexcept>

#include "rocev2_data_plane.hpp"
#include "rocev2_transmitter.hpp"

namespace hololink::emulation {

RoCEv2DataPlane::RoCEv2DataPlane(HSBEmulator& hsb_emulator, const IPAddress& source_ip, uint8_t data_plane_id, uint8_t sensor_id)
    : DataPlane(hsb_emulator, source_ip, data_plane_id, sensor_id)
{
    // allocate transmitter and its metadata
    transmitter_ = new RoCEv2Transmitter(source_ip);
    metadata_ = (TransmissionMetadata*)calloc(1, sizeof(LinuxTransmissionMetadata));
}

RoCEv2DataPlane::~RoCEv2DataPlane()
{
    delete transmitter_;
    free(metadata_);
}

void RoCEv2DataPlane::update_metadata()
{
    struct AddressValuePair address_value_pairs[] = {
        /* TransmissionMetadata */
        { vp_address_ + DP_HOST_MAC_LOW, 0 },
        { vp_address_ + DP_HOST_MAC_HIGH, 0 },
        { vp_address_ + DP_HOST_IP, 0 },
        { vp_address_ + DP_BUFFER_LENGTH, 0 },
        { hif_address_ + DP_PACKET_SIZE, 0 },
        { vp_address_ + DP_HOST_UDP_PORT, 0 },
        { hif_address_ + DP_PACKET_UDP_PORT, 0 },
        /* RoCE specific registers*/
        { vp_address_ + DP_QP, 0 }, // qp
        { vp_address_ + DP_RKEY, 0 }, // rkey
        { vp_address_ + DP_MAX_BUFF, 0 }, // max_buff
        { vp_address_ + DP_PAGE_LSB, 0 }, // page_lsb
        { vp_address_ + DP_PAGE_MSB, 0 }, // page_msb
        { vp_address_ + DP_PAGE_INC, 0 }, // page_inc
        /* address must be figured out later */
    };
    registers_->read_many(address_value_pairs, sizeof(address_value_pairs) / sizeof(address_value_pairs[0]));

    // assign common TransmissionMetadata fields
    metadata_->dest_mac_low = address_value_pairs[0].value;
    metadata_->dest_mac_high = address_value_pairs[1].value;
    metadata_->dest_ip_address = address_value_pairs[2].value;
    metadata_->frame_size = address_value_pairs[3].value;
    metadata_->payload_size = address_value_pairs[4].value * HSB_PAGE_SIZE;
    if (address_value_pairs[5].value > UINT16_MAX) {
        throw std::runtime_error("dest_port is too large for uint16_t");
    }
    metadata_->dest_port = (uint16_t)address_value_pairs[5].value;
    if (address_value_pairs[6].value > UINT16_MAX) {
        throw std::runtime_error("src_port is too large for uint16_t");
    }
    metadata_->src_port = (uint16_t)address_value_pairs[6].value;

    // assign RoCE specific fields
    LinuxTransmissionMetadata* linux_metadata = (LinuxTransmissionMetadata*)metadata_;

    linux_metadata->qp = address_value_pairs[7].value;
    linux_metadata->rkey = address_value_pairs[8].value;
    linux_metadata->metadata_offset = (metadata_->frame_size + HSB_PAGE_SIZE - 1) & ~(HSB_PAGE_SIZE - 1);

    uint32_t start_page = (address_value_pairs[9].value >> 16) & 0xFFF;
    uint32_t end_page = (address_value_pairs[9].value >> 0) & 0xFFF;
    if (page_ < start_page) {
        page_ = start_page;
    } else {
        page_++;
    }
    if (page_ > end_page) {
        page_ = start_page;
    }

    uint64_t address = address_value_pairs[10].value + ((uint64_t)address_value_pairs[11].value << 32);
    linux_metadata->page = page_;
    linux_metadata->address = address + ((uint64_t)address_value_pairs[12].value) * HSB_PAGE_SIZE * page_;
}

} // namespace hololink::emulation
