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

#include "coe_data_plane.hpp"
#include "coe_transmitter.hpp"

namespace hololink::emulation {

COEDataPlane::COEDataPlane(HSBEmulator& hsb_emulator, const IPAddress& source_ip, uint8_t data_plane_id, uint8_t sensor_id)
    : DataPlane(hsb_emulator, source_ip, data_plane_id, sensor_id)
{
    // allocate transmitter and its metadata
    // initialize the COE transmitter with the interface name of the ip_address configuration
    transmitter_ = new COETransmitter(ip_address_.if_name);
    COETransmissionMetadata* coe_metadata = (COETransmissionMetadata*)calloc(1, sizeof(COETransmissionMetadata));
    if (!coe_metadata) {
        throw std::runtime_error("Failed to allocate COE metadata");
    }
    coe_metadata->sensor_info = sensor_id;
    metadata_ = (TransmissionMetadata*)coe_metadata;
}

COEDataPlane::~COEDataPlane()
{
    delete transmitter_;
    free(metadata_);
}

void COEDataPlane::update_metadata()
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
        /* COE specific registers*/
        { vp_address_ + DP_QP, 0 },
    };
    registers_->read_many(address_value_pairs, sizeof(address_value_pairs) / sizeof(address_value_pairs[0]));

    // read many will lock the registers for reading while update_metadata will lock the TransmissionMetadata for writing
    // assign common TransmissionMetadata fields
    metadata_->dest_mac_low = address_value_pairs[0].value;
    metadata_->dest_mac_high = address_value_pairs[1].value;
    metadata_->dest_ip_address = address_value_pairs[2].value;
    metadata_->frame_size = address_value_pairs[3].value;
    if (address_value_pairs[4].value > UINT16_MAX / HSB_PAGE_SIZE) {
        throw std::runtime_error("payload_size * HSB_PAGE_SIZE must fit in uint16_t");
    }
    metadata_->payload_size = address_value_pairs[4].value * HSB_PAGE_SIZE;
    if (address_value_pairs[5].value > UINT16_MAX) {
        throw std::runtime_error("dest_port must be 16-bit unsigned integer");
    }
    metadata_->dest_port = (uint16_t)address_value_pairs[5].value;
    if (address_value_pairs[6].value > UINT16_MAX) {
        throw std::runtime_error("src_port must be 16-bit unsigned integer");
    }
    metadata_->src_port = (uint16_t)address_value_pairs[6].value;
    COETransmissionMetadata* coe_metadata = (COETransmissionMetadata*)metadata_;
    // NOTE: cannot check for overflow on casts because the register is packed this way
    coe_metadata->line_threshold_log2_enable_1722b = static_cast<uint8_t>(address_value_pairs[7].value >> 25);
    coe_metadata->enable_1722b = static_cast<bool>((address_value_pairs[7].value >> 24) & 0x1);
    coe_metadata->channel = static_cast<uint8_t>(address_value_pairs[7].value & 0x3F);
    coe_metadata->mac_dest[5] = static_cast<uint8_t>((metadata_->dest_mac_low >> 0) & 0xFF);
    coe_metadata->mac_dest[4] = static_cast<uint8_t>((metadata_->dest_mac_low >> 8) & 0xFF);
    coe_metadata->mac_dest[3] = static_cast<uint8_t>((metadata_->dest_mac_low >> 16) & 0xFF);
    coe_metadata->mac_dest[2] = static_cast<uint8_t>((metadata_->dest_mac_low >> 24) & 0xFF);
    coe_metadata->mac_dest[1] = static_cast<uint8_t>((metadata_->dest_mac_high >> 0) & 0xFF);
    coe_metadata->mac_dest[0] = static_cast<uint8_t>((metadata_->dest_mac_high >> 8) & 0xFF);
}

} // namespace hololink::emulation
