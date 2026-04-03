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

#include "STM32/data_plane.hpp"
#include "STM32/hsb_emulator.hpp"
#include "STM32/net.hpp"
#include "STM32/rocev2_transmitter.hpp"
#include "rocev2_data_plane.hpp"

namespace hololink::emulation {

uint16_t TRANSMITTER_COUNT = 0;

struct RoCEv2TransmissionMetadata ROCEV2_TRANSMISSION_METADATA[MAX_DATA_PLANES];
RoCEv2Transmitter ROCEV2_TRANSMITTER;

int vp_data_readback_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    RoCEv2TransmissionMetadata* metadata = (RoCEv2TransmissionMetadata*)ctxt;

    int i = 0;
    while (i < max_count) {
        uint32_t address = AVP_GET_ADDRESS(addr_val);
        if (address >= metadata->vp_address && address <= metadata->vp_address + DP_HOST_UDP_PORT) {
            AVP_SET_VALUE(addr_val, metadata->base.vp_data[(address - metadata->vp_address) / REGISTER_SIZE]);
        } else {
            return i;
        }
        addr_val++;
        i++;
    }
    return i;
}
int vp_data_configure_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    RoCEv2TransmissionMetadata* metadata = (RoCEv2TransmissionMetadata*)ctxt;

    int i = 0;
    while (i < max_count) {
        uint32_t address = AVP_GET_ADDRESS(addr_val);
        if (address >= metadata->vp_address && address <= metadata->vp_address + DP_HOST_UDP_PORT) {
            metadata->base.vp_data[(address - metadata->vp_address) / REGISTER_SIZE] = AVP_GET_VALUE(addr_val);
        } else {
            return i;
        }
        addr_val++;
        i++;
    }
    return i;
}

int hif_data_readback_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    RoCEv2TransmissionMetadata* metadata = (RoCEv2TransmissionMetadata*)ctxt;

    int i = 0;
    while (i < max_count) {
        uint32_t address = AVP_GET_ADDRESS(addr_val);
        if (address >= metadata->hif_address && address <= metadata->hif_address + DP_VP_MASK) {
            AVP_SET_VALUE(addr_val, metadata->hif_data[(address - metadata->hif_address) / REGISTER_SIZE]);
        } else {
            return i;
        }
        addr_val++;
        i++;
    }
    return i;
}
int hif_data_configure_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    RoCEv2TransmissionMetadata* metadata = (RoCEv2TransmissionMetadata*)ctxt;

    int i = 0;
    while (i < max_count) {
        uint32_t address = AVP_GET_ADDRESS(addr_val);
        if (address >= metadata->hif_address && address <= metadata->hif_address + DP_VP_MASK) {
            metadata->hif_data[(address - metadata->hif_address) / REGISTER_SIZE] = AVP_GET_VALUE(addr_val);
        } else {
            return i;
        }
        addr_val++;
        i++;
    }
    return i;
}

RoCEv2DataPlane::RoCEv2DataPlane(HSBEmulator& hsb_emulator, const IPAddress& source_ip, uint8_t data_plane_id, uint8_t sensor_id)
    : DataPlane(hsb_emulator, source_ip, data_plane_id, sensor_id)
{
    // allocate transmitter and its metadata
    transmitter_ = (BaseTransmitter*)&ROCEV2_TRANSMITTER;
    RoCEv2TransmissionMetadata* rocev2_metadata;
    if (TRANSMITTER_COUNT < MAX_DATA_PLANES) {
        rocev2_metadata = (RoCEv2TransmissionMetadata*)&ROCEV2_TRANSMISSION_METADATA[TRANSMITTER_COUNT++];
    } else {
        Error_Handler();
    }
    ((RoCEv2Transmitter*)transmitter_)->init_metadata(rocev2_metadata, source_ip);
    metadata_ = (TransmissionMetadata*)rocev2_metadata;
    rocev2_metadata->eth_handle = data_plane_ctxt_->eth_handle;
    rocev2_metadata->vp_address = vp_address_;
    rocev2_metadata->hif_address = hif_address_;

    // register callbacks for data plane vp data
    CHECK_CP_MAP_SET(hsb_emulator.register_read_callback(vp_address_ + DP_QP, vp_address_ + DP_HOST_UDP_PORT + REGISTER_SIZE, vp_data_readback_cb, rocev2_metadata));
    CHECK_CP_MAP_SET(hsb_emulator.register_write_callback(vp_address_ + DP_QP, vp_address_ + DP_HOST_UDP_PORT + REGISTER_SIZE, vp_data_configure_cb, rocev2_metadata));

    // register callbacks for data plane hif data
    CHECK_CP_MAP_SET(hsb_emulator.register_read_callback(hif_address_, hif_address_ + DP_VP_MASK + REGISTER_SIZE, hif_data_readback_cb, rocev2_metadata));
    CHECK_CP_MAP_SET(hsb_emulator.register_write_callback(hif_address_, hif_address_ + DP_VP_MASK + REGISTER_SIZE, hif_data_configure_cb, rocev2_metadata));
}

RoCEv2DataPlane::~RoCEv2DataPlane()
{
    // do nothing
}

void RoCEv2DataPlane::update_metadata()
{

    RoCEv2TransmissionMetadata* rocev2_metadata = (RoCEv2TransmissionMetadata*)metadata_;

    // update page
    uint32_t start_page = (rocev2_metadata->base.vp_data[DP_MAX_BUFF / REGISTER_SIZE] >> 16) & 0xFFF;
    uint32_t end_page = (rocev2_metadata->base.vp_data[DP_MAX_BUFF / REGISTER_SIZE] >> 0) & 0xFFF;
    if (page_ < start_page) {
        page_ = start_page;
    } else {
        page_++;
    }
    if (page_ > end_page) {
        page_ = start_page;
    }

    // derived metadata
    rocev2_metadata->payload_size = rocev2_metadata->hif_data[DP_PACKET_SIZE / REGISTER_SIZE] * HSB_PAGE_SIZE;
    uint64_t address = rocev2_metadata->base.vp_data[DP_PAGE_LSB / REGISTER_SIZE] + ((uint64_t)rocev2_metadata->base.vp_data[DP_PAGE_MSB / REGISTER_SIZE] << 32);
    address += ((uint64_t)rocev2_metadata->base.vp_data[DP_PAGE_INC / REGISTER_SIZE]) * HSB_PAGE_SIZE * page_;
    rocev2_metadata->address = address;
    rocev2_metadata->page = page_;
    rocev2_metadata->metadata_offset = (rocev2_metadata->base.vp_data[DP_BUFFER_LENGTH / REGISTER_SIZE] + HSB_PAGE_SIZE - 1) & ~(HSB_PAGE_SIZE - 1);
}

} // namespace hololink::emulation
