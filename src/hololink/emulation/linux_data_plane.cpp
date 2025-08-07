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

LinuxDataPlane::LinuxDataPlane(HSBEmulator& hsb_emulator, const IPAddress& source_ip, uint16_t source_port, DataPlaneID data_plane_id, SensorID sensor_id)
    : DataPlane(hsb_emulator, source_ip, data_plane_id, sensor_id)
{
    // allocate transmitter and its metadata
    transmitter_ = new LinuxTransmitter(IPAddress_to_string(source_ip), source_port);
    metadata_ = (TransmissionMetadata*)calloc(1, sizeof(LinuxTransmissionMetadata));
}

LinuxDataPlane::~LinuxDataPlane()
{
    delete transmitter_;
    free(metadata_);
}

void LinuxDataPlane::update_metadata()
{
    // retrieve configuration data
    SensorConfiguration& sensor_configuration = sensor_map[sensor_id_];
    DataPlaneConfiguration& data_plane_configuration = data_plane_map[(DataPlaneID)configuration_.data_plane];
    uint32_t vp_address = sensor_configuration.vp_address;
    uint32_t hif_address = data_plane_configuration.hif_address;

    std::vector<uint32_t> addresses = {
        /* all linux metadata addresses */
        vp_address + hololink::DP_HOST_IP, // dest_ip_address
        vp_address + hololink::DP_HOST_UDP_PORT, // dest_port
        hif_address + hololink::DP_PACKET_SIZE, // payload_size
        /* address must be figured out later */
        vp_address + hololink::DP_QP, // qp
        vp_address + hololink::DP_RKEY, // rkey
        vp_address + hololink::DP_BUFFER_MASK, // page mask
        vp_address + hololink::DP_BUFFER_LENGTH, // frame_size
    };

    // read many will lock the registers for reading while update_metadata will lock the TransmissionMetadata for writing
    std::vector<uint32_t> values = registers_->read_many(addresses);

    // update metadata fields
    uint32_t payload_size = values[2] * hololink::core::PAGE_SIZE;
    if (payload_size > UINT16_MAX) {
        throw std::runtime_error("payload_size is too large for uint16_t");
    }
    metadata_->payload_size = (uint16_t)payload_size;
    LinuxTransmissionMetadata* linux_metadata = (LinuxTransmissionMetadata*)metadata_;
    linux_metadata->dest_ip_address = values[0];
    if (values[1] > UINT16_MAX) {
        throw std::runtime_error("dest_port is too large for uint16_t");
    }
    linux_metadata->dest_port = (uint16_t)values[1];
    linux_metadata->qp = values[3];
    linux_metadata->rkey = values[4];
    uint32_t frame_size = values[6];
    linux_metadata->metadata_offset = (frame_size + hololink::core::PAGE_SIZE - 1) & ~(hololink::core::PAGE_SIZE - 1);

    uint32_t page_mask = values[5];

    uint32_t next_page = next_page_;
    uint32_t page = next_page;
    if (page_mask) {
        for (uint32_t i = 0; i < 0x20; i++) {
            page = next_page;
            next_page = (next_page + 1) & 0x1F; // % 32
            if (page_mask & (1 << page)) {
                break;
            }
        }
    }
    // NOTE that if page_mask is 0, the page will be reused
    next_page_ = next_page; // save for next frame
    linux_metadata->page = page;
    linux_metadata->address = registers_->read(sensor_configuration.vp_address + address_map[page]) << 7;
}

} // namespace hololink::emulation
