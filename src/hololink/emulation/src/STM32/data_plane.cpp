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

#include "STM32/data_plane.hpp"
#include "STM32/hsb_emulator.hpp"
#include "STM32/net.hpp"
#include "STM32/stm32_system.h"
#include "STM32/tim.h"
#include "utils.hpp"
#include <string.h>

namespace hololink::emulation {

#define BOOTP_PACKET_SIZE 342u
#define PACKETIZER_MODE 0x0Cu
#define PACKETIZER_RAM 0x04u
#define PACKETIZER_DATA 0x08u

struct DataPlaneCtxt DATA_PLANE_CTXT[MAX_DATA_PLANES];

// this is a subset of the HSBConfiguration struct that is part of the BootpPacket
// but it is kept separate because it must be packed.
struct VendorInfo {
    uint8_t tag;
    uint8_t tag_length;
    uint8_t vendor_id[VENDOR_ID_SIZE];
    uint8_t data_plane;
    uint8_t enum_version; // must be v2
    uint8_t board_id_lo;
    uint8_t board_id_hi;
    uint8_t uuid[BOARD_VERSION_SIZE];
    uint8_t serial_num[BOARD_SERIAL_NUM_SIZE];
    // have to break up the next 2 fields because they are not WORD-aligned
    uint8_t hsb_ip_version_lo;
    uint8_t hsb_ip_version_hi;
    uint8_t fpga_crc_lo;
    uint8_t fpga_crc_hi;
};

static_assert(sizeof(VendorInfo) == 41, "VendorInfo must be 41 bytes to ensure it is packed");

void init_bootp_packet(uint8_t* bootp_buffer, IPAddress& ip_address, HSBConfiguration& configuration)
{
    struct ether_header* ether_header = (struct ether_header*)bootp_buffer;
    {
        uint8_t broadcast_addr[] = ETHER_BROADCAST_ADDR;
        memcpy(ether_header->ether_dhost, broadcast_addr, ETHER_ADDR_LEN);
    }
    memcpy(ether_header->ether_shost, ip_address.mac, ETHER_ADDR_LEN);
    ether_header->ether_type = htons(ETHERTYPE_IP);

    struct iphdr* ip_header = (struct iphdr*)(bootp_buffer + ETHER_HDR_LEN);
    ip_header->ihl_version = (IPVERSION << 4) | (IP_HDR_LEN / 4);
    ip_header->tot_len = htons(sizeof(BootpPacket) + IP_HDR_LEN + UDP_HDR_LEN);
    ip_header->id = 0, // not used without fragmentation
        ip_header->frag_off = htons(IP_DF), // 0x4000 not used without fragmentation
        ip_header->ttl = IPDEFTTL, // not used without fragmentation
        ip_header->protocol = IPPROTO_UDP, // UDP
        ip_header->check = 0;
    ip_header->saddr = ip_address.ip_address;
    ip_header->daddr = ip_address.broadcast_address;

    struct udphdr* udp_header = (struct udphdr*)(bootp_buffer + ETHER_HDR_LEN + IP_HDR_LEN);
    udp_header->source = htons(BOOTP_REPLY_PORT);
    udp_header->dest = htons(BOOTP_REQUEST_PORT);
    udp_header->len = htons(sizeof(BootpPacket) + UDP_HDR_LEN);
    udp_header->check = 0;

    struct BootpPacket* bootp_packet = (struct BootpPacket*)(bootp_buffer + ETHER_HDR_LEN + IP_HDR_LEN + UDP_HDR_LEN);

    bootp_packet->op = 1;
    bootp_packet->htype = 1;
    bootp_packet->hlen = 6;
    bootp_packet->xid = htonl(configuration.data_plane);
    bootp_packet->ciaddr = ip_address.ip_address;
    memcpy(bootp_packet->chaddr, ip_address.mac, ETHER_ADDR_LEN);

    VendorInfo vendor_info = {
        .tag = configuration.tag,
        .tag_length = configuration.tag_length,
        .data_plane = configuration.data_plane,
        .enum_version = configuration.enum_version,
        .board_id_lo = configuration.board_id_lo,
        .board_id_hi = configuration.board_id_hi,
        .hsb_ip_version_lo = (uint8_t)(configuration.hsb_ip_version & 0xFF),
        .hsb_ip_version_hi = (uint8_t)(configuration.hsb_ip_version >> 8),
        .fpga_crc_lo = (uint8_t)(configuration.fpga_crc & 0xFF),
        .fpga_crc_hi = (uint8_t)(configuration.fpga_crc >> 8),
    };
    memcpy(vendor_info.vendor_id, configuration.vendor_id, sizeof(configuration.vendor_id));
    memcpy(vendor_info.uuid, configuration.uuid, sizeof(configuration.uuid));
    memcpy(vendor_info.serial_num, configuration.serial_num, sizeof(configuration.serial_num));

    memcpy(bootp_packet->vend, &vendor_info, sizeof(vendor_info));
}

void init_bootp_tx_config(DataPlaneCtxt* data_plane_ctxt)
{
    // initialize the tx buffer attached to the config
    data_plane_ctxt->tx_buffers = {
        .buffer = &(data_plane_ctxt->bootp_buffer[0]),
        .len = sizeof(data_plane_ctxt->bootp_buffer),
        .next = NULL,
    };

    data_plane_ctxt->tx_config = {
        .Attributes = ETH_TX_PACKETS_FEATURES_CSUM | ETH_TX_PACKETS_FEATURES_CRCPAD,
        .Length = data_plane_ctxt->tx_buffers.len,
        .TxBuffer = &data_plane_ctxt->tx_buffers,
        .CRCPadCtrl = ETH_CRC_PAD_INSERT,
        .ChecksumCtrl = ETH_CHECKSUM_IPHDR_PAYLOAD_INSERT_PHDR_CALC,
    };
}

static inline void update_bootp_packet(uint8_t* bootp_buffer, const struct timespec& start_time)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct BootpPacket* bootp_packet = (struct BootpPacket*)(bootp_buffer + ETHER_HDR_LEN + IP_HDR_LEN + UDP_HDR_LEN);
    bootp_packet->secs = (uint16_t)(get_delta_msecs(&start_time, &ts) / 1000);
}

void DataPlane_deleter(__attribute__((unused)) struct DataPlaneCtxt* ctxt)
{
    // do nothing
}

int packetizer_readback_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    DataPlaneCtxt* data_plane_ctxt = (DataPlaneCtxt*)ctxt;
    int n = 0;
    while (n < max_count) {
        uint32_t address = AVP_GET_ADDRESS(addr_val);
        if (address == data_plane_ctxt->sif_address + PACKETIZER_MODE) {
            AVP_SET_VALUE(addr_val, data_plane_ctxt->packetizer_mode);
        } else {
            return n;
        }
        addr_val++;
        n++;
    }
    return n;
}
int packetizer_configure_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    DataPlaneCtxt* data_plane_ctxt = (DataPlaneCtxt*)ctxt;
    int n = 0;
    while (n < max_count) {
        uint32_t address = AVP_GET_ADDRESS(addr_val);
        if (address == data_plane_ctxt->sif_address + PACKETIZER_MODE) {
            data_plane_ctxt->packetizer_mode = AVP_GET_VALUE(addr_val);
        } else {
            return n;
        }
        addr_val++;
        n++;
    }
    return n;
}

// HSBEmulator must remain alive for as long as the longest-lived DataPlane object it was used to construct
// IPAddress is both the source IP address and subnet mask to be able to set the appropriate broadcast address (!cannot use INADDR_BROADCAST!)
DataPlane::DataPlane(HSBEmulator& hsb_emulator, const IPAddress& ip_address, uint8_t data_plane_id, uint8_t sensor_id)
    : registers_(hsb_emulator.get_memory())
    , ip_address_(ip_address)
    , configuration_(hsb_emulator.get_config())
    , sensor_id_(sensor_id)
    , data_plane_id_(data_plane_id)
    , hif_address_(0x02000300 + 0x10000 * data_plane_id)
    , sif_address_(0x01000000 + 0x10000 * sensor_id)
{

    // DataPlaneConfiguration
    configuration_.data_plane = data_plane_id;
    // SensorConfiguration. See DataChannel::use_sensor() for more details
    uint8_t sif0_index = sensor_id_ * configuration_.sifs_per_sensor;
    vp_mask_ = 1 << sif0_index;
    switch (sif0_index) {
    case 0:
        frame_end_event_ = EVENT_SIF_0_FRAME_END;
        break;
    case 1:
        frame_end_event_ = EVENT_SIF_1_FRAME_END;
        break;
    default:
        // TODO: what's a reasonable default value since enumerator.cpp does not assign anything to the metadata here
        break;
    }
    vp_address_ = 0x1000 + 0x40 * sif0_index;

    if (data_plane_id_ >= MAX_DATA_PLANES) {
        Error_Handler();
    }
    net_set_ip_address(ip_address_.ip_address);
    data_plane_ctxt_.reset(&DATA_PLANE_CTXT[data_plane_id_]);
    data_plane_ctxt_.get_deleter() = DataPlane_deleter;
    data_plane_ctxt_->eth_handle = &(hsb_emulator.ctxt_->eth_handle);
    data_plane_ctxt_->sif_address = sif_address_;
    data_plane_ctxt_->new_frame = true;

    CHECK_CP_MAP_SET(hsb_emulator.register_read_callback(sif_address_ + PACKETIZER_MODE, sif_address_ + PACKETIZER_MODE + REGISTER_SIZE, packetizer_readback_cb, data_plane_ctxt_.get()));
    CHECK_CP_MAP_SET(hsb_emulator.register_write_callback(sif_address_ + PACKETIZER_MODE, sif_address_ + PACKETIZER_MODE + REGISTER_SIZE, packetizer_configure_cb, data_plane_ctxt_.get()));

    // register the DataPlane with the HSBEmulator so that it can be started/stopped
    hsb_emulator.add_data_plane(*this);
}

void DataPlane::start()
{
    if (init() != 0 || !data_plane_ctxt_) {
        Error_Handler();
    }
    if (data_plane_ctxt_->running) {
        return;
    }

    data_plane_ctxt_->running = true;
    clock_gettime(CLOCK_REALTIME, &data_plane_ctxt_->start_time);
    init_bootp_packet(data_plane_ctxt_->bootp_buffer, ip_address_, configuration_);
    init_bootp_tx_config(data_plane_ctxt_.get());
}

void DataPlane::stop()
{
    if (!is_running()) {
        return;
    }
    data_plane_ctxt_->running = false;
}

bool DataPlane::is_running()
{
    if (!data_plane_ctxt_) {
        return false;
    }
    return data_plane_ctxt_->running;
}

void DataPlane::update_metadata()
{
}

int64_t DataPlane::send(const DLTensor& tensor, FrameMetadata* frame_metadata)
{
    if (!transmitter_) {
        fprintf(stderr, "DataPlane::send() no transmitter\n");
        return -1;
    }

    if (frame_metadata == nullptr) {
        frame_metadata = DEFAULT_FRAME_METADATA;
    }
    return send((uint8_t*)tensor.data, DLTensor_n_bytes(tensor), frame_metadata);
}

int64_t DataPlane::send(const uint8_t* buffer, size_t buffer_size, FrameMetadata* frame_metadata)
{
    if (!transmitter_) {
        fprintf(stderr, "DataPlane::send_packet() no transmitter\n");
        return -1;
    }

    if (data_plane_ctxt_->new_frame) {
        update_metadata();
        data_plane_ctxt_->new_frame = false;
    }

    int64_t n_bytes = transmitter_->send(metadata_, buffer, buffer_size, frame_metadata);

    if (n_bytes > 0 && frame_metadata) {
        data_plane_ctxt_->new_frame = true;
    }

    return n_bytes;
}

int send_bootp(struct DataPlaneCtxt* data_plane_ctxt)
{
    update_bootp_packet(data_plane_ctxt->bootp_buffer, data_plane_ctxt->start_time);

    HAL_StatusTypeDef last_bootp_tx_status = HAL_ETH_Transmit(data_plane_ctxt->eth_handle, &data_plane_ctxt->tx_config, HSB_DEFAULT_TIMEOUT_MSEC);
    return (int)last_bootp_tx_status;
}

int DataPlane::broadcast_bootp()
{
    if (!is_running()) {
        return -1;
    }
    int last_bootp_tx_status = send_bootp(data_plane_ctxt_.get());
    return last_bootp_tx_status;
}

bool DataPlane::packetizer_enabled() const
{
    return (data_plane_ctxt_->packetizer_mode >> 28u) & 0x1u;
}

}