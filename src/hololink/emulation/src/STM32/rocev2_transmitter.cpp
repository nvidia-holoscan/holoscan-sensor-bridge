/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "dlpack/dlpack.h"
#include "zlib.h"

#include "STM32/data_plane.hpp"
#include "STM32/net.hpp"
#include "STM32/rocev2_transmitter.hpp"
#include "STM32/stm32_system.h"
#include "STM32/tim.h"
#include "base_transmitter.hpp"
#include "utils.hpp"

namespace hololink::emulation {

// this implementation requires that the FrameMetadata is packed to exactly 128 bytes
static_assert(sizeof(FrameMetadata) == 128, "FrameMetadata size is not 128 bytes");

void RoCEv2Transmitter::init_metadata(RoCEv2TransmissionMetadata* metadata, const IPAddress& source_ip)
{
    memset(metadata, 0, sizeof(RoCEv2TransmissionMetadata));
    metadata->packet_offset = PACKET_OFFSET_RESET;
    metadata->in_use = false;

    // initialize headers in payload buffer
    // set ethernet header
    struct ether_header* eth = (struct ether_header*)RoCEv2TransmissionMetadata_get_buffer(metadata);
    memcpy(eth->ether_shost, get_mac_address(), ETHER_ADDR_LEN);
    eth->ether_type = htons(ETHERTYPE_IP);

    // set IP header
    struct iphdr* ip_header = (struct iphdr*)NET_GET_ETHER_PAYLOAD(eth);
    ip_header->ihl_version = (IPVERSION << 4) | (IP_HDR_LEN / 4); // 5 32-bit words
    ip_header->frag_off = htons(IP_DF); // 0x4000 not used without fragmentation
    ip_header->ttl = IPDEFTTL; // not used without fragmentation
    ip_header->protocol = IPPROTO_UDP; // UDP
    ip_header->saddr = source_ip.ip_address;
    ip_header->daddr = htonl(metadata->base.vp_data[DP_HOST_IP / REGISTER_SIZE]);

    // set UDP header
    struct udphdr* udp_header = (struct udphdr*)NET_GET_IP_PAYLOAD(ip_header);
    udp_header->source = htons(DATA_SOURCE_UDP_PORT);

    // set BT header
    struct bthdr* bt_header = (struct bthdr*)NET_GET_UDP_PAYLOAD(udp_header);
    bt_header->p_key = 0xFFFF;

    // set RET header
    // struct rethdr* ret_header = (struct rethdr*)NET_GET_BT_PAYLOAD(bt_header);
}

// effectively purges the data buffers.
void update_packet_headers(RoCEv2TransmissionMetadata* metadata)
{
    // reset the packet offset to the beginning of the packet
    metadata->packet_offset = PACKET_OFFSET_RESET;
    // update packet buffer headers
    struct ether_header* ether_header = (struct ether_header*)RoCEv2TransmissionMetadata_get_buffer(metadata);
    ether_header->ether_dhost[0] = (metadata->base.vp_data[DP_HOST_MAC_HIGH / REGISTER_SIZE] >> 8) & 0xFF;
    ether_header->ether_dhost[1] = (metadata->base.vp_data[DP_HOST_MAC_HIGH / REGISTER_SIZE] >> 0) & 0xFF;
    ether_header->ether_dhost[2] = (metadata->base.vp_data[DP_HOST_MAC_LOW / REGISTER_SIZE] >> 24) & 0xFF;
    ether_header->ether_dhost[3] = (metadata->base.vp_data[DP_HOST_MAC_LOW / REGISTER_SIZE] >> 16) & 0xFF;
    ether_header->ether_dhost[4] = (metadata->base.vp_data[DP_HOST_MAC_LOW / REGISTER_SIZE] >> 8) & 0xFF;
    ether_header->ether_dhost[5] = (metadata->base.vp_data[DP_HOST_MAC_LOW / REGISTER_SIZE] >> 0) & 0xFF;

    struct iphdr* ip_header = (struct iphdr*)NET_GET_ETHER_PAYLOAD(ether_header);
    ip_header->daddr = htonl(metadata->base.vp_data[DP_HOST_IP / REGISTER_SIZE]);

    struct udphdr* udp_header = (struct udphdr*)NET_GET_IP_PAYLOAD(ip_header);
    udp_header->dest = htons(metadata->base.vp_data[DP_HOST_UDP_PORT / REGISTER_SIZE] & 0xFFFF);

    struct bthdr* bt_header = (struct bthdr*)NET_GET_UDP_PAYLOAD(udp_header);
    bt_header->opcode = IB_OPCODE_WRITE;
    bt_header->destqp = htonl((0xFF << 24) | (metadata->base.vp_data[DP_QP / REGISTER_SIZE] & 0xFFFFFF));

    struct rethdr* ret_header = (struct rethdr*)NET_GET_BT_PAYLOAD(bt_header);
    ret_header->va = htonll(metadata->address);
    ret_header->r_key = htonl(metadata->base.vp_data[DP_RKEY / REGISTER_SIZE]);
    ret_header->dmalen = htonl(metadata->payload_size);

    metadata->packet_offset += ROCEV2_HDR_LEN;
}

int64_t RoCEv2Transmitter::send(TransmissionMetadata* metadata, const DLTensor& tensor, FrameMetadata* frame_metadata)
{
    return send(metadata, (uint8_t*)tensor.data, DLTensor_n_bytes(tensor), frame_metadata);
}

static inline uint16_t calculate_payload_size(const ETH_BufferTypeDef* tx_buffer)
{
    uint16_t payload_size = tx_buffer->len;
    const ETH_BufferTypeDef* cur = tx_buffer;
    while (cur->next) {
        cur = cur->next;
        payload_size += cur->len;
    }
    return payload_size;
}

static inline uint32_t calculate_crc(const ETH_BufferTypeDef* tx_buffer)
{
    static const uint8_t crc_init_buf[8] = { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };
    uint32_t crc = crc32(0L, crc_init_buf, sizeof(crc_init_buf));
    const ETH_BufferTypeDef* cur = tx_buffer;
    // need to skip the ethernet header in the CRC calculation
    crc = crc32(crc, cur->buffer + ETHER_HDR_LEN, cur->len - ETHER_HDR_LEN);
    while (cur->next) {
        cur = cur->next;
        crc = crc32(crc, cur->buffer, cur->len);
    }
    return crc;
}

// tx_buffer_base is the first buffer in the chain that does not contain the CRC
static inline uint32_t condition_packet_crc(RoCEv2TransmissionMetadata* metadata, const ETH_BufferTypeDef* tx_buffer_base)
{
    // set up headers
    struct iphdr* ip_header = (struct iphdr*)(RoCEv2TransmissionMetadata_get_buffer_base(metadata) + IP_HDR_OFFSET);
    struct udphdr* udp_header = (struct udphdr*)NET_GET_IP_PAYLOAD(ip_header);
    struct bthdr* bt_header = (struct bthdr*)NET_GET_UDP_PAYLOAD(udp_header);
    struct rethdr* ret_header = (struct rethdr*)NET_GET_BT_PAYLOAD(bt_header);
    // initialize payload to the size of the buffer chain and add the CRC
    // ip header length is the full size of payload of ethernet frame including infiniband crc
    uint16_t payload_size = calculate_payload_size(tx_buffer_base) + sizeof(uint32_t) - ETHER_HDR_LEN;
    ip_header->tot_len = htons(payload_size);
    ip_header->tos = 0xFF;
    ip_header->ttl = 0xFF;
    ip_header->check = 0xFFFF;
    // udp header length is the full size of the ip payload
    payload_size -= IP_HDR_LEN;
    udp_header->len = htons(payload_size);
    udp_header->check = 0xFFFF;
    bt_header->psn = htonl(metadata->psn);
    // RoCEv2 dmalen is the size of the udp payload  less the RoCEv2 headers and CRC
    payload_size -= UDP_HDR_LEN + BT_HDR_LEN + RET_HDR_LEN + sizeof(uint32_t);
    if (bt_header->opcode == IB_OPCODE_WRITE_IMMEDIATE) {
        payload_size -= sizeof(uint32_t);
    }
    ret_header->dmalen = htonl((uint32_t)payload_size);
    // va is updated in send_packet because it must happen after the buffer is sent since we must know that data was sent to update this value
    // an alternative is to keep separate track of what this address should be in the metadata, but that's redundant. Another implementation can do this
    // ret_header->va
    uint32_t crc = calculate_crc(tx_buffer_base);
    // reset headers to original values for transmission
    ip_header->tos = 0x0;
    ip_header->ttl = IPDEFTTL;
    ip_header->check = 0x0;
    udp_header->check = 0x0;
    return crc;
}

static inline int16_t send_packet(RoCEv2TransmissionMetadata* metadata, const uint8_t* buffer, uint16_t buffer_size)
{
    ETH_BufferTypeDef tx_buffer_base = {
        .buffer = RoCEv2TransmissionMetadata_get_buffer_base(metadata),
        .len = RoCEv2TransmissionMetadata_get_buffer_size(metadata),
        .next = nullptr,
    };
    int16_t to_consume = RoCEv2TransmissionMetadata_get_max_size(metadata) - tx_buffer_base.len;
    if (to_consume > buffer_size) {
        to_consume = buffer_size;
    }
    ETH_BufferTypeDef tx_payload = {
        // the buffer itself should not be changed, but HAL expects a pointer to the buffer
        .buffer = (uint8_t*)buffer,
        .len = (uint16_t)to_consume,
        .next = nullptr,
    };
    tx_buffer_base.next = &tx_payload;

    uint32_t crc = condition_packet_crc(metadata, &tx_buffer_base);

    ETH_BufferTypeDef tx_crc = {
        .buffer = (uint8_t*)&crc,
        .len = sizeof(uint32_t),
        .next = nullptr,
    };
    tx_payload.next = &tx_crc;

    ETH_TxPacketConfigTypeDef tx_config = {
        .Attributes = ETH_TX_PACKETS_FEATURES_CSUM | ETH_TX_PACKETS_FEATURES_CRCPAD,
        .Length = tx_buffer_base.len + tx_payload.len + tx_crc.len,
        .TxBuffer = &tx_buffer_base,
        .CRCPadCtrl = ETH_CRC_PAD_INSERT,
        .ChecksumCtrl = ETH_CHECKSUM_IPHDR_PAYLOAD_INSERT_PHDR_CALC,
    };

    HAL_StatusTypeDef tx_status = HAL_ETH_Transmit(metadata->eth_handle, &tx_config, HSB_DEFAULT_TIMEOUT_MSEC);
    if (tx_status != HAL_OK) {
        return -1;
    }

    RoCEv2TransmissionMetadata_buffer_clear(metadata); // this is only necessary if buffering is enabled (not default)
    struct rethdr* ret_header = (struct rethdr*)(RoCEv2TransmissionMetadata_get_buffer_base(metadata) + RET_HDR_OFFSET);
    // update the target address for the next packet
    ret_header->va = htonll(ntohll(ret_header->va) + to_consume);
    return to_consume;
}

static inline void update_frame_metadata_start(RoCEv2TransmissionMetadata* metadata)
{
    FrameMetadata* frame_metadata = &metadata->frame_metadata;
    struct timespec timestamp;
    clock_gettime(CLOCK_REALTIME, &timestamp);
    frame_metadata->timestamp_s_low = htonl((uint32_t)timestamp.tv_sec);
    frame_metadata->timestamp_ns = htonl((uint32_t)timestamp.tv_nsec);
}

static inline void update_frame_metadata_end(RoCEv2TransmissionMetadata* metadata, const uint8_t** buffer)
{
    FrameMetadata* frame_metadata = &metadata->frame_metadata;
    // total number of bytes written is the current location of va in the rethdr minus the initial va
    struct rethdr* ret_header = (struct rethdr*)(RoCEv2TransmissionMetadata_get_buffer_base(metadata) + RET_HDR_OFFSET);
    int64_t bytes_written = ntohll(ret_header->va) - metadata->address;
    // reset the target address for the next packet (the FrameMetadata) to be at the appropriate offset
    ret_header->va = htonll(metadata->address + metadata->metadata_offset);
    struct timespec timestamp;
    clock_gettime(CLOCK_REALTIME, &timestamp);
    *buffer = (const uint8_t*)frame_metadata;
    frame_metadata->metadata_s_low = htonl((uint32_t)timestamp.tv_sec);
    frame_metadata->metadata_ns = htonl((uint32_t)timestamp.tv_nsec);
    frame_metadata->bytes_written_high = htonl((uint32_t)(bytes_written >> 32));
    frame_metadata->bytes_written_low = htonl((uint32_t)(bytes_written & 0xFFFFFFFF));
    frame_metadata->frame_number = htonl(metadata->frame_number);
    frame_metadata->psn = htonl(metadata->psn);
}

static inline void write_immediate_data(RoCEv2TransmissionMetadata* metadata, uint32_t immediate_value)
{
    uint8_t* immediate_buffer = RoCEv2TransmissionMetadata_get_buffer(metadata);
    memcpy(immediate_buffer, &immediate_value, sizeof(immediate_value));
    metadata->packet_offset += sizeof(immediate_value);
}

static inline void set_opcode(RoCEv2TransmissionMetadata* metadata, uint8_t opcode)
{
    struct bthdr* bt_header = (struct bthdr*)(RoCEv2TransmissionMetadata_get_buffer_base(metadata) + ETHER_HDR_LEN + IP_HDR_LEN + UDP_HDR_LEN);
    bt_header->opcode = opcode;
}

static inline bool is_write_immediate(const RoCEv2TransmissionMetadata* metadata)
{
    struct bthdr* bt_header = (struct bthdr*)(RoCEv2TransmissionMetadata_get_buffer_base(metadata) + ETHER_HDR_LEN + IP_HDR_LEN + UDP_HDR_LEN);
    return bt_header->opcode == IB_OPCODE_WRITE_IMMEDIATE;
}

int64_t RoCEv2Transmitter::send(TransmissionMetadata* transmission_metadata, const uint8_t* buffer, size_t buffer_size, FrameMetadata* frame_metadata)
{
    RoCEv2TransmissionMetadata* metadata = (RoCEv2TransmissionMetadata*)transmission_metadata;
    int64_t n_bytes_sent = 0;

    // if no destination assigned, short circuit and return 0
    if (!metadata->base.vp_data[DP_HOST_IP / REGISTER_SIZE] || !metadata->base.vp_data[DP_HOST_UDP_PORT / REGISTER_SIZE]) {
        goto cleanup;
    }

    if (!metadata->in_use) {
        update_packet_headers(metadata);
        // if we have frame_metadata and it is the default internal frame_metadata, update it
        if (frame_metadata && (frame_metadata == DEFAULT_FRAME_METADATA)) {
            update_frame_metadata_start(metadata);
        }
        metadata->in_use = true;
    }

    while (buffer_size) {

        int16_t sent_bytes = send_packet(metadata, buffer, buffer_size);
        if (sent_bytes < 0) {
            n_bytes_sent = -1;
            goto cleanup;
        }

        buffer_size -= sent_bytes;
        buffer += sent_bytes;
        n_bytes_sent += sent_bytes;
        metadata->psn = (metadata->psn + 1) & IB_PSN_MASK;

        // if we have frame metadata, set up a send immediate packet
        if (!buffer_size && frame_metadata) {
            set_opcode(metadata, IB_OPCODE_WRITE_IMMEDIATE);

            // write the immediate value into the transmission metadata buffer
            uint32_t immediate_value = htonl(metadata->page | ((metadata->psn & IB_PSN_MASK) << IB_PSN_SHIFT));
            write_immediate_data(metadata, immediate_value);

            // if the frame_metadata is the default, update the buffer using standard FrameMetadata
            if (frame_metadata && (frame_metadata == DEFAULT_FRAME_METADATA)) {
                update_frame_metadata_end(metadata, &buffer);
            } else { // put the provided frame_metadata into the buffer for next loop
                buffer = (uint8_t*)frame_metadata;
            }
            buffer_size = sizeof(FrameMetadata);

            // set the buffer to the frame_metadata and null the frame_metadata pointer so we do not do this again
            frame_metadata = nullptr;
        }
    }

    metadata->frame_number++;
    // if we made it this far and sent the immediate packet, release the metadata by marking as not in use
    if (is_write_immediate(metadata)) {
        metadata->in_use = false;
    }
cleanup:
    // for success or failure whether opcode is write immediate or write, reset to write
    set_opcode(metadata, IB_OPCODE_WRITE);
    return n_bytes_sent;
}

} // namespace hololink::emulation
