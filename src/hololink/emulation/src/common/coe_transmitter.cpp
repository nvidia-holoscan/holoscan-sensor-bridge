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

#include "base_transmitter.hpp"
#include "coe_data_plane.hpp"
#include "coe_transmitter.hpp"
#include "data_plane.hpp"
#include "net.hpp"
#include "utils.hpp"

#define SUBTYPE_AVTP_NTSCF 0x82u
#define COE_ACF_MSG_TYPE 0x06u /* ACF_SERIAL */
#define ACF_MSG_TYPE_SHIFT 9u
#define STREAM_ID                                      \
    {                                                  \
        0x00, 0x00, 0x00, 0x00, 0xAA, 0xAA, 0xAA, 0xAA \
    }
#define PACKET_SEQUENCE_NUMBER_MASK 0x7F
#define COE_VERSION_NTSCF_LEN_HIGH 0x80 /* SV = 0b1, Version = 0b000, R = 0b0, length_high_3bits = 0b000 */
#define ACFUSER0C_ADDRESS_MASK 0xFFFFFFF
#define COE_FLAG_FRAME_START 0b00000001u
#define COE_FLAG_FRAME_END 0b00000010u
#define COE_FLAG_LINE_END 0b00010000u

namespace hololink::emulation {

// Platform-specific wire emit. Builds the platform's transport (Linux: sendmsg over
// AF_PACKET socket; STM32: HAL_ETH_Transmit) from the scatter/gather list assembled by
// the common send_packet() below. Defined in linux/coe_transmitter.cpp and
// STM32/coe_transmitter.cpp.
int16_t send_coe_packet(COECtxt* coe_ctxt, ETH_BufferTypeDef* tx_buffers);

int32_t send_packet(COECtxt* coe_ctxt, const uint8_t* buffer, size_t buffer_size, FrameMetadata** frame_metadata)
{
    static uint8_t zeros[HSB_PAGE_SIZE] = { 0 };
    uint32_t zeros_len = sizeof(zeros);

    constexpr int kMaxTxBuffers = 4;
    ETH_BufferTypeDef tx_buffers[kMaxTxBuffers];
    int buffer_index = 0;

    // [0] header chunk (Ethernet/AVTP/ACFUser0C built into coe_ctxt->packet[])
    // STM32 and Linux ETH_BufferTypeDef have the buffer/len/next members in different
    // orders; omit .next from the initializer (zero-init) and chain it explicitly.
    tx_buffers[buffer_index] = {};
    tx_buffers[buffer_index].buffer = COECtxt_get_buffer_base(coe_ctxt);
    tx_buffers[buffer_index].len = COECtxt_get_buffer_size(coe_ctxt);

    uint16_t to_consume = coe_ctxt->payload_size;
    if (to_consume > buffer_size) {
        to_consume = static_cast<uint16_t>(buffer_size);
    }

    // [1] payload chunk
    ++buffer_index;
    tx_buffers[buffer_index] = {};
    tx_buffers[buffer_index].buffer = (uint8_t*)buffer;
    tx_buffers[buffer_index].len = to_consume;
    tx_buffers[buffer_index - 1].next = &tx_buffers[buffer_index];

    // If FrameMetadata is provided and it fits, append a zeros pad (so FrameMetadata
    // lands at coe_ctxt->metadata_offset) followed by the metadata. SEPARATE_FRAMEMETADATA_PACKET
    // controls whether the metadata can share a packet with payload bytes:
    //   =0 — always try to merge, even when to_consume > 0.
    //   =1 — only emit in a metadata-only packet (to_consume == 0). The upper-level
    //        send() loop drives entry into that state by exhausting buffer_size and
    //        bumping coe_ctxt->address to metadata_offset before re-entering. Without
    //        this gate-by-to_consume the SEPARATE=1 path never emitted the metadata at
    //        all and send() looped forever — receiver never saw FRAME_END.
    bool insert_frame_metadata = false;
    const bool embed_eligible = (*frame_metadata != nullptr)
#if SEPARATE_FRAMEMETADATA_PACKET
        && (to_consume == 0)
#endif
        ;
    if (embed_eligible) {
        int32_t available_space = static_cast<int32_t>(coe_ctxt->payload_size);
        available_space -= to_consume;
        available_space -= sizeof(FrameMetadata);
        available_space -= sizeof(uint32_t); // immediate header; matches RoCEv2 budgeting
        zeros_len = coe_ctxt->metadata_offset - (coe_ctxt->address + to_consume);
        available_space -= zeros_len;
        if (available_space >= (int16_t)sizeof(zeros) && sizeof(zeros) >= zeros_len) {
            insert_frame_metadata = true;
            if (*frame_metadata == DEFAULT_FRAME_METADATA) {
                update_frame_metadata_end(coe_ctxt, to_consume);
                *frame_metadata = &coe_ctxt->frame_metadata;
            }
            // [2] zeros pad to metadata_offset
            ++buffer_index;
            tx_buffers[buffer_index] = {};
            tx_buffers[buffer_index].buffer = &zeros[0];
            tx_buffers[buffer_index].len = zeros_len;
            tx_buffers[buffer_index - 1].next = &tx_buffers[buffer_index];
            // [3] FrameMetadata
            ++buffer_index;
            tx_buffers[buffer_index] = {};
            tx_buffers[buffer_index].buffer = (uint8_t*)*frame_metadata;
            tx_buffers[buffer_index].len = sizeof(FrameMetadata);
            tx_buffers[buffer_index - 1].next = &tx_buffers[buffer_index];
        }
    }
    update_packet_headers_midframe(coe_ctxt, to_consume, insert_frame_metadata);

    // Wire emit — platform-specific.
    if (send_coe_packet(coe_ctxt, &tx_buffers[0]) < 0) {
        return -1;
    }

    // cleanup
    COECtxt_buffer_clear(coe_ctxt); // only relevant if buffering is enabled (not default)
    if (insert_frame_metadata) {
        coe_ctxt->address = 0;
        *frame_metadata = nullptr;
    } else {
        coe_ctxt->address += to_consume;
    }
    return to_consume;
}

void COETransmitter::init_metadata(COECtxt* metadata, const IPAddress& source_ip, uint8_t sensor_id)
{
    memset(metadata->packet, 0, sizeof(metadata->packet));
    metadata->packet_offset = COE_PACKET_OFFSET_RESET;
    metadata->line_offset = 0;
    metadata->frame_number = 0;
    metadata->psn = 0;
    metadata->address = 0;
    COECtxt_mark_available(metadata);

    // initialize headers in payload buffer
    // set ethernet header
    struct ether_header* eth = (struct ether_header*)COECtxt_get_buffer(metadata);
    memcpy(eth->ether_shost, source_ip.mac, ETHER_ADDR_LEN);
    eth->ether_type = htons(ETHERTYPE_AVTP);

    struct NTSCFHeader* ntscf_header = (struct NTSCFHeader*)NET_GET_ETHER_PAYLOAD(eth);
    ntscf_header->avtpdu_header.subtype = SUBTYPE_AVTP_NTSCF;
    ntscf_header->version_ntscf_len_high = COE_VERSION_NTSCF_LEN_HIGH;
    ntscf_header->ntscf_len_low = 0;
    ntscf_header->sequence_num = 0;
    uint8_t stream_id[8] = STREAM_ID;
    memcpy(ntscf_header->stream_id, stream_id, sizeof(stream_id));

    struct ACFUser0CHeader* acf_user0c_header = (struct ACFUser0CHeader*)NET_GET_NTSCF_PAYLOAD(ntscf_header);
    acf_user0c_header->acf_header.acf_metadata = htons(COE_ACF_MSG_TYPE << ACF_MSG_TYPE_SHIFT);
    acf_user0c_header->reserved = 0x20;
    acf_user0c_header->sensor_info = sensor_id;
}

// effectively purges the data buffers.
void update_packet_headers(COECtxt* coe_ctxt)
{
    struct DPSensorRegisters* dp_sensor_reg = coe_ctxt->base.base.dp_sensor_registers;
    // reset the packet offset to the beginning of the packet
    coe_ctxt->packet_offset = COE_PACKET_OFFSET_RESET;
    // update packet buffer headers
    struct ether_header* ether_header = (struct ether_header*)COECtxt_get_buffer(coe_ctxt);
    uint32_t host_mac = dp_sensor_reg->vp_data[DP_HOST_MAC_HIGH / REGISTER_SIZE];
    ether_header->ether_dhost[0] = (host_mac >> 8) & 0xFF;
    ether_header->ether_dhost[1] = (host_mac >> 0) & 0xFF;
    host_mac = dp_sensor_reg->vp_data[DP_HOST_MAC_LOW / REGISTER_SIZE];
    ether_header->ether_dhost[2] = (host_mac >> 24) & 0xFF;
    ether_header->ether_dhost[3] = (host_mac >> 16) & 0xFF;
    ether_header->ether_dhost[4] = (host_mac >> 8) & 0xFF;
    ether_header->ether_dhost[5] = (host_mac >> 0) & 0xFF;

    struct NTSCFHeader* ntscf_header = (struct NTSCFHeader*)NET_GET_ETHER_PAYLOAD(ether_header);
    struct ACFUser0CHeader* acf_user0c_header = (struct ACFUser0CHeader*)NET_GET_NTSCF_PAYLOAD(ntscf_header);
    acf_user0c_header->channel = coe_ctxt->channel;
    coe_ctxt->packet_offset += COE_HDR_LEN;
#ifdef __linux__
    // TODO: This should probably be added as a callback
    memcpy(coe_ctxt->dest_addr.sll_addr, ether_header->ether_dhost, 6);
#endif
}

void update_packet_headers_midframe(COECtxt* coe_ctxt, uint32_t to_consume, bool frame_end)
{
    // try to wrap this into a common function call
    struct NTSCFHeader* ntscf_header = (struct NTSCFHeader*)(COECtxt_get_buffer_base(coe_ctxt) + NTSCF_HDR_OFFSET);
    struct ACFUser0CHeader* acf_user0c_header = (struct ACFUser0CHeader*)NET_GET_NTSCF_PAYLOAD(ntscf_header);
    struct timespec timestamp;
    clock_gettime(CLOCK_REALTIME, &timestamp);
    const uint32_t packet_frame_number = (coe_ctxt->frame_number & 0x3) << 28; // pre-shift for | operation
    acf_user0c_header->address = htonl((packet_frame_number) | (coe_ctxt->address & 0xFFFFFFF));
    acf_user0c_header->timestamp_sec = htonl((uint32_t)timestamp.tv_sec);
    acf_user0c_header->timestamp_nsec = htonl((uint32_t)timestamp.tv_nsec);
    acf_user0c_header->psn = coe_ctxt->psn & 0x7F;
    coe_ctxt->line_offset += to_consume;

    uint8_t flags = 0;
    if (coe_ctxt->line_offset >= coe_ctxt->line_threshold) {
        flags |= COE_FLAG_LINE_END;
        coe_ctxt->line_offset = 0;
    }

    if (frame_end) {
        flags |= COE_FLAG_FRAME_END;
    } else {
        flags |= (!coe_ctxt->address ? COE_FLAG_FRAME_START : 0);
    }
    // set frame_flags before transmitting
    acf_user0c_header->frame_flags = flags;
}

void update_frame_metadata_start(COECtxt* coe_ctxt)
{
    FrameMetadata* frame_metadata = &coe_ctxt->frame_metadata;
    struct timespec timestamp;
    if (clock_gettime(CLOCK_REALTIME, &timestamp)) {
        Error_Handler("Failed to get timestamp in update_frame_metadata_start");
    }
    frame_metadata->timestamp_s_low = htonl((uint32_t)timestamp.tv_sec);
    frame_metadata->timestamp_ns = htonl((uint32_t)timestamp.tv_nsec);
}

void update_frame_metadata_end(COECtxt* coe_ctxt, uint16_t packet_bytes)
{
    FrameMetadata* frame_metadata = &coe_ctxt->frame_metadata;
    // total number of bytes written is the current location of virtual address + this packet's bytes
    int64_t bytes_written = coe_ctxt->address + packet_bytes;
    struct timespec timestamp;
    if (clock_gettime(CLOCK_REALTIME, &timestamp)) {
        Error_Handler("Failed to get timestamp in update_frame_metadata_end");
    }
    frame_metadata->metadata_s_low = htonl((uint32_t)timestamp.tv_sec);
    frame_metadata->metadata_ns = htonl((uint32_t)timestamp.tv_nsec);
    frame_metadata->bytes_written_high = htonl((uint32_t)(bytes_written >> 32));
    frame_metadata->bytes_written_low = htonl((uint32_t)(bytes_written & 0xFFFFFFFF));
    frame_metadata->frame_number = htonl(coe_ctxt->frame_number);
    frame_metadata->psn = htonl(coe_ctxt->psn);
}

int64_t COETransmitter::send(DataPlaneCtxt* ctxt, const uint8_t* buffer, size_t buffer_size, FrameMetadata* frame_metadata)
{
    // COECtxt's first member chain (LinuxDataPlaneCtxt/STM32DataPlaneCtxt -> DataPlaneCtxt)
    // is the same address as the COECtxt itself; cast back to the transport type.
    COECtxt* coe_ctxt = reinterpret_cast<COECtxt*>(ctxt);
    int64_t n_bytes_sent = 0;
    const bool is_frame_end = (frame_metadata != nullptr);

    // if no destination assigned, short circuit and return 0
    if (!coe_ctxt->enable_1722b) {
        goto cleanup;
    }

    if (!COECtxt_is_in_use(coe_ctxt)) {
        update_packet_headers(coe_ctxt);
        // if we have frame_metadata and it is the default internal frame_metadata, update it
        if (frame_metadata && (frame_metadata == DEFAULT_FRAME_METADATA)) {
            update_frame_metadata_start(coe_ctxt);
        }
        COECtxt_mark_in_use(coe_ctxt);
    }

    while (buffer_size || frame_metadata) {

        int32_t sent_bytes = send_packet(coe_ctxt, buffer, buffer_size, &frame_metadata);
        if (sent_bytes < 0) {
            n_bytes_sent = -1;
            goto cleanup;
        }

        buffer_size -= sent_bytes;
        buffer += sent_bytes;
        n_bytes_sent += sent_bytes;
        coe_ctxt->psn = (coe_ctxt->psn + 1) & COE_PSN_MASK;

        if (!buffer_size && frame_metadata) {
            // The last data packet didn't have room to embed the FrameMetadata
            // in the same packet. Send a final metadata-only packet WITH the
            // FRAME_END flag set so the receiver actually finalizes this frame.
            //
            // Reuse send_packet's existing embed-metadata path by:
            //   - keeping *frame_metadata non-null (point it at the populated
            //     FrameMetadata so send_packet won't call update_frame_metadata_end
            //     a second time and overwrite bytes_written)
            //   - leaving buffer_size at 0 so to_consume == 0 (no data prefix)
            //   - moving coe_ctxt->address to metadata_offset so the receiver's
            //     "address + payload_bytes == metadata_offset + METADATA_SIZE"
            //     check passes for a 128-byte payload.
            if (frame_metadata == DEFAULT_FRAME_METADATA) {
                update_frame_metadata_end(coe_ctxt, 0);
                frame_metadata = &coe_ctxt->frame_metadata;
            }
            coe_ctxt->address = coe_ctxt->metadata_offset;
        }
    }

    coe_ctxt->frame_number++;
    // if we made it this far and sent the immediate packet, release the metadata by marking as not in use
    if (is_frame_end) {
        COECtxt_mark_available(coe_ctxt);
    }
cleanup:
    return n_bytes_sent;
}

} // namespace hololink::emulation
