#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <time.h>

#include "dlpack/dlpack.h"

#include "base_transmitter.hpp"
#include "data_plane.hpp"
#include "net.hpp"
#include "rocev2_data_plane.hpp"
#include "rocev2_transmitter.hpp"
#include "utils.hpp"

namespace hololink::emulation {

// Platform-specific wire emit. Builds the platform's transport (Linux: sendmsg over a
// UDP socket — kernel handles L2/L3; STM32: HAL_ETH_Transmit) from the scatter/gather
// list assembled by the common send_packet() below. Defined in linux/rocev2_transmitter.cpp
// and STM32/rocev2_transmitter.cpp.
int16_t send_rocev2_packet(RoCEv2Ctxt* rocev2_ctxt, ETH_BufferTypeDef* tx_buffers);

int32_t send_packet(RoCEv2Ctxt* rocev2_ctxt, const uint8_t* buffer, size_t buffer_size, FrameMetadata** frame_metadata);

void RoCEv2Transmitter::init_metadata(RoCEv2Ctxt* metadata, const IPAddress& source_ip)
{
    memset(metadata->packet, 0, sizeof(metadata->packet));
    metadata->packet_offset = ROCEV2_PACKET_OFFSET_RESET;
    RoCEv2Ctxt_mark_available(metadata);

    // initialize headers in payload buffer
    // set ethernet header
    struct ether_header* eth = (struct ether_header*)RoCEv2Ctxt_get_buffer(metadata);
    memcpy(eth->ether_shost, source_ip.mac, ETHER_ADDR_LEN);
    eth->ether_type = htons(ETHERTYPE_IP);

    // set IP header
    struct iphdr* ip_header = (struct iphdr*)NET_GET_ETHER_PAYLOAD(eth);
    IPHDR_SET_IHL_VERSION(ip_header, IP_HDR_LEN, IPVERSION);
    ip_header->frag_off = htons(IP_DF); // 0x4000 not used without fragmentation
    ip_header->ttl = IPDEFTTL; // not used without fragmentation
    ip_header->protocol = IPPROTO_UDP; // UDP
    ip_header->saddr = source_ip.ip_address;
    ip_header->daddr = htonl(metadata->base.base.dp_sensor_registers->vp_data[DP_HOST_IP / REGISTER_SIZE]);

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
void update_packet_headers(RoCEv2Ctxt* rocev2_ctxt)
{
    struct DPSensorRegisters* dp_sensor_reg = rocev2_ctxt->base.base.dp_sensor_registers;
    // reset the packet offset to the beginning of the packet
    rocev2_ctxt->packet_offset = ROCEV2_PACKET_OFFSET_RESET;
    // update packet buffer headers
    struct ether_header* ether_header = (struct ether_header*)RoCEv2Ctxt_get_buffer(rocev2_ctxt);
    uint32_t host_mac = dp_sensor_reg->vp_data[DP_HOST_MAC_HIGH / REGISTER_SIZE];
    ether_header->ether_dhost[0] = (host_mac >> 8) & 0xFF;
    ether_header->ether_dhost[1] = (host_mac >> 0) & 0xFF;
    host_mac = dp_sensor_reg->vp_data[DP_HOST_MAC_LOW / REGISTER_SIZE];
    ether_header->ether_dhost[2] = (host_mac >> 24) & 0xFF;
    ether_header->ether_dhost[3] = (host_mac >> 16) & 0xFF;
    ether_header->ether_dhost[4] = (host_mac >> 8) & 0xFF;
    ether_header->ether_dhost[5] = (host_mac >> 0) & 0xFF;

    struct iphdr* ip_header = (struct iphdr*)NET_GET_ETHER_PAYLOAD(ether_header);
    ip_header->daddr = htonl(dp_sensor_reg->vp_data[DP_HOST_IP / REGISTER_SIZE]);

    struct udphdr* udp_header = (struct udphdr*)NET_GET_IP_PAYLOAD(ip_header);
    udp_header->dest = htons(dp_sensor_reg->vp_data[DP_HOST_UDP_PORT / REGISTER_SIZE] & 0xFFFF);

    struct bthdr* bt_header = (struct bthdr*)NET_GET_UDP_PAYLOAD(udp_header);
    bt_header->opcode = IB_OPCODE_WRITE;
    bt_header->destqp = htonl((0xFF << 24) | (dp_sensor_reg->vp_data[DP_QP / REGISTER_SIZE] & 0xFFFFFF));

    struct rethdr* ret_header = (struct rethdr*)NET_GET_BT_PAYLOAD(bt_header);
    ret_header->va = htonll(rocev2_ctxt->virtual_address);
    ret_header->r_key = htonl(dp_sensor_reg->vp_data[DP_RKEY / REGISTER_SIZE]);
    ret_header->dmalen = htonl(rocev2_ctxt->payload_size);

    rocev2_ctxt->packet_offset += ROCEV2_HDR_LEN;
#ifdef __linux__
    // TODO: This should probably be added as a callback
    rocev2_ctxt->dest_addr = {
        .sin_family = AF_INET,
        .sin_port = udp_header->dest,
        .sin_addr = ip_header->daddr,
    };
#endif
}

#ifdef CRC_OFFLOAD
// Lazily initialize and return a reference to a CRC peripheral handle configured for the
// RoCEv2 iCRC polynomial. The STM32 CRC unit natively implements CRC-32 with polynomial
// 0x04C11DB7 and initial value 0xFFFFFFFF. Input byte inversion feeds the LSB of each byte
// first (big-endian byte order, LSB-first within a byte) and output bit inversion with a
// final bitwise complement match the Ethernet/zlib CRC-32 convention used for RoCEv2 iCRC.
static inline CRC_HandleTypeDef& get_icrc_handle()
{
    static CRC_HandleTypeDef hcrc = [] {
        CRC_HandleTypeDef h = {};
        h.Instance = CRC;
        h.Init.DefaultPolynomialUse = DEFAULT_POLYNOMIAL_ENABLE;
        h.Init.DefaultInitValueUse = DEFAULT_INIT_VALUE_ENABLE;
        h.Init.InputDataInversionMode = CRC_INPUTDATA_INVERSION_BYTE;
        h.Init.OutputDataInversionMode = CRC_OUTPUTDATA_INVERSION_ENABLE;
        h.InputDataFormat = CRC_INPUTDATA_FORMAT_BYTES;
        __HAL_RCC_CRC_CLK_ENABLE();
        if (HAL_CRC_Init(&h) != HAL_OK) {
            Error_Handler(NULL);
        }
        return h;
    }();
    return hcrc;
}

static inline uint32_t calculate_crc(const ETH_BufferTypeDef* tx_buffer)
{
    static const uint8_t crc_init_buf[8] = { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };
    CRC_HandleTypeDef& hcrc = get_icrc_handle();
    const ETH_BufferTypeDef* cur = tx_buffer;
    // HAL_CRC_Calculate resets the CRC data register to the configured init value (0xFFFFFFFF)
    // before accumulating. The 8 dummy LRH bytes are fed first per the IBA/RoCEv2 iCRC spec.
    HAL_CRC_Calculate(&hcrc, (uint32_t*)crc_init_buf, sizeof(crc_init_buf));
    // need to skip the ethernet header in the CRC calculation
    uint32_t crc = HAL_CRC_Accumulate(&hcrc, (uint32_t*)(cur->buffer + ETHER_HDR_LEN), cur->len - ETHER_HDR_LEN);
    while (cur->next) {
        cur = cur->next;
        crc = HAL_CRC_Accumulate(&hcrc, (uint32_t*)cur->buffer, cur->len);
    }
    // spec: "the bit sequence from the calculation is complemented and the result is the ICRC"
    return ~crc;
}
#else
#include <zlib.h>
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
#endif

// tx_buffer_base is the first buffer in the chain that does not contain the CRC
// it must include all the headers and the payload
uint32_t condition_packet_crc(RoCEv2Ctxt* metadata, const ETH_BufferTypeDef* tx_buffer_base)
{
    // set up headers
    struct iphdr* ip_header = (struct iphdr*)(RoCEv2Ctxt_get_buffer_base(metadata) + IP_HDR_OFFSET);
    struct udphdr* udp_header = (struct udphdr*)NET_GET_IP_PAYLOAD(ip_header);
    struct bthdr* bt_header = (struct bthdr*)NET_GET_UDP_PAYLOAD(udp_header);
    struct rethdr* ret_header = (struct rethdr*)NET_GET_BT_PAYLOAD(bt_header);
    // initialize payload to the size of the buffer chain and add the CRC
    // ip header length is the full size of payload of ethernet frame including infiniband crc
    uint16_t payload_size = calculate_buffer_length(tx_buffer_base) + sizeof(uint32_t) - ETHER_HDR_LEN;
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

static inline void update_frame_metadata_start(RoCEv2Ctxt* rocev2_ctxt)
{
    FrameMetadata* frame_metadata = &rocev2_ctxt->frame_metadata;
    struct timespec timestamp;
    if (clock_gettime(CLOCK_REALTIME, &timestamp)) {
        Error_Handler("Failed to get timestamp in update_frame_metadata_start");
    }
    frame_metadata->timestamp_s_low = htonl((uint32_t)timestamp.tv_sec);
    frame_metadata->timestamp_ns = htonl((uint32_t)timestamp.tv_nsec);
}

void update_frame_metadata_end(RoCEv2Ctxt* rocev2_ctxt, uint16_t packet_bytes)
{
    FrameMetadata* frame_metadata = &rocev2_ctxt->frame_metadata;
    // total number of bytes written is the current location of va in the rethdr minus the initial va
    struct rethdr* ret_header = (struct rethdr*)(RoCEv2Ctxt_get_buffer_base(rocev2_ctxt) + RET_HDR_OFFSET);
    int64_t bytes_written = ntohll(ret_header->va) - rocev2_ctxt->virtual_address + packet_bytes;
    struct timespec timestamp;
    if (clock_gettime(CLOCK_REALTIME, &timestamp)) {
        Error_Handler("Failed to get timestamp in update_frame_metadata_end");
    }
    frame_metadata->metadata_s_low = htonl((uint32_t)timestamp.tv_sec);
    frame_metadata->metadata_ns = htonl((uint32_t)timestamp.tv_nsec);
    frame_metadata->bytes_written_high = htonl((uint32_t)(bytes_written >> 32));
    frame_metadata->bytes_written_low = htonl((uint32_t)(bytes_written & 0xFFFFFFFF));
    frame_metadata->frame_number = htonl(rocev2_ctxt->frame_number);
    frame_metadata->psn = htonl(rocev2_ctxt->psn);
}

static inline void write_immediate_data(RoCEv2Ctxt* rocev2_ctxt, uint32_t immediate_value)
{
    uint8_t* immediate_buffer = RoCEv2Ctxt_get_buffer(rocev2_ctxt);
    memcpy(immediate_buffer, &immediate_value, sizeof(immediate_value));
    rocev2_ctxt->packet_offset += sizeof(immediate_value);
}

static inline bool is_write_immediate(const RoCEv2Ctxt* rocev2_ctxt)
{
    struct bthdr* bt_header = (struct bthdr*)(RoCEv2Ctxt_get_buffer_base(rocev2_ctxt) + ETHER_HDR_LEN + IP_HDR_LEN + UDP_HDR_LEN);
    return bt_header->opcode == IB_OPCODE_WRITE_IMMEDIATE;
}

void set_opcode(RoCEv2Ctxt* rocev2_ctxt, uint8_t opcode)
{
    struct bthdr* bt_header = (struct bthdr*)(RoCEv2Ctxt_get_buffer_base(rocev2_ctxt) + ETHER_HDR_LEN + IP_HDR_LEN + UDP_HDR_LEN);
    bt_header->opcode = opcode;
}

int64_t RoCEv2Transmitter::send(DataPlaneCtxt* ctxt, const uint8_t* buffer, size_t buffer_size, FrameMetadata* frame_metadata)
{
    // RoCEv2Ctxt's first member chain (LinuxDataPlaneCtxt/STM32DataPlaneCtxt ->
    // DataPlaneCtxt) is the same address as the RoCEv2Ctxt itself; cast back.
    RoCEv2Ctxt* rocev2_ctxt = reinterpret_cast<RoCEv2Ctxt*>(ctxt);
    struct DPSensorRegisters* dp_sensor_reg = rocev2_ctxt->base.base.dp_sensor_registers;
    int64_t n_bytes_sent = 0;

    // if no destination assigned, short circuit and return 0
    if (!dp_sensor_reg->vp_data[DP_HOST_IP / REGISTER_SIZE] || !dp_sensor_reg->vp_data[DP_HOST_UDP_PORT / REGISTER_SIZE]) {
        goto cleanup;
    }

    if (!RoCEv2Ctxt_is_in_use(rocev2_ctxt)) {
        RoCEv2Ctxt_mark_in_use(rocev2_ctxt);
        update_packet_headers(rocev2_ctxt);
        // if we have frame_metadata and it is the default internal frame_metadata, update it
        if (frame_metadata && (frame_metadata == DEFAULT_FRAME_METADATA)) {
            update_frame_metadata_start(rocev2_ctxt);
        }
    }

    while (buffer_size || frame_metadata) {

        int32_t sent_bytes = send_packet(rocev2_ctxt, buffer, buffer_size, &frame_metadata);
        if (sent_bytes < 0) {
            n_bytes_sent = -1;
            goto cleanup;
        }

        buffer_size -= sent_bytes;
        buffer += sent_bytes;
        n_bytes_sent += sent_bytes;
        rocev2_ctxt->psn = (rocev2_ctxt->psn + 1) & IB_PSN_MASK;

        if (!buffer_size && frame_metadata) {
            // The previous data packet didn't have room to embed the FrameMetadata, so we
            // send a metadata-only WRITE_IMMEDIATE follow-up. Configure the immediate header
            // and arrange to feed the FrameMetadata as send_packet's "data" payload.
            set_opcode(rocev2_ctxt, IB_OPCODE_WRITE_IMMEDIATE);
            uint32_t immediate_value = htonl(rocev2_ctxt->page | ((rocev2_ctxt->psn & IB_PSN_MASK) << IB_PSN_SHIFT));
            write_immediate_data(rocev2_ctxt, immediate_value);
            if (frame_metadata == DEFAULT_FRAME_METADATA) {
                update_frame_metadata_end(rocev2_ctxt, 0);
                buffer = (uint8_t*)&rocev2_ctxt->frame_metadata;
            } else {
                buffer = (uint8_t*)frame_metadata;
            }
            // The previous packet's cleanup left ret_header->va at the byte just past the
            // end of the frame data (virtual_address + frame_size), but the FrameMetadata
            // must land at the page-aligned slot virtual_address + metadata_offset. The host
            // RDMA receiver only writes payload bytes to va and leaves the gap untouched, so
            // bump va here before the next send_packet reads it. Mirror of the
            // coe_ctxt->address = coe_ctxt->metadata_offset assignment in COETransmitter.
            struct rethdr* ret_header = (struct rethdr*)(RoCEv2Ctxt_get_buffer_base(rocev2_ctxt) + RET_HDR_OFFSET);
            ret_header->va = htonll(rocev2_ctxt->virtual_address + rocev2_ctxt->metadata_offset);
            frame_metadata = nullptr;
            buffer_size = sizeof(FrameMetadata);
        }
    }

    rocev2_ctxt->frame_number++;
    // if we made it this far and sent the immediate packet, release the metadata by marking as not in use
    if (is_write_immediate(rocev2_ctxt)) {
        RoCEv2Ctxt_mark_available(rocev2_ctxt);
    }
cleanup:
    // for success or failure whether opcode is write immediate or write, reset to write
    set_opcode(rocev2_ctxt, IB_OPCODE_WRITE);
    return n_bytes_sent;
}

int32_t send_packet(RoCEv2Ctxt* rocev2_ctxt, const uint8_t* buffer, size_t buffer_size, FrameMetadata** frame_metadata)
{
    static uint8_t zeros[HSB_PAGE_SIZE] = { 0 };
    uint32_t zeros_len = sizeof(zeros);

    constexpr int kMaxTxBuffers = 6;
    ETH_BufferTypeDef tx_buffers[kMaxTxBuffers];
    int buffer_index = 0;

    // [0] header chunk (Ethernet/IP/UDP/BTH/RETH built into rocev2_ctxt->packet[])
    // STM32 and Linux ETH_BufferTypeDef have the buffer/len/next members in different
    // orders; omit .next from the initializer (zero-init) and chain it explicitly.
    tx_buffers[buffer_index] = {};
    tx_buffers[buffer_index].buffer = RoCEv2Ctxt_get_buffer_base(rocev2_ctxt);
    tx_buffers[buffer_index].len = RoCEv2Ctxt_get_buffer_size(rocev2_ctxt);

    uint16_t to_consume = rocev2_ctxt->payload_size;
    if (to_consume > buffer_size) {
        to_consume = static_cast<uint16_t>(buffer_size);
    }

    struct rethdr* ret_header = (struct rethdr*)(RoCEv2Ctxt_get_buffer_base(rocev2_ctxt) + RET_HDR_OFFSET);
    uint64_t va = ntohll(ret_header->va);

    // Gated by SEPARATE_FRAMEMETADATA_PACKET: when =1, never embed — leave
    // *frame_metadata non-null and let the upper-level send() loop issue a separate
    // metadata-only WRITE_IMMEDIATE follow-up packet. When =0 (default on STM32),
    // try to merge as below.
    bool insert_frame_metadata = false;
    uint32_t immediate_value;
#if !SEPARATE_FRAMEMETADATA_PACKET
    if (*frame_metadata) {
        int32_t available_space = static_cast<int32_t>(rocev2_ctxt->payload_size);
        available_space -= to_consume;
        available_space -= sizeof(FrameMetadata);
        available_space -= sizeof(uint32_t); // immediate header; max_size accounts for iCRC
        zeros_len = rocev2_ctxt->metadata_offset - (uint32_t)(va - rocev2_ctxt->virtual_address + to_consume);
        available_space -= zeros_len;
        if (available_space >= (int16_t)sizeof(zeros) && sizeof(zeros) >= zeros_len) {
            insert_frame_metadata = true;
            set_opcode(rocev2_ctxt, IB_OPCODE_WRITE_IMMEDIATE);
            // [1] immediate header
            immediate_value = htonl(rocev2_ctxt->page | ((rocev2_ctxt->psn & IB_PSN_MASK) << IB_PSN_SHIFT));
            ++buffer_index;
            tx_buffers[buffer_index] = {};
            tx_buffers[buffer_index].buffer = (uint8_t*)&immediate_value;
            tx_buffers[buffer_index].len = sizeof(immediate_value);
            tx_buffers[buffer_index - 1].next = &tx_buffers[buffer_index];
        }
    }
#else
    (void)immediate_value;
    (void)zeros;
    (void)zeros_len;
#endif

    // [next] payload chunk
    ++buffer_index;
    tx_buffers[buffer_index] = {};
    tx_buffers[buffer_index].buffer = (uint8_t*)buffer;
    tx_buffers[buffer_index].len = to_consume;
    tx_buffers[buffer_index - 1].next = &tx_buffers[buffer_index];

    if (insert_frame_metadata) {
        // [next] zeros pad — FrameMetadata must land at virtual_address + metadata_offset
        ++buffer_index;
        tx_buffers[buffer_index] = {};
        tx_buffers[buffer_index].buffer = &zeros[0];
        tx_buffers[buffer_index].len = zeros_len;
        tx_buffers[buffer_index - 1].next = &tx_buffers[buffer_index];

        if (*frame_metadata == DEFAULT_FRAME_METADATA) {
            update_frame_metadata_end(rocev2_ctxt, to_consume);
            *frame_metadata = &rocev2_ctxt->frame_metadata;
        }
        // [next] FrameMetadata
        ++buffer_index;
        tx_buffers[buffer_index] = {};
        tx_buffers[buffer_index].buffer = (uint8_t*)*frame_metadata;
        tx_buffers[buffer_index].len = sizeof(FrameMetadata);
        tx_buffers[buffer_index - 1].next = &tx_buffers[buffer_index];
    }

    // [last] iCRC trailer (computed across the chain assembled so far)
    uint32_t crc = condition_packet_crc(rocev2_ctxt, &tx_buffers[0]);
    ++buffer_index;
    tx_buffers[buffer_index] = {};
    tx_buffers[buffer_index].buffer = (uint8_t*)&crc;
    tx_buffers[buffer_index].len = sizeof(uint32_t);
    tx_buffers[buffer_index - 1].next = &tx_buffers[buffer_index];

    // Wire emit — platform-specific.
    if (send_rocev2_packet(rocev2_ctxt, &tx_buffers[0]) < 0) {
        return -1;
    }

    // cleanup — bump ret_header->va so the next packet lands where the host expects
    RoCEv2Ctxt_buffer_clear(rocev2_ctxt); // only relevant if buffering is enabled (not default)
    if (insert_frame_metadata) {
        ret_header->va = htonll(va + to_consume + zeros_len + sizeof(FrameMetadata));
        *frame_metadata = nullptr;
    } else {
        ret_header->va = htonll(va + to_consume);
    }
    return to_consume;
}

} // namespace hololink::emulation