#ifndef STM32_ROCEV2_TRANSMITTER_HPP
#define STM32_ROCEV2_TRANSMITTER_HPP

#include "../../rocev2_data_plane.hpp"
#include "../common/rocev2_transmitter.hpp"
#include "data_plane.hpp"

// offset from the start of the packet buffer to the start of the data buffer to ensure 64-bit alignment of headers
#define ROCEV2_PACKET_OFFSET_RESET 2u

#define ROCEV2_HDR_LEN (ETHER_HDR_LEN + IP_HDR_LEN + UDP_HDR_LEN + BT_HDR_LEN + RET_HDR_LEN)
#define RoCEv2Ctxt_get_buffer_base(metadatap) (&((metadatap)->packet[ROCEV2_PACKET_OFFSET_RESET]))
#define RoCEv2Ctxt_get_buffer(metadatap) (&((metadatap)->packet[(metadatap)->packet_offset]))
#define RoCEv2Ctxt_get_buffer_size(metadatap) ((metadatap)->packet_offset - ROCEV2_PACKET_OFFSET_RESET)
#define RoCEv2Ctxt_buffer_clear(metadatap) ((metadatap)->packet_offset = ROCEV2_PACKET_OFFSET_RESET + ROCEV2_HDR_LEN)
// static packet size (MTU + ROCEV2_PACKET_OFFSET_RESET) - ROCEV2_PACKET_OFFSET_RESET - iCRC data (4 bytes)
#define RoCEv2Ctxt_get_max_size(metadatap) (sizeof(metadatap->packet) - ROCEV2_PACKET_OFFSET_RESET - sizeof(uint32_t))
#define RoCEv2Ctxt_mark_in_use(metadatap) ((metadatap)->in_use = true)
#define RoCEv2Ctxt_mark_available(metadatap) ((metadatap)->in_use = false)
#define RoCEv2Ctxt_is_in_use(metadatap) ((metadatap)->in_use == true)

namespace hololink::emulation {

/**
 * Per-transmission context for RoCEv2Transmitter (`RoCEv2Ctxt`).
 *
 * Holds RoCEv2 state that can change every frame sent.
 */
struct RoCEv2Ctxt {
    // First member is the STM32 DataPlaneCtxt extension; chain reaches the common
    // DataPlaneCtxt. &rocev2_ctxt == &rocev2_ctxt->base == &rocev2_ctxt->base.base ==
    // DataPlaneCtxt*. Register slices reachable via base.base.dp_registers /
    // base.base.dp_sensor_registers.
    STM32DataPlaneCtxt base;
    // this shall be 64-bit aligned
    uint64_t virtual_address;
    uint32_t metadata_offset;
    // frame_metadata also has these values, but they are in network byte order. Use these for actual tracking
    uint32_t frame_number;
    uint32_t psn;
    uint16_t payload_size;
    uint16_t page;
    // frame_metadata, packet, and packet_offset are possibly not in all implementations
    // FrameMetadata should be 64-bit aligned
    FrameMetadata frame_metadata;
    // should already be 64-bit aligned
    alignas(uint64_t) uint8_t packet[TX_BUFFER_SIZE + ROCEV2_PACKET_OFFSET_RESET];
    uint16_t packet_offset;
    // final due to alignment requirements
    bool in_use { false };
};

} // namespace hololink::emulation

#endif