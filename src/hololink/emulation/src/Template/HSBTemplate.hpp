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
 * TEMPLATE port — single header collecting every platform struct and free-function
 * declaration the common emulator code expects from the platform side. The
 * platform-named header files in this directory (data_plane.hpp, hsb_emulator.hpp,
 * net.hpp, i2c.hpp, coe_data_plane.hpp, rocev2_data_plane.hpp) are thin shims that
 * just include this file so the common include paths still resolve.
 *
 * Function bodies live in HSBTemplate.cpp as empty stubs. Fill them in with the
 * board / OS / network bindings appropriate to your target.
 */

#ifndef EMULATION_TEMPLATE_HSBTEMPLATE_HPP
#define EMULATION_TEMPLATE_HSBTEMPLATE_HPP

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <time.h> // struct timespec, clockid_t, CLOCK_REALTIME (from newlib)

// TEMPLATE is intended to be cross-compiled for a bare-metal target. The supplied
// `cmake/toolchains/arm-none-eabi-gcc.cmake` toolchain is the reference setup (no
// `__linux__` defined, no posix headers). Configure with:
//   cmake -S src/hololink/emulation -B build \
//         -DHSB_EMULATOR_TARGET=TEMPLATE \
//         -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-none-eabi-gcc.cmake
//
// No system-network headers (<netinet/in.h>, <linux/if_packet.h>, <arpa/inet.h>,
// <sys/socket.h>, …) are included. Every networking type and helper the common
// emulator code expects is declared below; provide bodies in HSBTemplate.cpp.

// Public emulator headers — define DataPlaneCtxt, HSBEmulatorCtxt, I2CControllerCtxt,
// BootpPacket, IPAddress, the host/network byte-order helpers, the IB/COE wire-format
// structs, etc. Pulled in here so the platform extensions below can name them.
#include "../../base_transmitter.hpp"
#include "../../coe_data_plane.hpp"
#include "../../data_plane.hpp"
#include "../../hsb_config.hpp"
#include "../../hsb_emulator.hpp"
#include "../../i2c_interface.hpp"
#include "../../net.hpp"
#include "../../rocev2_data_plane.hpp"
#include "../../utils.hpp"

// ============================================================================
// net.hpp surface — platform-specific networking types and helpers
// ============================================================================
//
// Mirrors the STM32/linux convention: every type and helper in this block sits
// at GLOBAL namespace, not in `hololink::emulation`. Common code (and the
// public net.hpp's macro accessors NET_GET_*_HDR / NET_GET_*_PAYLOAD) reaches
// them via unqualified lookup with no `hololink::emulation::` prefix.

// Ethernet length / type constants. Match the STM32 / linux conventions; if your
// target has its own definitions (e.g. via a vendor SDK header), include those
// instead and remove the duplicates here.
#define ETH_ALEN 6u
#define ETHER_ADDR_LEN ETH_ALEN
#define ETHER_TYPE_LEN 2u
#define ETHER_HDR_LEN (ETHER_ADDR_LEN * 2 + ETHER_TYPE_LEN)

#define ETHERTYPE_IP 0x0800u
#define ETHERTYPE_ARP 0x0806u

#define IPPROTO_UDP 17u
#define IPPROTO_TCP 6u
#define IPPROTO_ICMP 1u
#define IPPROTO_IP 0u
#define IPVERSION 4u
#define IPDEFTTL 64u
#define IP_DF 0x4000u

#define INADDR_BROADCAST ((uint32_t)0xffffffff)
#define INADDR_ANY ((uint32_t)0x00000000)
#define INADDR_LOOPBACK ((uint32_t)0x7f000001)
#define INADDR_NONE ((uint32_t)0xffffffff)

typedef uint32_t in_addr_t;

// in_addr is a posix shape; kept here so IPAddress_from_string / IPAddress_to_string
// / inet_aton / inet_ntoa can talk in the same currency as the linux / STM32 ports.
typedef struct in_addr {
    in_addr_t s_addr;
} in_addr;

// Ethernet / IP / UDP wire-format headers (host byte order for fields the
// common code writes directly; network byte order where the field is sent
// without further translation — see how STM32/linux net.hpp lay these out).
struct ether_header {
    uint8_t ether_dhost[ETH_ALEN];
    uint8_t ether_shost[ETH_ALEN];
    uint16_t ether_type;
};

// IPHDR_SET_IHL_VERSION is platform-specific because linux's <netinet/ip.h> uses
// separate `version:4` / `ihl:4` bitfields while STM32 uses a combined byte. The
// template's iphdr below uses the combined byte form; switch to bitfields if your
// platform headers do.
#define IPHDR_SET_IHL_VERSION(hdr, ihl_, version_) hdr->ihl_version = ((version_) << 4) | ((ihl_) / 4)

struct iphdr {
    uint8_t ihl_version;
    uint8_t tos;
    uint16_t tot_len;
    uint16_t id;
    uint16_t frag_off;
    uint8_t ttl;
    uint8_t protocol;
    uint16_t check;
    uint32_t saddr;
    uint32_t daddr;
};

struct udphdr {
    uint16_t source;
    uint16_t dest;
    uint16_t len;
    uint16_t check;
};

// Scatter/gather buffer used by the platform wire-emit helpers (send_coe_packet /
// send_rocev2_packet) and the common send_packet bodies. The shape
// (buffer / next / len) matches what calculate_buffer_length expects.
struct ETH_BufferTypeDef {
    uint8_t* buffer;
    struct ETH_BufferTypeDef* next;
    size_t len;
};

// Receive buffer size advertised to the rest of the code. Override if your
// target needs something different.
#ifndef RX_BUFFER_SIZE
#define RX_BUFFER_SIZE MTU_SIZE
#endif
#ifndef TX_BUFFER_SIZE
#define TX_BUFFER_SIZE MTU_SIZE
#endif

// Sum the .len fields across the scatter/gather chain. Defined in src/common/net.cpp;
// declared here because the public net.hpp can't forward-declare ETH_BufferTypeDef
// portably across the typedef/struct-tag conventions of different platforms.
uint16_t calculate_buffer_length(const ETH_BufferTypeDef* tx_buffer);

// Host/network byte-order helpers. Replace these with your target's intrinsics if
// you have hardware byte-swap (e.g. ARM's __REV / __REV16) — these portable
// versions are correct but not the fastest.
static inline uint16_t htons(uint16_t value)
{
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    return value;
#else
    return (uint16_t)((value << 8) | (value >> 8));
#endif
}
static inline uint16_t ntohs(uint16_t value) { return htons(value); }
static inline uint32_t htonl(uint32_t value)
{
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    return value;
#else
    return (value << 24) | ((value << 8) & 0x00FF0000)
        | ((value >> 8) & 0x0000FF00) | (value >> 24);
#endif
}
static inline uint32_t ntohl(uint32_t value) { return htonl(value); }

// Common emulator code calls clock_gettime(CLOCK_REALTIME, &ts) on every frame
// to stamp packet/metadata timestamps. newlib (arm-none-eabi) ships the types
// (struct timespec, clockid_t, CLOCK_REALTIME) in <time.h> but does NOT provide
// the function body — supply it in HSBTemplate.cpp using your target's clock
// source (SysTick, RTC, hardware counter, ...).
extern "C" int clock_gettime(clockid_t clock_id, struct timespec* tp);

// Common transmitter headers — declare COETransmitter / RoCEv2Transmitter classes
// referenced from HSBTemplate_coe.cpp / HSBTemplate_rocev2.cpp (and from the COECtxt /
// RoCEv2Ctxt struct definitions below). Included AFTER the global-scope network
// types because the transmitter headers reference ETH_BufferTypeDef.
#include "../common/coe_transmitter.hpp"
#include "../common/rocev2_transmitter.hpp"

namespace hololink::emulation {

// ============================================================================
// data_plane.hpp surface — platform DataPlane extension
// ============================================================================

/**
 * @brief Template-platform DataPlane context extension. `base` must be the first
 * member so a TemplateDataPlaneCtxt* aliases the common DataPlaneCtxt*. Add any
 * per-DataPlane platform state (sockets, HAL handles, buffers, ...) below `base`.
 */
struct TemplateDataPlaneCtxt {
    DataPlaneCtxt base;
    // TODO: platform specific state members for the default DataPlane.
    /* This should usually include anything that is needed for bootp or buffering for the DataPlane transmitter */
};

// ============================================================================
// hsb_emulator.hpp surface — platform HSBEmulator extension
// ============================================================================

/**
 * @brief Template-platform HSBEmulator context extension. `base` must be the first
 * member. Add target-specific resources (network handles, thread state, GPIO/SPI
 * controllers, ...) below `base`.
 */
struct TemplateHSBEmulatorCtxt {
    HSBEmulatorCtxt base;
    // TODO: target-specific state.
    /* this should usually include any context structures needed for peripheral initialization or management, but that may also live somewhere else */
};

// ============================================================================
// i2c.hpp surface — platform I2CController extension + transaction hook
// ============================================================================

/**
 * @brief Template-platform I2CControllerCtxt extension. `base` first; add per-
 * controller state (HAL handle, peripheral list, mutex, ...) below.
 */
struct TemplateI2CControllerCtxt {
    I2CControllerCtxt base;
    // TODO: target-specific state.
};

/**
 * @brief Execute a host-issued I2C transaction. Called by common/i2c.cpp's
 * i2c_configure_cb whenever the host writes the I2C control register. The
 * argument `value` packs the command in the low 16 bits and the peripheral
 * address in the high bits — see linux/STM32 i2c_transaction for the decoding.
 */
void i2c_transaction(I2CControllerCtxt* i2c_ctxt, uint32_t value);

// i2c_configure_cb / i2c_readback_cb are defined in src/common/i2c.cpp — they are
// platform-invariant. They're declared here purely so the platform i2c.hpp shim
// looks symmetric with linux/STM32 and so HSBEmulator::reset() picks them up via
// the same include path.
int i2c_configure_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count);
int i2c_readback_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count);

// ============================================================================
// coe_data_plane.hpp surface — COE per-transmission ctxt + accessor macros
// ============================================================================

// Offset from the start of the packet buffer to the start of the data buffer to
// ensure 64-bit alignment of headers.
#define COE_PACKET_OFFSET_RESET 2u
#define COE_HDR_LEN (ETHER_HDR_LEN + NTSCF_HDR_LEN + ACFUSER0C_HDR_LEN)

#define COECtxt_get_buffer_base(metadatap) (&((metadatap)->packet[COE_PACKET_OFFSET_RESET]))
#define COECtxt_get_buffer(metadatap) (&((metadatap)->packet[(metadatap)->packet_offset]))
#define COECtxt_get_buffer_size(metadatap) ((metadatap)->packet_offset - COE_PACKET_OFFSET_RESET)
#define COECtxt_buffer_clear(metadatap) ((metadatap)->packet_offset = COE_PACKET_OFFSET_RESET + COE_HDR_LEN)
#define COECtxt_get_max_size(metadatap) (sizeof(metadatap->packet) - COE_PACKET_OFFSET_RESET - sizeof(uint32_t))
#define COECtxt_mark_in_use(metadatap) ((metadatap)->in_use = true)
#define COECtxt_mark_available(metadatap) ((metadatap)->in_use = false)
#define COECtxt_is_in_use(metadatap) ((metadatap)->in_use == true)

/**
 * @brief COE per-transmission context. First-member chain
 * `&coe_ctxt == &coe_ctxt->base == &coe_ctxt->base.base == DataPlaneCtxt*` lets
 * COEDataPlane hand it straight to the protected DataPlane(...) ctor. Add any
 * target-specific COE transport state (socket fd, HAL handle, dest address, ...)
 * after the shared fields.
 */
struct COECtxt {
    TemplateDataPlaneCtxt base;
    uint32_t frame_size;
    uint32_t payload_size;
    uint32_t line_threshold;
    uint32_t frame_number;
    uint32_t psn;
    uint32_t line_offset;
    uint32_t address;
    uint32_t metadata_offset;
    FrameMetadata frame_metadata;
    alignas(uint32_t) uint8_t packet[TX_BUFFER_SIZE + COE_PACKET_OFFSET_RESET];
    uint16_t packet_offset;
    uint8_t channel;
    bool in_use;
    bool enable_1722b;
    // TODO: add target-specific COE transport handles (HAL eth_handle, DMA channel,
    // destination MAC cache, ...). The linux build adds a raw AF_PACKET socket fd
    // and sockaddr_ll here; STM32 holds an ETH_HandleTypeDef pointer. Pick whatever
    // your transport needs.
};

// ============================================================================
// rocev2_data_plane.hpp surface — RoCEv2 per-transmission ctxt + accessor macros
// ============================================================================

#define ROCEV2_PACKET_OFFSET_RESET 2u
#define ROCEV2_HDR_LEN (ETHER_HDR_LEN + IP_HDR_LEN + UDP_HDR_LEN + BT_HDR_LEN + RET_HDR_LEN)

#define RoCEv2Ctxt_get_buffer_base(metadatap) (&((metadatap)->packet[ROCEV2_PACKET_OFFSET_RESET]))
#define RoCEv2Ctxt_get_buffer(metadatap) (&((metadatap)->packet[(metadatap)->packet_offset]))
#define RoCEv2Ctxt_get_buffer_size(metadatap) ((metadatap)->packet_offset - ROCEV2_PACKET_OFFSET_RESET)
#define RoCEv2Ctxt_buffer_clear(metadatap) ((metadatap)->packet_offset = ROCEV2_PACKET_OFFSET_RESET + ROCEV2_HDR_LEN)
#define RoCEv2Ctxt_get_max_size(metadatap) (sizeof(metadatap->packet) - ROCEV2_PACKET_OFFSET_RESET - sizeof(uint32_t))
#define RoCEv2Ctxt_mark_in_use(metadatap) ((metadatap)->in_use = true)
#define RoCEv2Ctxt_mark_available(metadatap) ((metadatap)->in_use = false)
#define RoCEv2Ctxt_is_in_use(metadatap) ((metadatap)->in_use == true)

/**
 * @brief RoCEv2 per-transmission context. Same first-member-chain pattern as
 * COECtxt.
 */
struct RoCEv2Ctxt {
    TemplateDataPlaneCtxt base;
    uint64_t virtual_address;
    uint32_t metadata_offset;
    uint32_t payload_size;
    uint32_t frame_number;
    uint32_t psn;
    FrameMetadata frame_metadata;
    alignas(uint64_t) uint8_t packet[TX_BUFFER_SIZE + ROCEV2_PACKET_OFFSET_RESET];
    uint16_t page;
    uint16_t packet_offset;
    bool in_use { false };
    // TODO: add target-specific RoCEv2 transport handles (HAL eth_handle, DMA channel,
    // destination IP cache, ...). The linux build adds a UDP socket fd and sockaddr_in
    // here; STM32 holds an ETH_HandleTypeDef pointer.
};

// ============================================================================
// Platform wire-emit hooks — implementations are stubs in HSBTemplate.cpp.
// ============================================================================

/**
 * @brief Wire-emit hook for COE packets, called by common/coe_transmitter.cpp's
 * send_packet after it has built the scatter/gather list. Return bytes sent or
 * < 0 on failure.
 */
int16_t send_coe_packet(COECtxt* coe_ctxt, ETH_BufferTypeDef* tx_buffers);

/**
 * @brief Wire-emit hook for RoCEv2 packets, called by common/rocev2_transmitter.cpp's
 * send_packet after it has built the scatter/gather list. Return bytes sent or
 * < 0 on failure.
 */
int16_t send_rocev2_packet(RoCEv2Ctxt* rocev2_ctxt, ETH_BufferTypeDef* tx_buffers);

} // namespace hololink::emulation

#endif
