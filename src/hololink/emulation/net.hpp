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

#ifndef EMULATION_NET_HPP
#define EMULATION_NET_HPP

#include <cstdint>
#include <string>
#include <tuple>

#include "hsb_config.hpp"

#ifdef __linux__
#include <net/if.h>
#else
#define IFNAMSIZ 16
#endif

#define ETHER_BROADCAST_ADDR               \
    {                                      \
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF \
    }

/* AVTP IEEE 1722 */
#define ETHERTYPE_AVTP 0x22F0u

// maximum size of an Ethernet frame excluding CRC/VLAN. target-invariant.
#define MTU_SIZE 1514u

// calculate_buffer_length(): the function declaration lives in the platform net.hpp files
// where ETH_BufferTypeDef is concretely defined (STM32: HAL typedef; Linux: a struct in
// src/linux/net.hpp). The single implementation lives in src/common/net.cpp.

// 64-bit host/network byte-order helpers. Neither POSIX <arpa/inet.h> nor the
// STM32 HAL provides these; they were previously hand-rolled identically in
// linux/net.hpp and STM32/net.hpp. Target-invariant.
static inline uint64_t htonll(uint64_t value)
{
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    return value;
#else
    return ((value & 0x00000000000000FFULL) << 56) | ((value & 0x000000000000FF00ULL) << 40) | ((value & 0x0000000000FF0000ULL) << 24) | ((value & 0x00000000FF000000ULL) << 8) | ((value & 0x000000FF00000000ULL) >> 8) | ((value & 0x0000FF0000000000ULL) >> 24) | ((value & 0x00FF000000000000ULL) >> 40) | ((value & 0xFF00000000000000ULL) >> 56);
#endif
}

static inline uint64_t ntohll(uint64_t value)
{
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    return value;
#else
    return ((value & 0x00000000000000FFULL) << 56) | ((value & 0x000000000000FF00ULL) << 40) | ((value & 0x0000000000FF0000ULL) << 24) | ((value & 0x00000000FF000000ULL) << 8) | ((value & 0x000000FF00000000ULL) >> 8) | ((value & 0x0000FF0000000000ULL) >> 24) | ((value & 0x00FF000000000000ULL) >> 40) | ((value & 0xFF00000000000000ULL) >> 56);
#endif
}

// ----------------------------------------------------------------------------
// Ethernet / IP / UDP layered offsets and payload accessors (target-invariant).
// The chain ultimately depends on ETHER_HDR_LEN, which is platform-defined:
//   - Linux: pulled in via <net/ethernet.h> in linux/net.hpp
//   - STM32: defined as (ETHER_ADDR_LEN * 2 + ETHER_TYPE_LEN) in STM32/net.hpp
// All resolution happens at expansion time in the consuming TU, which always
// includes one of the two platform net.hpp files (which themselves include
// this header).
// ----------------------------------------------------------------------------

#define ETHER_PAYLOAD_OFFSET ETHER_HDR_LEN
#define NET_GET_ETHER_HDR(hal_buffer_td) ((struct ether_header*)((hal_buffer_td)->buffer))
#define NET_GET_ETHER_PAYLOAD(hdr) ((uint8_t*)(hdr) + ETHER_HDR_LEN)

#define IP_HDR_LEN 20u
#define IP_ADDR_LEN 4u
#define IP_HDR_OFFSET ETHER_PAYLOAD_OFFSET
#define IP_PAYLOAD_OFFSET (IP_HDR_OFFSET + IP_HDR_LEN)
#define NET_GET_IP_HDR(hal_buffer_td) ((struct iphdr*)((hal_buffer_td)->buffer + ETHER_PAYLOAD_OFFSET))
#define NET_GET_IP_PAYLOAD(hdr) ((uint8_t*)(hdr) + IP_HDR_LEN)

#define UDP_HDR_LEN 8u
#define UDP_PAYLOAD_OFFSET (IP_PAYLOAD_OFFSET + UDP_HDR_LEN)
#define NET_GET_UDP_HDR(hal_buffer_td) ((struct udphdr*)((hal_buffer_td)->buffer + IP_PAYLOAD_OFFSET))
#define NET_GET_UDP_PAYLOAD(hdr) ((uint8_t*)(hdr) + UDP_HDR_LEN)

// ----------------------------------------------------------------------------
// InfiniBand BTH / RETH wire formats and constants (target-invariant)
// Used by RoCEv2 transport on both Linux and STM32.
// ----------------------------------------------------------------------------

#define IB_OPCODE_WRITE 0x2A
#define IB_OPCODE_WRITE_IMMEDIATE 0x2B

#define BT_HDR_LEN 12u
#define BT_HDR_OFFSET UDP_PAYLOAD_OFFSET
#define BT_PAYLOAD_OFFSET (BT_HDR_OFFSET + BT_HDR_LEN)
#define NET_GET_BT_PAYLOAD(hdr) ((uint8_t*)(hdr) + BT_HDR_LEN)

// from Infiniband specification
struct bthdr {
    uint8_t opcode; // opcode {= 0x2A for write, 0x2B for write immediate}
    // Note we should be paying attention to the pad count flag in bits 5:4
    // if we want to have non-4-byte boundary payloads.
    uint8_t flags; // flags {= 0}.
    uint16_t p_key; // partition {= 0xFFFF}
    uint32_t destqp; // destination qp (lower 24 bits only) {= 0xFF << 24 | qp}
    uint32_t psn; // psn only lower 24 bits are used. ack bit on 31.
};

static_assert(sizeof(struct bthdr) == BT_HDR_LEN, "bthdr size mismatch with BT_HDR_LEN");

#define RET_HDR_LEN 16u
#define RET_HDR_OFFSET BT_PAYLOAD_OFFSET

// from Infiniband specification
struct rethdr {
    uint64_t va; // virtual address where data is stored on receiver side
    uint32_t r_key; // rkey
    uint32_t dmalen; // dma length of the payload. Does not include headers or iCRC
};

static_assert(sizeof(struct rethdr) == RET_HDR_LEN, "rethdr size mismatch with RET_HDR_LEN");

// ----------------------------------------------------------------------------
// IEEE 1722B COE wire formats and constants (target-invariant)
// Used by COE transport on both Linux and STM32.
// ----------------------------------------------------------------------------

// constant in HSB 1722 packets
struct AVTPDUCommonHeader {
    uint8_t subtype; // 0x82
    // header specific and version fields subsumed by that header
};

#define NTSCF_HDR_LEN 12u
#define NTSCF_HDR_OFFSET ETHER_PAYLOAD_OFFSET
#define NTSCF_PAYLOAD_OFFSET (NTSCF_HDR_OFFSET + NTSCF_HDR_LEN)
#define NET_GET_NTSCF_PAYLOAD(hdr) ((uint8_t*)(hdr) + NTSCF_HDR_LEN)

// constants in HSB 1722 packets
struct NTSCFHeader {
    AVTPDUCommonHeader avtpdu_header;
    uint8_t version_ntscf_len_high; // lowest 3 bits for NTSCF data
    uint8_t ntscf_len_low; // data length occupies 11 lsbs. See S 9.2 Figure 50 of IEEE 1722.1-2016. This is the size of the ACF_Payload (everything after) NTSCFHeader
    uint8_t sequence_num; // this is NOT packet sequence number
    uint8_t stream_id[8]; // defined in IEEE 802.1Q-2014
};

static_assert(sizeof(NTSCFHeader) == NTSCF_HDR_LEN, "NTSCFHeader size mismatch with NTSCF_HDR_LEN");

// constants in HSB 1722 packets
struct ACFCommonHeader {
    uint16_t acf_metadata; // msg_type occupies 7 msbs, msg_length occupies 9 lsbs. See S 9.4 Figure 52 of IEEE 1722.1-2106
                           // msg_length is the number of quadlets (4 bytes/quadlet) in the ACF_Payload (including this header)
                           // NOTE: the way the msg_length is read, it is not 9 bits, but 8.
};

#define ACFUSER0C_HDR_LEN 20u
#define ACFUSER0C_HDR_OFFSET NTSCF_PAYLOAD_OFFSET

struct ACFUser0CHeader {
    ACFCommonHeader acf_header;
    uint8_t reserved;
    uint8_t sensor_info; // SIF port index
    uint32_t timestamp_sec;
    uint32_t timestamp_nsec;
    uint8_t psn; // packet sequence number. only 7 bits are used, so max is 127
    uint8_t flags;
    uint8_t channel; // only 6 bits are used, so max is 63
    uint8_t frame_flags;
    uint32_t address; // first 4 bits are 0b00 and 2 bits of frame number. Then 28-bit of virtual address offset
};

static_assert(sizeof(ACFUser0CHeader) == ACFUSER0C_HDR_LEN, "ACFUser0CHeader size mismatch with ACFUSER0C_HDR_LEN");

namespace hololink::emulation {

#define DATA_SOURCE_UDP_PORT 12288u
#define UDP_PACKET_SIZE 10240u

#define IP_ADDRESS_MAX_BITS 32u
#define IP_ADDRESS_DEFAULT_BITS 24u

#define IPADDRESS_HAS_ADDR 0x1
#define IPADDRESS_HAS_NETMASK 0x2
#define IPADDRESS_HAS_BROADCAST 0x4
#define IPADDRESS_IS_PTP 0x40
#define IPADDRESS_HAS_MAC 0x80

#define HSB_PAGE_SIZE 128u

#define IB_PSN_MASK 0xFFFFFu
#define IB_PSN_SHIFT 12u

#define COE_PSN_MASK 0x7Fu

/**
 * python (limited API):
 *
 * `def __init__(self: hemu.IPAddress, ip_address: str, /)`
 *
 * @brief Encapsulates all the relevant information for transmission from an Emulator device IP address
 *
 * This structure contains the network interface configuration required for the emulator to send
 * packets, including interface name, IP address, MAC address, optional port and broadcast address
 * (defaulting to 255.255.255.255 if on interface that does not have an explicit broadcast, e.g. loopback).
 */
typedef struct IPAddress {
    char if_name[IFNAMSIZ]; ///< Network interface name (e.g., "eth0", "lo")
    uint32_t ip_address { 0 }; ///< IP address in network byte order
    uint32_t subnet_mask { 0 }; ///< Subnet mask in network byte order
    uint32_t broadcast_address { 0 }; ///< Broadcast address in network byte order
    uint8_t mac[6] { 0 }; ///< MAC address of the interface
    uint16_t port { DATA_SOURCE_UDP_PORT }; ///< UDP port number (defaults to DATA_SOURCE_UDP_PORT)
    uint8_t flags { 0 }; ///< Configuration flags indicating which fields are valid
    /**
     * @brief Configuration flags bitfield:
     * - 0x01 (IPADDRESS_HAS_ADDR): IP address is valid (and whole struct is valid)
     * - 0x02 (IPADDRESS_HAS_NETMASK): netmask is valid
     * - 0x04 (IPADDRESS_HAS_BROADCAST): broadcast address is valid
     * - 0x40 (IPADDRESS_IS_PTP): indicates point-to-point (1) or broadcast (0) interface
     * - 0x80 (IPADDRESS_HAS_MAC): MAC address is valid
     */
} IPAddress;

/**
 * python (use IPAddress object constructor)
 *
 * @brief Construct an IPAddress object from a string representation of the IP address.
 * @param ip_address The string representation of the IP address. Currently must be in format accepted by inet_addr().
 * @return An IPAddress instance
 */
IPAddress IPAddress_from_string(const std::string& ip_address);

/**
 * python (use IPAddress object __str__() method)
 *
 * `def __str__(self: hemu.IPAddress) -> str`
 *
 * @brief Convert an IPAddress object to its string representation.
 * @param ip_address The IPAddress object to convert
 * @return String representation of the IP address
 */
std::string IPAddress_to_string(const IPAddress& ip_address);

/**
 * @brief Get the broadcast address for a given IPAddress.
 * @param ip_address The IPAddress object containing the network configuration
 * @return The broadcast address as a 32-bit unsigned integer
 */
uint32_t get_broadcast_address(const IPAddress& ip_address);

/**
 * @brief Disable broadcast configuration warning messages.
 */
void DISABLE_BROADCAST_CONFIG_WARNING();

struct ControlMessageHeader {
    uint8_t cmd_code;
    uint8_t flags;
    uint16_t sequence;
    uint8_t status;
    uint8_t reserved;
};

// size of header: cmd_code, flags, sequence, status, reserved
#define CONTROL_MESSAGE_HEADER_LENGTH 6u

static_assert(sizeof(struct ControlMessageHeader) == CONTROL_MESSAGE_HEADER_LENGTH, "ControlMessageHeader must be 6 bytes");

struct ControlMessage {
    struct ControlMessageHeader* ctrl_hdr;
    struct AddressValuePair* addr_vals;
    uint16_t num_addresses; // addr_vals is of length num_addresses
};

} // namespace hololink::emulation

struct PTPConfig {
    uint32_t ctrl;
    uint32_t delay_asymetry;
    uint32_t dpll_cfg1;
    uint32_t dpll_cfg2;
    uint32_t delay_avg_factor;
    uint32_t sync_ts_0;
    uint32_t sync_stat;
    uint32_t ofm;
};

#endif /* EMULATION_NET_HPP */
