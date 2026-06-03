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

#ifndef STM32_NET_HPP
#define STM32_NET_HPP

#include "../../net.hpp"
#include "stm32f7xx_hal.h"

#ifndef RX_BUFFER_SIZE
#define RX_BUFFER_SIZE ETH_RX_BUF_SIZE
#endif

#ifndef TX_BUFFER_SIZE
#define TX_BUFFER_SIZE ETH_TX_BUF_SIZE
#endif

// this is the size of the ethernet frame, but not including the CRC or vlan (not supported yet)
#define MTU_SIZE 1514u

#define ETH_ALEN 6u /* Ethernet address length */
#define ETHER_ADDR_LEN ETH_ALEN
#define ETHER_TYPE_LEN 2u
#define ETHER_BROADCAST_ADDR               \
    {                                      \
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF \
    }

#define ETHER_HDR_LEN (ETHER_ADDR_LEN * 2 + ETHER_TYPE_LEN)
#define ETHER_HDR_OFFSET 0u
#define ETHER_PAYLOAD_OFFSET (ETHER_HDR_OFFSET + ETHER_HDR_LEN)
#define NET_GET_ETHER_HDR(hal_buffer_td) ((struct ether_header*)((hal_buffer_td)->buffer))
// offset by the ether header length. TODO: handle VLANs
#define NET_GET_ETHER_PAYLOAD(hdr) ((uint8_t*)(hdr) + ETHER_HDR_LEN)

/******************************* Ethernet protocol ID's *****************************/
// for important list: https://datatracker.ietf.org/doc/html/rfc5342
/* IPv4 */
#define ETHERTYPE_IP 0x0800U
/* Address resolution */
#define ETHERTYPE_ARP 0x0806U
/* IPv6 */
#define ETHERTYPE_IPv6 0x86DDu
/* AVTP IEEE 1722 */
#define ETHERTYPE_AVTP 0x22F0u
/* MACsec */
#define ETHERTYPE_MACSEC 0x88E5u

struct ether_header {
    uint8_t ether_dhost[ETH_ALEN];
    uint8_t ether_shost[ETH_ALEN];
    uint16_t ether_type;
};

// This only does ARP IPV4

#define ARP_HDR_LEN 8u
#define ARP_HDR_OFFSET ETHER_PAYLOAD_OFFSET
#define ARP_PAYLOAD_OFFSET (ARP_HDR_OFFSET + ARP_HDR_LEN)

// for full lists of each ARP enumeration type: https://www.iana.org/assignments/arp-parameters/arp-parameters.xhtml
#define ARPOP_REQUEST 1 /* ARP request.  */
#define ARPOP_REPLY 2 /* ARP reply.  */
#define ARPOP_NAK 10 /* (ATM)ARP NAK.  */

#define ARPHRD_ETHER 1 /* Ethernet 10/100Mbps.  */
#define ARPHRD_IEEE802 6 /* IEEE 802.2 Ethernet/TR/TB.  */
#define ARPHRD_INFINIBAND 32 /* InfiniBand.  */

struct arphdr {
    uint16_t ar_hrd; /* Format of hardware address.  */
    uint16_t ar_pro; /* Format of protocol address.  */
    uint8_t ar_hln; /* Length of hardware address.  */
    uint8_t ar_pln; /* Length of protocol address.  */
    uint16_t ar_op; /* ARP opcode (command).  */
    /* what the subsequent buffer looks like
    uint8_t ar_sha[arphdr.ar_hln];	// Sender hardware address.
    uint8_t ar_spa[arphdr.ar_pln];	// Sender protocol address.
    uint8_t ar_tha[arphdr.ar_hln];	// Target hardware address.
    uint8_t ar_tpa[arphdr.ar_pln];	// Target protocol address.
    */
};

#define IP_HDR_LEN 20u
#define IP_HDR_OFFSET ETHER_PAYLOAD_OFFSET
#define IP_PAYLOAD_OFFSET (IP_HDR_OFFSET + IP_HDR_LEN)
#define IP_ADDR_LEN 4u
#define NET_GET_IP_HDR(hal_buffer_td) ((struct iphdr*)((hal_buffer_td)->buffer + ETHER_PAYLOAD_OFFSET))
#define NET_GET_IP_PAYLOAD(hdr) ((uint8_t*)(hdr) + IP_HDR_LEN)

#define IPPROTO_IP 0u
#define IPPROTO_TCP 6u
#define IPPROTO_ICMP 1u
#define IPPROTO_UDP 17u
#define IPVERSION 4u
#define INADDR_BROADCAST ((in_addr_t)0xffffffff)
#define INADDR_ANY ((in_addr_t)0x00000000)
#define INADDR_LOOPBACK ((in_addr_t)0x7f000001)
#define INADDR_NONE ((in_addr_t)0xffffffff)
#define IP_DF 0x4000u /* don't fragment flag */
#define IPDEFTTL 64u /*  default ttl */

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

#define ICMP_HDR_LEN 8u
#define ICMP_HDR_OFFSET IP_PAYLOAD_OFFSET
#define ICMP_PAYLOAD_OFFSET (ICMP_HDR_OFFSET + ICMP_HDR_LEN)

#define ICMP_ECHOREPLY 0 /* Echo Reply			*/
#define ICMP_DEST_UNREACH 3 /* Destination Unreachable	*/
#define ICMP_SOURCE_QUENCH 4 /* Source Quench		*/
#define ICMP_REDIRECT 5 /* Redirect (change route)	*/
#define ICMP_ECHO 8 /* Echo Request			*/
#define ICMP_TIME_EXCEEDED 11 /* Time Exceeded		*/
#define ICMP_PARAMETERPROB 12 /* Parameter Problem		*/
#define ICMP_TIMESTAMP 13 /* Timestamp Request		*/
#define ICMP_TIMESTAMPREPLY 14 /* Timestamp Reply		*/

// copy of linux struct for ICMP header for API compatibility
struct icmphdr {
    uint8_t type;
    uint8_t code;
    uint16_t checksum;
    union {
        struct {
            uint16_t id;
            uint16_t sequence;
        } echo;
        uint32_t gateway;
    } un;
};

#define UDP_HDR_LEN 8u
#define UDP_HDR_OFFSET IP_PAYLOAD_OFFSET
#define UDP_PAYLOAD_OFFSET (UDP_HDR_OFFSET + UDP_HDR_LEN)
#define NET_GET_UDP_HDR(hal_buffer_td) ((struct udphdr*)((hal_buffer_td)->buffer + IP_PAYLOAD_OFFSET))
#define NET_GET_UDP_PAYLOAD(hdr) ((uint8_t*)(hdr) + UDP_HDR_LEN)

struct udphdr {
    uint16_t source;
    uint16_t dest;
    uint16_t len;
    uint16_t check;
};

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

#define RET_HDR_LEN 16u
#define RET_HDR_OFFSET BT_PAYLOAD_OFFSET
#define RET_PAYLOAD_OFFSET (RET_HDR_OFFSET + RET_HDR_LEN)
#define NET_GET_RET_PAYLOAD(hdr) ((uint8_t*)(hdr) + RET_HDR_LEN)

// from Infiniband specification
struct rethdr {
    uint64_t va; // virtual address where data is stored on receiver side
    uint32_t r_key; // rkey
    uint32_t dmalen; // dma length of the payload. Does not include headers or iCRC
};

typedef uint32_t in_addr_t;

// network/host conversion functions to match corresponding posix functions
static inline uint16_t htons(uint16_t value)
{
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    return value;
#else
    return (value << 8) | (value >> 8);
#endif
}

static inline uint16_t ntohs(uint16_t value)
{
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    return value;
#else
    return (value << 8) | (value >> 8);
#endif
}

static inline uint32_t htonl(uint32_t value)
{
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    return value;
#else
    return (value << 24) | ((value << 8) & 0x00FF0000) | ((value >> 8) & 0x0000FF00) | (value >> 24);
#endif
}

static inline uint32_t ntohl(uint32_t value)
{
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    return value;
#else
    return (value << 24) | ((value << 8) & 0x00FF0000) | ((value >> 8) & 0x0000FF00) | (value >> 24);
#endif
}

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

typedef struct in_addr {
    in_addr_t s_addr;
} in_addr;

// basic implementation of posix inet_ntoa to convert an in_addr_t to a string
char* inet_ntoa(struct in_addr addr);

// basic implementation of posix inet_aton to parse an IP address string into a in_addr_t
int inet_aton(const char* cp, in_addr* inp);

// set a custom mac address for the system.
// The default generated mac address is for development use only
void set_mac_address(const uint8_t* mac_address);

// generate a mac address for the system. NOT for production use
void generate_mac_address(void);

// get the system mac address
const uint8_t* get_mac_address(void);

// set the IP address for the system. Note only one interface is effectively supported.
void net_set_ip_address(in_addr_t ip_address);

// release a network buffer back to the pool
void eth_release(ETH_BufferTypeDef* buffer);

// returns 0 on success, < 0 on error. returns > 0 are reserved for future use. Valid buffer only if buffer returned is not NULL.
int eth_receive(ETH_BufferTypeDef** buffer_rtn);

// initialize the network module
int net_init(ETH_HandleTypeDef* heth);

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

// configure the PTP module
int ptp_configure(struct PTPConfig* ptp_config);

#endif