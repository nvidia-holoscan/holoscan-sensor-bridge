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

#ifndef EMULATION_LINUX_NET_HPP
#define EMULATION_LINUX_NET_HPP

#include <arpa/inet.h>
#include <net/ethernet.h>
#include <net/if.h>
#include <netinet/in.h>

#include <netinet/ip.h>
#include <netinet/udp.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

// this must include the net.hpp at top level — provides MTU_SIZE, htonll, ntohll,
// IB BTH/RETH + COE protocol structs/macros (target-invariant wire formats).
#include "../../net.hpp"
#include "utils.hpp"
#include <linux/if_packet.h>

#ifndef TX_BUFFER_SIZE
#define TX_BUFFER_SIZE MTU_SIZE
#endif

struct ETH_BufferTypeDef {
    uint8_t* buffer;
    struct ETH_BufferTypeDef* next;
    size_t len;
};

// Shared Linux wire emit for the COE/RoCEv2 transmitters and control_plane_reply.
// Walks the scatter/gather chain into an iovec and sendmsg's it over the provided BSD
// socket; `skip_head_bytes` drops the first N bytes of the first chunk (caller picks
// 0 for raw AF_PACKET, ETHER_HDR_LEN for UDP socket needing IP+UDP headers, or all
// three header sizes for UDP-payload-only). Defined in linux/net.cpp.
int16_t eth_socket_send(int fd, const void* dest_addr, socklen_t addr_len,
    ETH_BufferTypeDef* tx_buffers, size_t skip_head_bytes);

// Sum the .len fields across the scatter/gather chain. Implementation in src/common/net.cpp.
// TODO: this declaration is duplicated in STM32/net.hpp because the STM32 HAL's
// `typedef struct __ETH_BufferTypeDef { ... } ETH_BufferTypeDef;` makes the typedef-name
// and the struct tag-name disagree, which prevents a single `struct ETH_BufferTypeDef;`
// forward declaration in the public net.hpp from satisfying both platforms. Likely to
// move once we introduce a platform-neutral scatter/gather type.
uint16_t calculate_buffer_length(const ETH_BufferTypeDef* tx_buffer);

#define ETHER_ADDR_LEN ETH_ALEN

// IPHDR_SET_IHL_VERSION stays platform-specific: Linux's <netinet/ip.h> defines
// iphdr with separate `version:4` and `ihl:4` bitfields, so we write them as
// distinct fields. STM32's iphdr uses a combined `ihl_version` byte and needs
// a different macro body.
#define IPHDR_SET_IHL_VERSION(hdr, ihl_, version_) \
    (hdr)->version = (version_);                   \
    ((hdr)->ihl = (ihl_ / 4))

// Ethernet/IP/UDP offsets and payload accessors + IB BTH/RETH + COE protocol
// structs/macros all live in public emulation/net.hpp (target-invariant).

namespace hololink::emulation {
void mac_from_if(IPAddress& iface, std::string const& if_name);
}

#endif