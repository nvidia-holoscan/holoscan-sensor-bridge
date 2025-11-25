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
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifndef __linux__
#include <sys/sysctl.h>
// sockaddr_dl and link_ntoa (not really used)
#include <net/if_dl.h>
#endif
#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/in.h>
// ETHER_ADDR_LEN is defined in net/ethernet.h on linux and netinet/if_ether.h on BSD
#include <net/ethernet.h>
// ether_aton
#include <netinet/ether.h>
// must come after net/if.h
#include <ifaddrs.h>

#include "net.hpp"

#define MAC_HEX_DOT_LENGTH 17

namespace hololink::emulation {

void mac_from_if(IPAddress& iface, std::string const& if_name)
{
    char* buf = NULL;
    size_t len = 0;
    size_t if_name_len = if_name.length();
    len = if_name_len + strlen("/sys/class/net/") + strlen("/address") + 1; // +1 for null terminator
    buf = new char[len];

    snprintf(buf, len, "/sys/class/net/"
                       "%.*s"
                       "/address",
        (int)if_name_len, if_name.c_str());
    FILE* file = fopen(buf, "rb");
    if (!file) {
        std::string error_message = "failed to open file: " + std::string(buf);
        delete[] buf;
        throw std::runtime_error(error_message);
    }
    size_t read_size = fread(buf, 1, len, file);
    if (read_size < 3 * ETHER_ADDR_LEN) { // 2 bytes per octet + 1 ':' delimiter per ETHER_ADDR_LEN - 1 octets + 1 null terminator/LF minimum
        fclose(file);
        std::string error_message = "failed to read file: " + std::string(buf);
        delete[] buf;
        throw std::runtime_error(error_message);
    }
    fclose(file);
    buf[3 * ETHER_ADDR_LEN - 1] = '\0'; // null terminate the string
    struct ether_addr* ether_addr = ether_aton(buf);
    if (!ether_addr) {
        std::string error_message = "failed to parse MAC address from file: " + std::string(buf);
        delete[] buf;
        throw std::runtime_error(error_message);
    }
    memcpy(iface.mac.data(), ether_addr->ether_addr_octet, std::min((long unsigned int)ETHER_ADDR_LEN, iface.mac.size()));
    iface.flags |= IPADDRESS_HAS_MAC;
    delete[] buf;
}

// ipv4 in in_addr_t *
int comp_ifaddrs_addr(struct ifaddrs* ifa, void* ipv4)
{
    in_addr_t ipv4_ = *(in_addr_t*)ipv4;

    if (!ifa->ifa_addr) {
        return 1;
    }

    if (AF_INET == ifa->ifa_addr->sa_family) {
        struct sockaddr_in* ifa_in = (struct sockaddr_in*)ifa->ifa_addr;
        if (ipv4_ == ifa_in->sin_addr.s_addr) {
            return 0;
        }
        return 1;
    }
    return 1;
}

int comp_ifaddrs_iface(struct ifaddrs* ifa, void* iface)
{
    char const* iface_ = (char const*)iface;

    if (!strcmp(ifa->ifa_name, iface_)) {
        return 0;
    }
    return 1;
}

struct ifaddrs* filter_ifaddrs(struct ifaddrs* ifa,
    int (*comp)(struct ifaddrs*, void*),
    void* data)
{

    while (ifa && comp(ifa, data)) {
        ifa = ifa->ifa_next;
    }
    return ifa;
}

void ifaddrs_ipv4_addr(IPAddress& dest, const struct ifaddrs* ifa)
{
    if (!ifa->ifa_addr) {
        return;
    }
    dest.ip_address = ((struct sockaddr_in*)(ifa->ifa_addr))->sin_addr.s_addr;
    dest.flags |= IPADDRESS_HAS_ADDR;
}

void ifaddrs_name(IPAddress& dest, const struct ifaddrs* ifa)
{
    dest.if_name = ifa->ifa_name;
}

// returns 0 on success, else -1
void ifaddrs_ipv4_netmask(IPAddress& dest, const struct ifaddrs* ifa)
{
    if (!ifa->ifa_netmask) {
        return;
    }
    dest.subnet_mask = ((struct sockaddr_in*)(ifa->ifa_netmask))->sin_addr.s_addr;
    dest.flags |= IPADDRESS_HAS_NETMASK;
}

void ifaddrs_ipv4_broadaddr(IPAddress& dest, const struct ifaddrs* ifa)
{
    if (!ifa->ifa_broadaddr) {
        return;
    }
    // linux specific handling of PTP vs broadcast
    if (!(ifa->ifa_flags & IFF_BROADCAST)) {
        dest.flags |= IPADDRESS_IS_PTP;
    }

    dest.broadcast_address = ((struct sockaddr_in*)(ifa->ifa_broadaddr))->sin_addr.s_addr;
    dest.flags |= IPADDRESS_HAS_BROADCAST;
}

// success is determined by checking the appropriate *HAS* flags in IPAddress->flags
IPAddress IPAddress_from_string(const std::string& ip)
{
    IPAddress iface = {
        .port = hololink::DATA_SOURCE_UDP_PORT,
    };
    in_addr_t addr = inet_addr(ip.c_str());
    if (addr == INADDR_NONE) {
        return iface;
    }

    struct ifaddrs* ifaddrs = NULL;
    if (0 != getifaddrs(&ifaddrs)) {
        fprintf(stderr, "failed to getifaddrs()\n");
        return iface;
    }

    struct ifaddrs* ifa = filter_ifaddrs(ifaddrs, comp_ifaddrs_addr, &addr);
    if (!ifa) {
        fprintf(stderr, "failed to filter_ifaddrs\n");
        goto cleanup;
    }

    ifaddrs_name(iface, ifa);
    ifaddrs_ipv4_addr(iface, ifa);
    ifaddrs_ipv4_netmask(iface, ifa);
    ifaddrs_ipv4_broadaddr(iface, ifa);
    mac_from_if(iface, iface.if_name.c_str());

cleanup:
    freeifaddrs(ifaddrs);
    return iface;
}

std::string IPAddress_to_string(const IPAddress& ip_address)
{
    if (!(ip_address.flags & IPADDRESS_HAS_ADDR)) {
        throw std::runtime_error("invalid IP address. ipv4 address not found");
    }
    struct in_addr addr = {
        .s_addr = ip_address.ip_address,
    };
    return std::string(inet_ntoa(addr));
}

bool disable_broadcast_config_warning = false;
void DISABLE_BROADCAST_CONFIG_WARNING()
{
    disable_broadcast_config_warning = true;
}

void issue_broadcast_address_warning(const std::string& if_name, uint32_t broadcast_address)
{
    fprintf(stderr, "WARNING: interface %s may be misconfigured. "
                    "broadcast address %010x does not match expected format\n"
                    "if interface was manually configured, check that broadcast was set, "
                    "e.g. on Linux: "
                    "\tip addr add <ip address>/<subnet_bits> <interface_name> brd +\n"
                    "to disable this warning, call DISABLE_BROADCAST_CONFIG_WARNING()\n",
        if_name.c_str(), broadcast_address);
}

uint32_t get_broadcast_address(const IPAddress& ip_address)
{
    if (ip_address.flags & IPADDRESS_IS_PTP) {
        return INADDR_BROADCAST;
    }
    if (!(ip_address.flags & IPADDRESS_HAS_BROADCAST)) {
        if (!((ip_address.flags & IPADDRESS_HAS_NETMASK) && (ip_address.flags & IPADDRESS_HAS_ADDR))) {
            throw std::runtime_error("broadcast address cannot be determined: missing addr or netmask for interface " + ip_address.if_name);
        }
        return ip_address.ip_address | ~ip_address.subnet_mask;
    }
    in_addr_t broadcast_address = ip_address.broadcast_address;

    // check for potential misconfiguration
    if (!disable_broadcast_config_warning && (ip_address.flags & IPADDRESS_HAS_NETMASK)) {
        if (ip_address.flags & IPADDRESS_HAS_ADDR) {
            in_addr_t expected_broadcast_address = ip_address.ip_address | ~ip_address.subnet_mask;
            if (ip_address.broadcast_address != expected_broadcast_address) {
                issue_broadcast_address_warning(ip_address.if_name, ip_address.broadcast_address);
                // even though expected broadcast address is probably correct, it cannot be guaranteed so leave warning as is.
            }
        } else {
            // broadcast address should have 1s in last 0s of subnet mask. This is not strictly exact, but a good heuristic.
            in_addr_t expected_broadcast_address = 0;
            in_addr_t subnet_mask = ip_address.subnet_mask;
            uint64_t bit_mask = 1;
            while (bit_mask < subnet_mask) {
                if (subnet_mask & bit_mask) {
                    break;
                }
                expected_broadcast_address |= bit_mask;
                bit_mask <<= 1;
            }
            if ((broadcast_address & expected_broadcast_address) != expected_broadcast_address) {
                issue_broadcast_address_warning(ip_address.if_name, broadcast_address);
            }
        }
    }
    return broadcast_address;
}

} // namespace hololink::emulation
