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
 */

#include "arp_wrapper.hpp"
#include "logging_internal.hpp"

#include <arpa/inet.h>
#include <errno.h>
#include <net/if.h>
#include <net/if_arp.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>

#define NUM_OF(a) (sizeof(a) / sizeof(a[0]))

namespace hololink::core {

int ArpWrapper::arp_set(int socket_fd, char const* eth_device, char const* ip, const char* mac_id)
{
    HSB_LOG_TRACE("socket_fd={} eth_device={} ip={} mac_id={}",
        socket_fd, eth_device, ip, mac_id);
    struct arpreq arpreq = {}; // C++ fills this with 0s
    struct sockaddr_in* ip_address = (struct sockaddr_in*)&(arpreq.arp_pa);
    ip_address->sin_family = AF_INET;
    ip_address->sin_addr.s_addr = inet_addr(ip);
    arpreq.arp_ha.sa_family = AF_LOCAL;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-truncation"
    strncpy(arpreq.arp_dev, eth_device, IFNAMSIZ);
#pragma GCC diagnostic pop
    unsigned b[6];
    int r = sscanf(mac_id, "%x:%x:%x:%x:%x:%x", &b[0], &b[1], &b[2], &b[3], &b[4], &b[5]);
    if (r != NUM_OF(b)) {
        HSB_LOG_ERROR("sscanf r={}", r);
        return EINVAL;
    }
    for (unsigned i = 0; i < NUM_OF(b); i++) {
        arpreq.arp_ha.sa_data[i] = b[i];
    }
    arpreq.arp_flags = ATF_COM;
    r = ioctl(socket_fd, SIOCSARP, &arpreq);
    if (r != 0) {
        if (r == 1) {
            HSB_LOG_WARN("SIOCSARP operation not permitted, rediscovering HSB after a "
                         "reset may take a while. To avoid this warning, run as root");
        } else {
            HSB_LOG_ERROR("SIOCSARP operation failed: r={}, e={}", r, errno);
            return EINVAL;
        }
    }
    return r;
}

} // namespace hololink::core
