/**
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SRC_HOLOLINK_NATIVE_ARP_WRAPPER
#define SRC_HOLOLINK_NATIVE_ARP_WRAPPER

#include <arpa/inet.h>
#include <errno.h>
#include <net/if.h>
#include <net/if_arp.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>

#define NUM_OF(a) (sizeof(a) / sizeof(a[0]))

#define TRACE(fmt, ...) /* ignored */
#define DEBUG(fmt...) fprintf(stderr, "DEBUG -- " fmt)
#define ERROR(fmt...) fprintf(stderr, "ERROR -- " fmt)

namespace hololink::native {

class ArpWrapper {
public:
    /** socket_fd is an AF_INET socket */
    static int arp_set(int socket_fd, char const* eth_device, char const* ip, const char* mac_id)
    {
        TRACE("socket_fd=%d eth_device=%s ip=%s mac_id=%s.\n",
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
            ERROR("sscanf r=%d.\n", r);
            return EINVAL;
        }
        for (unsigned i = 0; i < NUM_OF(b); i++) {
            arpreq.arp_ha.sa_data[i] = b[i];
        }
        arpreq.arp_flags = ATF_COM;
        r = ioctl(socket_fd, SIOCSARP, &arpreq);
        if (r != 0) {
            ERROR("ioctl r=%d.\n", r);
        }
        return r;
    }
};

} // namespace hololink::native

#endif /* SRC_HOLOLINK_NATIVE_ARP_WRAPPER */
