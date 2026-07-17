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

#include <cstdint>
#include <sys/socket.h>

#include "../common/rocev2_transmitter.hpp"
#include "net.hpp"
#include "rocev2_data_plane.hpp"

namespace hololink::emulation {

// eth_socket_send() is declared in linux/net.hpp. DLTensor handling has moved up to
// DataPlane::send (Linux side), which performs the double-buffer copy and then calls
// the buffer-mode RoCEv2Transmitter::send.

int16_t send_rocev2_packet(RoCEv2Ctxt* rocev2_ctxt, ETH_BufferTypeDef* tx_buffers)
{
    // RoCEv2 rides on a UDP socket; the kernel rebuilds the L2 header from socket state,
    // so drop the Ethernet header bytes from the first chunk.
    return eth_socket_send(rocev2_ctxt->data_socket_fd,
        &rocev2_ctxt->dest_addr, sizeof(rocev2_ctxt->dest_addr),
        tx_buffers, /*skip_head_bytes=*/ETHER_HDR_LEN);
}

} // namespace hololink::emulation
