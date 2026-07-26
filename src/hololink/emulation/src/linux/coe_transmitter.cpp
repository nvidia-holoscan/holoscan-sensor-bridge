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

#include <cstdint>

#include "coe_data_plane.hpp"
#include "coe_transmitter.hpp"
#include "net.hpp"

namespace hololink::emulation {

// eth_socket_send() is declared in linux/net.hpp. DLTensor handling has moved up to
// DataPlane::send (Linux side), which performs the double-buffer copy and then calls
// the buffer-mode COETransmitter::send.

int16_t send_coe_packet(COECtxt* coe_ctxt, ETH_BufferTypeDef* tx_buffers)
{
    // COE rides on a raw AF_PACKET socket; the kernel does NOT rebuild the Ethernet
    // header, so we send the whole frame (skip_head_bytes = 0).
    return eth_socket_send(coe_ctxt->data_socket_fd,
        &coe_ctxt->dest_addr, sizeof(coe_ctxt->dest_addr),
        tx_buffers, /*skip_head_bytes=*/0);
}

} // namespace hololink::emulation
