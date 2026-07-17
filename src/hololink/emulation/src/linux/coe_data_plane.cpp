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

#include <stdexcept>

#include "coe_data_plane.hpp"
#include "coe_transmitter.hpp"
#include "data_plane.hpp"
#include "net.hpp"
namespace hololink::emulation {

COETransmitter COE_TRANSMITTER;

// Allocate the COECtxt on the heap and hand its DataPlaneCtxt-aliased base back to
// the protected DataPlane constructor inside a unique_ptr whose deleter knows to
// downcast and delete the COECtxt. Standard-layout first-member chain
// (COECtxt -> LinuxDataPlaneCtxt -> DataPlaneCtxt) guarantees the pointer is the same
// address as the COECtxt itself.
static std::unique_ptr<DataPlaneCtxt, std::function<void(DataPlaneCtxt*)>>
make_coe_ctxt()
{
    COECtxt* coe_ctxt = new COECtxt(); // value-init: trivial fields zero; std::mutex default-constructed
    return {
        &coe_ctxt->base.base,
        [](DataPlaneCtxt* p) { delete reinterpret_cast<COECtxt*>(p); }
    };
}

COEDataPlane::COEDataPlane(HSBEmulator& hsb_emulator, const IPAddress& source_ip, uint8_t data_plane_id, uint8_t sensor_id)
    : DataPlane(hsb_emulator, source_ip, data_plane_id, sensor_id, make_coe_ctxt())
{
    // data_plane_ctxt_ now points at &coe_ctxt->base.base — same memory as the COECtxt.
    COECtxt* coe_ctxt = reinterpret_cast<COECtxt*>(data_plane_ctxt_.get());

    transmitter_ = &COE_TRANSMITTER;
    ((COETransmitter*)transmitter_)->init_metadata(coe_ctxt, source_ip, sensor_id);

    int data_socket_fd = socket(AF_PACKET, SOCK_RAW, htons(ETHERTYPE_AVTP));
    if (data_socket_fd < 0) {
        fprintf(stderr, "Failed to create socket: %d - %s\n", errno, strerror(errno));
        throw std::runtime_error("Failed to create socket");
    }

    // minimal sockaddr_ll initialization for binding to receive on interface
    // TODO: check if the bind is even necessary
    struct sockaddr_ll sockaddr;
    sockaddr.sll_family = AF_PACKET;
    sockaddr.sll_protocol = htons(ETHERTYPE_AVTP);
    sockaddr.sll_ifindex = if_nametoindex(ip_address_.if_name);

    if (bind(data_socket_fd, (struct sockaddr*)&sockaddr, sizeof(sockaddr)) == -1) {
        throw std::runtime_error("Failed to bind socket to interface " + std::string(source_ip.if_name));
    }
    coe_ctxt->data_socket_fd = data_socket_fd;
    coe_ctxt->dest_addr.sll_family = sockaddr.sll_family;
    coe_ctxt->dest_addr.sll_ifindex = sockaddr.sll_ifindex;
    coe_ctxt->dest_addr.sll_protocol = sockaddr.sll_protocol;
    coe_ctxt->dest_addr.sll_halen = 6;
}

COEDataPlane::~COEDataPlane()
{
    // The COECtxt itself (with its embedded ~LinuxDataPlaneCtxt that frees double_buffer)
    // is destroyed by data_plane_ctxt_'s deleter. Just close the socket here.
    close(reinterpret_cast<COECtxt*>(data_plane_ctxt_.get())->data_socket_fd);
}

} // namespace hololink::emulation
