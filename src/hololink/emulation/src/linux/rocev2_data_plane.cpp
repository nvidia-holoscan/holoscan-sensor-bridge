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

#include "../common/rocev2_transmitter.hpp"
#include "data_plane.hpp"
#include "rocev2_data_plane.hpp"

namespace hololink::emulation {

RoCEv2Transmitter ROCEV2_TRANSMITTER;

// See linux/coe_data_plane.cpp::make_coe_ctxt for the equivalent pattern + rationale.
static std::unique_ptr<DataPlaneCtxt, std::function<void(DataPlaneCtxt*)>>
make_rocev2_ctxt()
{
    RoCEv2Ctxt* rocev2_ctxt = new RoCEv2Ctxt();
    return {
        &rocev2_ctxt->base.base,
        [](DataPlaneCtxt* p) { delete reinterpret_cast<RoCEv2Ctxt*>(p); }
    };
}

RoCEv2DataPlane::RoCEv2DataPlane(HSBEmulator& hsb_emulator, const IPAddress& source_ip, uint8_t data_plane_id, uint8_t sensor_id)
    : DataPlane(hsb_emulator, source_ip, data_plane_id, sensor_id, make_rocev2_ctxt())
{
    RoCEv2Ctxt* rocev2_ctxt = reinterpret_cast<RoCEv2Ctxt*>(data_plane_ctxt_.get());

    transmitter_ = &ROCEV2_TRANSMITTER;
    ((RoCEv2Transmitter*)transmitter_)->init_metadata(rocev2_ctxt, source_ip);

    int data_socket_fd = socket(AF_INET, SOCK_RAW, IPPROTO_RAW);
    if (data_socket_fd < 0) {
        fprintf(stderr, "Failed to create socket: %d - %s\n", errno, strerror(errno));
        throw std::runtime_error("Failed to create socket");
    }
    int on = 1;
    if (setsockopt(data_socket_fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)) < 0) {
        close(data_socket_fd);
        data_socket_fd = -1;
        fprintf(stderr, "Failed to set socket reused address: %d - %s\n", errno, strerror(errno));
        throw std::runtime_error("Failed to set socket reuse");
    }
    rocev2_ctxt->data_socket_fd = data_socket_fd;
}

RoCEv2DataPlane::~RoCEv2DataPlane()
{
    // RoCEv2Ctxt destroyed by data_plane_ctxt_'s deleter; close the socket here.
    close(reinterpret_cast<RoCEv2Ctxt*>(data_plane_ctxt_.get())->data_socket_fd);
}

} // namespace hololink::emulation
