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
#include "hsb_emulator.hpp"
#include "net.hpp"
#include "rocev2_data_plane.hpp"

namespace hololink::emulation {

uint16_t TRANSMITTER_METADATA_COUNT = 0;

struct RoCEv2Ctxt ROCEV2_TRANSMISSION_METADATA[MAX_DATA_PLANES];
RoCEv2Transmitter ROCEV2_TRANSMITTER;

static std::unique_ptr<DataPlaneCtxt, std::function<void(DataPlaneCtxt*)>>
make_rocev2_ctxt()
{
    if (TRANSMITTER_METADATA_COUNT >= MAX_DATA_PLANES) {
        Error_Handler(NULL);
    }
    RoCEv2Ctxt* rocev2_ctxt = &ROCEV2_TRANSMISSION_METADATA[TRANSMITTER_METADATA_COUNT++];
    return {
        &rocev2_ctxt->base.base,
        [](DataPlaneCtxt*) { /* statically allocated */ }
    };
}

RoCEv2DataPlane::RoCEv2DataPlane(HSBEmulator& hsb_emulator, const IPAddress& source_ip, uint8_t data_plane_id, uint8_t sensor_id)
    : DataPlane(hsb_emulator, source_ip, data_plane_id, sensor_id, make_rocev2_ctxt())
{
    RoCEv2Ctxt* rocev2_ctxt = reinterpret_cast<RoCEv2Ctxt*>(data_plane_ctxt_.get());

    transmitter_ = (BaseTransmitter*)&ROCEV2_TRANSMITTER;
    ((RoCEv2Transmitter*)transmitter_)->init_metadata(rocev2_ctxt, source_ip);
}

RoCEv2DataPlane::~RoCEv2DataPlane()
{
    // do nothing — static pool, no resources to free
}

} // namespace hololink::emulation
