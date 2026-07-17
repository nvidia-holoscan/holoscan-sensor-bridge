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

#include "../common/coe_transmitter.hpp"
#include "coe_data_plane.hpp"
#include "data_plane.hpp"
#include "hsb_emulator.hpp"
#include "net.hpp"

namespace hololink::emulation {

uint16_t TRANSMITTER_METADATA_COUNT = 0;

struct COECtxt COE_TRANSMISSION_METADATA[MAX_DATA_PLANES];
COETransmitter COE_TRANSMITTER;

// Claim the next slot in the COECtxt static pool and hand its DataPlaneCtxt-aliased
// base back to the protected DataPlane constructor via a unique_ptr with a no-op
// deleter (static lifetime). Aborts via Error_Handler if the pool is exhausted.
static std::unique_ptr<DataPlaneCtxt, std::function<void(DataPlaneCtxt*)>>
make_coe_ctxt()
{
    if (TRANSMITTER_METADATA_COUNT >= MAX_DATA_PLANES) {
        Error_Handler(NULL);
    }
    COECtxt* coe_ctxt = &COE_TRANSMISSION_METADATA[TRANSMITTER_METADATA_COUNT++];
    return {
        &coe_ctxt->base.base,
        [](DataPlaneCtxt*) { /* statically allocated */ }
    };
}

COEDataPlane::COEDataPlane(HSBEmulator& hsb_emulator, const IPAddress& source_ip, uint8_t data_plane_id, uint8_t sensor_id)
    : DataPlane(hsb_emulator, source_ip, data_plane_id, sensor_id, make_coe_ctxt())
{
    // data_plane_ctxt_ now points at &coe_ctxt->base.base (the embedded
    // STM32DataPlaneCtxt's common DataPlaneCtxt base). DataPlane's ctor has already
    // set coe_ctxt->base.eth_handle / bootp_buffer / tx_config etc. via init().
    COECtxt* coe_ctxt = reinterpret_cast<COECtxt*>(data_plane_ctxt_.get());

    transmitter_ = (BaseTransmitter*)&COE_TRANSMITTER;
    ((COETransmitter*)transmitter_)->init_metadata(coe_ctxt, source_ip, sensor_id);
}

COEDataPlane::~COEDataPlane()
{
    // do nothing — static pool, no resources to free
}

} // namespace hololink::emulation
