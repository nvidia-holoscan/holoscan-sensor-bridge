/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * TEMPLATE port — RoCEv2 transport platform stubs. Companion to HSBTemplate_coe.cpp;
 * see that file's header for why this content is split off from HSBTemplate.cpp.
 *
 * Fill in the bodies below with whatever your target needs for RoCEv2 transmit.
 */

#include "HSBTemplate.hpp"

#include <functional>
#include <memory>

namespace hololink::emulation {

// Single RoCEv2Transmitter instance referenced by RoCEv2DataPlane::RoCEv2DataPlane.
RoCEv2Transmitter ROCEV2_TRANSMITTER;

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
    // TODO: set up the RoCEv2 transport (UDP socket on Linux; HAL transmit
    // channel on bare metal).
}

RoCEv2DataPlane::~RoCEv2DataPlane()
{
    // RoCEv2Ctxt destroyed by data_plane_ctxt_'s deleter; tear down transport state.
}

// TODO: push the prepared scatter/gather chain onto the wire — the RoCEv2 wire
// format is UDP-encapsulated InfiniBand BTH/RETH.
int16_t send_rocev2_packet(RoCEv2Ctxt* /*rocev2_ctxt*/, ETH_BufferTypeDef* /*tx_buffers*/)
{
    return 0;
}

} // namespace hololink::emulation
