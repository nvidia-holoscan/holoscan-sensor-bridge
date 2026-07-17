/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * TEMPLATE port — COE transport platform stubs. Compiled into emulationcoe.a so
 * the symbols sit alongside the common COE transmitter / data-plane code (which
 * is also in emulationcoe.a). Kept separate from HSBTemplate.cpp because that
 * file goes into emulation_host.a and including these symbols there would
 * create a circular dependency with emulationcoe.a (the place that defines
 * COETransmitter's vtable).
 *
 * Fill in the bodies below with whatever your target needs for COE transmit.
 */

#include "HSBTemplate.hpp"

#include <functional>
#include <memory>

namespace hololink::emulation {

// Single COETransmitter instance referenced by COEDataPlane::COEDataPlane below.
COETransmitter COE_TRANSMITTER;

// Heap-allocate a COECtxt and hand its DataPlaneCtxt-aliased base back to the
// protected DataPlane ctor with a deleter that downcasts and deletes the COECtxt.
static std::unique_ptr<DataPlaneCtxt, std::function<void(DataPlaneCtxt*)>>
make_coe_ctxt()
{
    COECtxt* coe_ctxt = new COECtxt();
    return {
        &coe_ctxt->base.base,
        [](DataPlaneCtxt* p) { delete reinterpret_cast<COECtxt*>(p); }
    };
}

COEDataPlane::COEDataPlane(HSBEmulator& hsb_emulator, const IPAddress& source_ip, uint8_t data_plane_id, uint8_t sensor_id)
    : DataPlane(hsb_emulator, source_ip, data_plane_id, sensor_id, make_coe_ctxt())
{
    COECtxt* coe_ctxt = reinterpret_cast<COECtxt*>(data_plane_ctxt_.get());
    transmitter_ = &COE_TRANSMITTER;
    ((COETransmitter*)transmitter_)->init_metadata(coe_ctxt, source_ip, sensor_id);
    // TODO: set up the COE transport for this data plane (open a socket / claim
    // a HAL transmit channel, configure the destination address, ...).
}

COEDataPlane::~COEDataPlane()
{
    // COECtxt itself is destroyed by data_plane_ctxt_'s deleter. Add any
    // transport-side teardown here (close sockets, free HAL resources, ...).
}

// TODO: push the prepared scatter/gather chain onto the wire — the COE wire
// format is IEEE 1722B AVTP frames. Build whatever the target's transport API
// expects (HAL DMA descriptor, raw-socket sendmsg, ...) and emit the frame.
int16_t send_coe_packet(COECtxt* /*coe_ctxt*/, ETH_BufferTypeDef* /*tx_buffers*/)
{
    return 0;
}

} // namespace hololink::emulation
