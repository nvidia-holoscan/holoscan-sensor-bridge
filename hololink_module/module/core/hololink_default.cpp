/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hololink_default.hpp"

#include <stdexcept>

#include "data_channel_default.hpp"

namespace hololink::module::module_core {

void HololinkV1::configure(const EnumerationMetadata& metadata)
{
    // configure runs exactly once per instance — the framework's
    // ConfigurableService<T>::get_service wraps this call in
    // std::call_once on the instance's configure_once_ latch, so the
    // body doesn't need its own guard.
    serial_number_ = metadata.get<std::string>("serial_number");
    enumeration_metadata_ = metadata;
    // Virtual factory: a transport-specific HololinkV1 subclass
    // substitutes its own backing here (see create_backing()).
    backing_ = create_backing(metadata);

    // Construct the default per-board DataChannel from the same
    // metadata blob so its per-board fields match this Hololink's by
    // construction. Only meaningful when metadata carries data_plane;
    // a Hololink configured for control-plane work alone leaves
    // default_data_channel_ null.
    if (metadata.contains("data_plane")) {
        default_data_channel_
            = std::make_shared<DataChannelV1>(shared_from_this(), metadata);
    }
}

std::shared_ptr<LegacyHololinkAccess> HololinkV1::create_backing(
    const EnumerationMetadata& metadata)
{
    // "interface" is the NIC the board's bootp reply arrived on
    // (deserialize_ancdata records it from IP_PKTINFO). Handing it to the
    // backing Hololink pins control-plane egress to that NIC, so two boards
    // on overlapping data-plane subnets stay individually reachable without
    // host route configuration.
    return std::make_shared<LegacyHololinkAccess>(
        metadata.get<std::string>("peer_ip"),
        static_cast<uint32_t>(metadata.get<int64_t>("control_port")),
        serial_number_,
        sequence_number_checking_,
        // These three restate the constructor's defaults, which this call
        // relied on implicitly before; they are spelled out only to reach
        // the trailing peer_interface.
        false /*skip_sequence_initialization*/,
        true /*ptp_enable*/,
        true /*block_enable*/,
        // Optional for the same reason as in from_enumeration_metadata: the
        // bootp path always sets it, hand-built metadata need not.
        metadata.get<std::string>("interface", std::string {}));
}

} // namespace hololink::module::module_core
