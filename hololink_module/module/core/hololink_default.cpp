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
    backing_ = std::make_shared<LegacyHololinkAccess>(
        metadata.get<std::string>("peer_ip"),
        static_cast<uint32_t>(metadata.get<int64_t>("control_port")),
        serial_number_,
        sequence_number_checking_);

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

} // namespace hololink::module::module_core
