/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_DATA_CHANNEL_HPP
#define HOLOLINK_MODULE_DATA_CHANNEL_HPP

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "enumeration_metadata.hpp"
#include "module.hpp"
#include "service.hpp"

namespace hololink::module {

class HololinkInterfaceV1;

/* Per-channel anchor — transport-agnostic. Identifies the channel by
 * "serial=<serial_number>;data_channel=<n>" and caches the two values
 * every per-channel service in the system reads off: the channel's
 * enumeration metadata, and the parent per-board HololinkInterfaceV1.
 *
 * The `data_channel` index is per-sensor — distinct from `data_plane`,
 * which is the bootp-side per-data-plane index. On HSB-Lite (1:1
 * sensor:data_plane) the two coincide; on N-sensors-per-data-plane
 * boards (Leopard VB1940-AIO) the same data_plane carries multiple
 * sensors and only `data_channel` separates them.
 *
 * Primary EnumerationMetadata cache. This is the per-channel
 * EnumerationMetadata snapshot for the V1 service surface. Every
 * per-channel V1 service that needs configuration data from the
 * metadata holds a shared_ptr<DataChannelInterfaceV1> and reads it
 * back through enumeration_metadata() rather than caching its own
 * copy. Per-board services use HololinkInterfaceV1::enumeration_metadata()
 * the same way.
 *
 * A transport-specific service (e.g. RoceDataChannelInterfaceV1) is a
 * separate ConfigurableService — has-a, not is-a — that holds a
 * shared_ptr to this anchor and adds its own surface (attach_receiver,
 * etc.). A channel with no transport-specific service is still a valid
 * DataChannelInterfaceV1: the anchor stands on its own.
 *
 * Construct order. Application code drives anchor materialization
 * either directly (DataChannelInterfaceV1::get_service(metadata)) or
 * transitively (e.g. RoceDataChannelInterfaceV1::get_service(metadata)
 * configures its anchor first). Sibling per-channel services do a
 * cache-only lookup of this anchor and throw if it isn't configured. */
class DataChannelInterfaceV1
    : public ConfigurableService<DataChannelInterfaceV1> {
public:
    static constexpr const char* type_id = "data_channel.v1";

    /* Per-channel instance_id derivation used by the metadata-form
     * get_service. Throws if the metadata is missing 'serial_number'
     * or 'data_channel'. */
    static std::string locator_id(const EnumerationMetadata& metadata)
    {
        return "serial=" + metadata.get<std::string>("serial_number")
            + ";data_channel=" + std::to_string(metadata.get<int64_t>("data_channel"));
    }

    virtual ~DataChannelInterfaceV1() = default;

    /* The per-channel enumeration metadata the impl was configured
     * with. Sibling per-channel services read every metadata field
     * they need off this snapshot. */
    virtual const EnumerationMetadata& enumeration_metadata() const = 0;

    /* The per-board HololinkInterfaceV1 this channel belongs to. The
     * impl resolves it via HololinkInterfaceV1::get_service(module,
     * metadata) at configure time and caches the shared_ptr. */
    virtual std::shared_ptr<HololinkInterfaceV1> hololink() const = 0;

    /* Channel-level entry point for HololinkInterfaceV1::device_lost, so the
     * reconnection path can invalidate a lost device through a data channel
     * it already holds rather than resolving the Hololink. */
    virtual hololink_module_status_t device_lost() = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_DATA_CHANNEL_HPP
