/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CHANNEL_CONFIGURATION_HPP
#define HOLOLINK_MODULE_CHANNEL_CONFIGURATION_HPP

#include <cstdint>
#include <memory>
#include <string>

#include "enumeration_metadata.hpp"
#include "module.hpp"
#include "service.hpp"

namespace hololink::module {

/* Per-module helper for module-aware EnumerationMetadata edits the
 * application makes BEFORE handing the metadata to a per-channel
 * service constructor.
 *
 * `use_sensor` re-points the metadata at a specific sensor on the
 * board so the per-channel address fields (vp_mask, sif_address,
 * vp_address, hif_address, frame_end_event, …) line up with that
 * sensor's data plane. The supplement owns the board layout, so the
 * supplement owns this edit — applications go through
 * `Adapter::use_sensor(metadata, sensor_number)` instead of stamping
 * fields themselves.
 *
 * `use_mtu` records the MTU the application wants the RoCE data plane
 * to honour; the per-channel constructors read it when they compute
 * packet sizes.
 *
 * `use_multicast` records the multicast destination (group address +
 * port) the application wants the data plane to target; the per-channel
 * constructors copy it onto the legacy DataChannel, which programs the
 * FPGA to send there.
 *
 * Modules publish a single ChannelConfigurationInterfaceV1 instance
 * under instance_id "" (singleton, like EnumerationInterfaceV1). */
class ChannelConfigurationInterfaceV1
    : public Service<ChannelConfigurationInterfaceV1> {
public:
    static constexpr const char* type_id = "channel_configuration.v1";

    // Singleton: hides the inherited three-arg form, passes "" instance_id.
    static std::shared_ptr<ChannelConfigurationInterfaceV1> get_service(
        std::shared_ptr<Module> module, bool allow_null = false)
    {
        return Service<ChannelConfigurationInterfaceV1>::get_service(
            std::move(module), "", allow_null);
    }

    virtual ~ChannelConfigurationInterfaceV1() = default;

    /* Re-stamp `metadata` for the given sensor on the board. The
     * supplement edits whichever fields its per-board layout makes
     * sensor-dependent — typically `data_plane`, `sensor`,
     * `sensor_number`, the SIF / VP / HIF address fields, and the
     * frame-end event. */
    virtual void use_sensor(
        EnumerationMetadata& metadata, int64_t sensor_number)
        = 0;

    /* True when sensor_number names a sensor this board exposes. Lets
     * callers validate an (untrusted) sensor_number before use_sensor,
     * which throws on an out-of-range value. */
    virtual bool is_sensor_valid(int64_t sensor_number) const = 0;

    /* Record the requested MTU on `metadata`. Per-channel services
     * (RoceDataChannel, …) read this when sizing packets. */
    virtual void use_mtu(EnumerationMetadata& metadata, uint32_t mtu) = 0;

    /* Record a multicast destination (group address + port) on
     * `metadata`. Per-channel services (RoceDataChannel, …) read this
     * when they program the data plane so frames are sent to the
     * multicast group rather than a unicast peer. */
    virtual void use_multicast(
        EnumerationMetadata& metadata, std::string address, uint16_t port)
        = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_CHANNEL_CONFIGURATION_HPP
