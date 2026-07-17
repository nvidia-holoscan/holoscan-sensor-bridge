/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_COE_DATA_CHANNEL_HPP
#define HOLOLINK_MODULE_COE_DATA_CHANNEL_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "enumeration_metadata.hpp"
#include "module.hpp"
#include "service.hpp"
#include "status.h"

namespace hololink::module {

class DataChannelInterfaceV1;

/* CoE transport view of a per-channel DataChannelInterfaceV1. A
 * separate ConfigurableService — has-a, not is-a — whose impl holds a
 * shared_ptr to its DataChannelInterfaceV1 anchor and adds the
 * IEEE-1722 CoE transport surface (configure_coe / unconfigure).
 *
 * Cached under instance_id "serial=<serial_number>;data_channel=<n>"
 * (the same key as the anchor it wraps). The supplement's
 * construct_service for this type_id fetches the anchor through the
 * Publisher and passes it to the impl's constructor. The impl's
 * configure(metadata) drives the anchor's ensure_configured(metadata)
 * before building its own legacy backing — application code calls
 * CoeDataChannelInterfaceV1::get_service(metadata) without first
 * constructing the DataChannelInterfaceV1 itself. */
class CoeDataChannelInterfaceV1
    : public ConfigurableService<CoeDataChannelInterfaceV1> {
public:
    static constexpr const char* type_id = "coe_data_channel.v1";

    /* Per-channel instance_id derivation used by the metadata-form
     * get_service. Same shape as DataChannelInterfaceV1::locator_id
     * — both services key by (serial, data_channel). */
    static std::string locator_id(const EnumerationMetadata& metadata)
    {
        return "serial=" + metadata.get<std::string>("serial_number")
            + ";data_channel=" + std::to_string(metadata.get<int64_t>("data_channel"));
    }

    virtual ~CoeDataChannelInterfaceV1() = default;

    /* Select the on-device packetizer program for the given CSI pixel
     * format. Values match hololink::csi::PixelFormat. Must be called
     * before configure_coe() when the stream carries CSI image data. */
    virtual hololink_module_status_t set_packetizer_for_pixel_format(
        uint32_t pixel_format)
        = 0;

    /* Program the data plane for IEEE-1722 CoE delivery into NvFusa
     * capture buffers. Pair with unconfigure() before tearing the
     * channel down. */
    virtual hololink_module_status_t configure_coe(
        uint8_t channel, size_t frame_size, uint32_t pixel_width,
        bool vlan_enabled = false)
        = 0;

    /* Clear the operating state set up by configure_coe(). */
    virtual hololink_module_status_t unconfigure() = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_COE_DATA_CHANNEL_HPP
