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

/**
 * @brief Configures a data channel for use with COE encapsulation.
 *
 * Use the `configure_coe` method to configure this data channel (as
 * described in the enumeration metadata) for P1722B encapsulation.
 */
class CoeDataChannelInterfaceV1
    : public ConfigurableService<CoeDataChannelInterfaceV1> {
public:
    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    static constexpr const char* type_id = "coe_data_channel.v1";
    static constexpr const char* release_version = "2.7.0";

    /* Per-channel instance_id derivation used by the metadata-form
     * get_service. Throws if the metadata is missing 'serial_number'
     * or 'data_channel'. */
    static std::string locator_id(const EnumerationMetadata& metadata)
    {
        return "serial=" + metadata.get<std::string>("serial_number")
            + ";data_channel=" + std::to_string(metadata.get<int64_t>("data_channel"));
    }

    virtual ~CoeDataChannelInterfaceV1() = default;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Configure the on-device packetizer program based on
     *      a CSI data format.
     *
     * This service exists to configure HSB IP to swizzle image data
     * for interpretation by hardware ISP.
     *
     * @param pixel_format See hololink::module::csi::PixelFormat
     * @return HOLOLINK_MODULE_OK or an error status
     */
    virtual hololink_module_status_t set_packetizer_for_pixel_format(
        uint32_t pixel_format)
        = 0;

    /**
     * @brief Configure the data channel for COE (P1722B) framing.
     *
     * @param channel CoE channel to include in egress packets.
     * @param frame_size Size of a complete frame, in bytes.
     * @param pixel_width Bytes per line of the transmitted image.
     * @param vlan_enabled Tag the transmitted packets for 802.1Q VLAN.
     * @return HOLOLINK_MODULE_OK or an error status.
     */
    virtual hololink_module_status_t configure_coe(
        uint8_t channel, size_t frame_size, uint32_t pixel_width,
        bool vlan_enabled = false)
        = 0;

    /**
     * @brief Stop transmitting, clearing the state configure_coe() set up.
     *
     * @return HOLOLINK_MODULE_OK or an error status.
     */
    virtual hololink_module_status_t unconfigure() = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_COE_DATA_CHANNEL_HPP
