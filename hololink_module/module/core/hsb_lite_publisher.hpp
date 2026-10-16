/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_HSB_LITE_PUBLISHER_HPP
#define HOLOLINK_MODULE_CORE_HSB_LITE_PUBLISHER_HPP

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/logging.hpp"
#include "hololink/module/publisher.hpp"

#include "hsb_ip_publisher.hpp"
#include "hsb_lite_default.hpp"
#include "mipi_dphy/clnx.hpp"

namespace hololink::module::module_core {

/* Canonical ChannelConfigurationInterfaceV1 impl for HSB-Lite carriers.
 * use_sensor re-stamps the per-sensor VP / SIF address fields through the
 * shared hsb_ip_sensor_metadata formula (two SIFs per sensor) while leaving
 * the data plane on the enumerated one — sensors are multiplexed onto a data
 * plane by VP, not 1:1 with data planes. */
class HsbLiteChannelConfigurationV1 : public HsbIpChannelConfigurationV1 {
public:
    // HSB-Lite carries 3 sensors (sensor 2 is the I2S audio interface)
    // mapped 1:1 onto their own data planes, two SIFs each. The single
    // source of truth for the valid sensor range, shared by use_sensor's
    // bounds check and is_sensor_valid().
    static constexpr int64_t TOTAL_SENSORS = 3;

    HsbLiteChannelConfigurationV1() = default;

    bool is_sensor_valid(int64_t sensor_number) const override
    {
        return sensor_number >= 0 && sensor_number < TOTAL_SENSORS;
    }

    void use_sensor(
        EnumerationMetadata& metadata, int64_t sensor_number) override
    {
        // data_plane == sensor_number, so selecting a sensor also
        // re-points hif_address at that sensor's data plane.
        if (!is_sensor_valid(sensor_number)) {
            throw std::runtime_error(
                "While selecting an HSB-Lite sensor: sensor_number "
                + std::to_string(sensor_number)
                + " is out of range (HSB-Lite supports 0.."
                + std::to_string(TOTAL_SENSORS - 1) + ")");
        }
        const int64_t data_plane = metadata.get<int64_t>("data_plane", int64_t { 0 });
        hsb_ip_sensor_metadata(metadata, sensor_number,
            /*data_plane=*/data_plane,
            /*sifs_per_sensor=*/2);
        // Diagnostic: confirms which use_sensor code path is loaded. The
        // fixed path keeps hif_address on the enumerated data plane (so
        // sensor 1 stays on data plane 0); the old data_plane==sensor_number
        // path would log hif_address=0x2010300 for sensor 1.
        HSB_LOG_DEBUG(
            "use_sensor sensor_number={} data_plane={} hif_address={:#x} "
            "vp_address={:#x} data_channel={}",
            sensor_number, data_plane,
            metadata.get<int64_t>("hif_address", int64_t { 0 }),
            metadata.get<int64_t>("vp_address", int64_t { 0 }),
            metadata.get<int64_t>("data_channel", int64_t { 0 }));
    }
};

/* Publisher for HSB-Lite carriers. Everything HSB-shaped comes from
 * HsbIpPublisher; this class adds only what makes a board an HSB-Lite
 * one — the CrossLink-NX D-PHY, the Renesas Bajoran TS1 clock
 * generator, the 3-sensor / two-SIF layout, and the HsbLiteInterfaceV1
 * supplement. A board that shares the HSB IP but not the carrier
 * inherits HsbIpPublisher directly instead of this. */
class HsbLitePublisher : public HsbIpPublisher {
protected:
    /* HSB-Lite is CPNX + CLNX and its D-PHY lives in the CrossLink, so the
     * shape stamped here and the layout construct_board_services publishes
     * are both the CLNX ones. The two must agree: stamping one design's
     * shape while constructing another's layout leaves the service
     * unconfigured, and every program() reports NOT_FOUND. */
    hololink_module_status_t update_board_metadata(
        EnumerationMetadata& metadata) override
    {
        stamp_clnx_mipi_dphy_metadata(metadata);
        return HOLOLINK_MODULE_OK;
    }

    void publish_channel_configuration() override
    {
        auto impl = std::make_shared<HsbLiteChannelConfigurationV1>();
        ServicePublisher<HsbIpChannelConfigurationV1>(shared_from_this())
            .publish("", impl);
    }

    /* Bajoran TS1, shared across the board's data planes. */
    bool construct_oscillator(
        const std::string& instance_id,
        const std::string& type_id) override
    {
        return construct_shared_oscillator<HsbLiteOscillatorV1>(
            instance_id, type_id);
    }

    /* The HSB-Lite board supplement and the CrossLink-NX D-PHY, published
     * on the tail hook so they follow the canonical services rather than
     * preempting them. */
    bool construct_board_services(
        const std::string& instance_id,
        const std::string& type_id) override
    {
        return construct_hsb_lite(instance_id, type_id)
            || construct_clnx_mipi_dphy(*this, instance_id, type_id);
    }

    virtual bool construct_hsb_lite(
        const std::string& instance_id,
        const std::string& type_id)
    {
        if (!Publisher::has_type_id<HsbLiteV1>(type_id)) {
            return false;
        }
        auto impl = std::make_shared<HsbLiteV1>();
        ServicePublisher<HsbLiteV1>(shared_from_this())
            .publish(instance_id, impl);
        return true;
    }
};

} // namespace hololink::module::module_core

#endif // HOLOLINK_MODULE_CORE_HSB_LITE_PUBLISHER_HPP
