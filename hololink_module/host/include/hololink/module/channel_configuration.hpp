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

/**
 * @brief A Module provides one of these to update EnumerationMetadata
 *      based on requests from the application layer.
 *
 * This service provides device-specific implementation for various
 * configurations that are requested by the application layer.  Applications
 * usually call methods on Adapter to modify EnumerationMetadata (e.g.
 * Adapter::use_sensor); many of these work by loading up this
 * device-specific object and calling the appropriate method here.
 * This allows the module to set up the values needed by other module services
 * which will be instantiated later.  The framework puts no restrictions
 * on the name/value pairs that can be stored in EnumerationMetadata,
 * Module instances are free to coordinate using this as they wish.
 *
 * For example, an application can change the sensor number a specific
 * data path transports:
 *
 * @code
 *      adapter = hololink_module.Adapter.get_adapter()
 *      discovery_timeout_s = 30
 *      enumeration_metadata = adapter.wait_for_channel(hololink="192.168.0.2", discovery_timeout_s)
 *      # By default, this first channel publishes data from sensor 0.
 *      # Let's change it to publish data from sensor 1.
 *      sensor_number = 1
 *      adapter.use_sensor(enumeration_metadata, sensor_number)
 *      ...
 * @endcode
 *
 * The call to adapter.use_sensor results in loading this service
 * from the module and calling use_sensor on it.
 */
class ChannelConfigurationInterfaceV1
    : public Service<ChannelConfigurationInterfaceV1> {
public:
    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    static constexpr const char* type_id = "channel_configuration.v1";
    static constexpr const char* release_version = "2.7.0";
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Fetch the instance a module publishes.
     *
     * @param module The module that claims the device, from
     * @ref Adapter::get_module.
     * @param allow_null Return an empty pointer, rather than throwing, when
     * the module publishes no channel configuration service.
     * @return The module's channel configuration service.
     * @throws std::runtime_error if the module publishes no channel
     * configuration service and allow_null is false.
     */
    static std::shared_ptr<ChannelConfigurationInterfaceV1> get_service(
        std::shared_ptr<Module> module, bool allow_null = false)
    {
        return Service<ChannelConfigurationInterfaceV1>::get_service(
            std::move(module), "", allow_null);
    }

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    virtual ~ChannelConfigurationInterfaceV1() = default;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Select the physical sensor to be used with this data channel.
     *
     * Following device enumeration, EnumerationMetadata includes device-specific
     * information about which data channel to use and which sensor is to be
     * transported over that channel.  You can modify this default behavior
     * and listen to a different sensor by calling Adapter::use_sensor, which
     * delegates to this method.
     *
     * Each HSB implementation is able to do what it wants here, but
     * typically `data_plane`, `sensor`, `sensor_number`, the SIF, VP and HIF
     * address fields, and the frame-end event are adjusted.
     *
     * @param[in,out] metadata Announcement to rewrite in place.
     * @param sensor_number Zero-based index of the sensor to describe.
     * @throws std::runtime_error if sensor_number is invalid.
     */
    virtual void use_sensor(
        EnumerationMetadata& metadata, int64_t sensor_number)
        = 0;

    /**
     * @brief Returns true if sensor_number is valid for this configuration.
     *
     * @param sensor_number Sensor number to test.
     * @return True if sensor_number is valid.
     */
    virtual bool is_sensor_valid(int64_t sensor_number) const = 0;

    /**
     * @brief Set the data-plane MTU.
     *
     * Suggest an MTU for use with this data channel; the
     * implementation is allowed to do any of these:
     *
     *  - use the specific value you've passed here,
     *  - adjust that value based on e.g. GPU page size, NIC constraints, or RDMA requirements
     *  - stick with it's default value, ignoring this suggestion entirely
     *
     * If the MTU isn't supported by all links in the network chain, the
     * result is undefined.  If the implementation doesn't ignore this setting,
     * then no data plane packets larger than the given MTU will be sent.
     *
     * @param[in,out] metadata Encapsulates connection parameters
     * @param mtu Data-plane MTU, in bytes.
     */
    virtual void use_mtu(EnumerationMetadata& metadata, uint32_t mtu) = 0;

    /**
     * @brief Set a multicast destination for the data plane.
     *
     * @param[in,out] metadata Encapsulates configuration data.
     * @param address Multicast group address.
     * @param port Destination UDP port to send to.
     */
    virtual void use_multicast(
        EnumerationMetadata& metadata, std::string address, uint16_t port)
        = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_CHANNEL_CONFIGURATION_HPP
