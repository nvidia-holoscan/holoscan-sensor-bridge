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

/**
 * @brief Handle to a single data path on an HSB device.
 *
 * DataChannel represents the path that sensor or actuator data
 * travels between the host system (usually via RDMA) and the device
 * itself.
 *
 * - A single HSB device can support any number of DataChannel instances.
 * - Data producers or consumers use this object to find enumeration metadata
 *    provided by the application at construction time.  This data is configured
 *    by the application per application requirements (e.g. using a specific
 *    sensor number or configuration)-- enumeration metadata found just by
 *    listening to BOOTP messages may not have this application-specific setup.
 * - Channel specific objects are invalidated when `device_lost` is called here,
 *    so later calls to get_service will reinitialize these-- supporting
 *    device reconnection.
 *
 * @code
 *  auto& adapter = hololink::module::Adapter::get_adapter();
 *  auto metadata = adapter.wait_for_channel(hololink_ip, discovery_timeout);
 *  auto channel = hololink::module::DataChannelInterfaceV1::get_service(metadata);
 *  // Reach the device this channel runs on.
 *  channel->hololink()->start();
 * @endcode
 */
class DataChannelInterfaceV1
    : public ConfigurableService<DataChannelInterfaceV1> {
public:
    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    static constexpr const char* type_id = "data_channel.v1";
    static constexpr const char* release_version = "2.7.0";

    /* Per-channel instance_id derivation used by the metadata-form
     * get_service. Throws if the metadata is missing 'serial_number'
     * or 'data_channel'. */
    static std::string locator_id(const EnumerationMetadata& metadata)
    {
        return "serial=" + metadata.get<std::string>("serial_number")
            + ";data_channel=" + std::to_string(metadata.get<int64_t>("data_channel"));
    }

    virtual ~DataChannelInterfaceV1() = default;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief The enumeration metadata this channel was configured with.
     *
     * Every service on this channel is configured from this snapshot.
     *
     * @return The metadata describing this channel.
     */
    virtual const EnumerationMetadata& enumeration_metadata() const = 0;

    /**
     * @brief The device this channel runs on.
     *
     * The same device an application gets by fetching HololinkInterfaceV1
     * with this channel's enumeration metadata.
     *
     * @return The device.
     */
    virtual std::shared_ptr<HololinkInterfaceV1> hololink() const = 0;

    /**
     * @brief Invalidate the services cached for the device this channel
     *      runs on, following loss of contact with it.
     *
     * Does what HololinkInterfaceV1::device_lost() does, for applications
     * that hold a channel and not the device it runs on.
     *
     * @return HOLOLINK_MODULE_OK or an error status.
     */
    virtual hololink_module_status_t device_lost() = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_DATA_CHANNEL_HPP
