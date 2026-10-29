/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_ENUMERATION_HPP
#define HOLOLINK_MODULE_ENUMERATION_HPP

#include <cstddef>
#include <cstdint>
#include <memory>

#include "enumeration_metadata.hpp"
#include "module.hpp"
#include "service.hpp"
#include "status.h"

namespace hololink::module {

/**
 * @brief A loadable Module supplements device EnumerationMetadata
 *      through this service.
 *
 * Devices typically announce themselves via BOOTP messages which
 * carry a specific set of identification and configuration information,
 * like device UUID and HSB-IP version ("compat_id").  That's usually
 * not enough information for most applications to work-- for example,
 * the number of sensors and data channels on the system determine
 * the addressing for the specific channels within the device.
 * This service allows the device module, which is loaded via UUID
 * and compat_id, to update EnumerationMetadata with other appropriate
 * configuration data for this device.  This supplemented enumeration
 * metadata is passed up to the application layer, and will be used
 * to direct construction of services used on the device.
 */
class EnumerationInterfaceV1 : public Service<EnumerationInterfaceV1> {
public:
    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    static constexpr const char* type_id = "enumeration.v1";
    static constexpr const char* release_version = "2.7.0";
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Fetch a module's EnumerationInterfaceV1 instance.
     *
     * @param module The module that claims the device, from
     * @ref Adapter::get_module.
     * @param allow_null Return an empty pointer, rather than throwing, when
     * the module publishes no enumeration service.
     * @return The module's enumeration service.
     * @throws std::runtime_error if the module publishes no enumeration
     * service and allow_null is false.
     */
    static std::shared_ptr<EnumerationInterfaceV1> get_service(
        std::shared_ptr<Module> module, bool allow_null = false)
    {
        return Service<EnumerationInterfaceV1>::get_service(
            std::move(module), "", allow_null);
    }

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    virtual ~EnumerationInterfaceV1() = default;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Complete the metadata describing an announced device.
     *
     * Called once per announcement, before the announcement reaches the
     * application; announcements decoded from bootp messages arrive on the
     * Reactor thread.
     *
     * @param[in,out] metadata Announcement to complete in place.
     * @param raw_packet The bootp packet `metadata` was decoded from; a module
     * whose devices carry vendor-specific bytes needs it to complete the
     * metadata. Null when the announcement did not come from a packet.
     * @param raw_packet_len Length of `raw_packet`, in bytes.
     * @return HOLOLINK_MODULE_OK once the metadata is complete;
     * HOLOLINK_MODULE_ENUMERATION_SKIPPED to decline the device, which
     * withholds the announcement from the application and lets enumeration of
     * other devices continue; any other status reports a failure, which the
     * Adapter raises as a std::runtime_error.
     */
    virtual hololink_module_status_t update_metadata(
        EnumerationMetadata& metadata,
        const uint8_t* raw_packet, size_t raw_packet_len)
        = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_ENUMERATION_HPP
