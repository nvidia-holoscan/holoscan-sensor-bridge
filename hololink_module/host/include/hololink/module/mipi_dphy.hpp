/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HOLOLINK_MODULE_MIPI_DPHY_HPP
#define HOLOLINK_MODULE_MIPI_DPHY_HPP

#include <string>

#include "enumeration_metadata.hpp"
#include "module.hpp"
#include "service.hpp"
#include "status.h"

namespace hololink::module {

/**
 * @brief Programs the MIPI D-PHY receivers of a device.
 *
 * Each physical MIPI interface on the device has its own D-PHY receiver.
 * Use program() to tell a receiver how many CSI-2 data lanes the attached
 * sensor drives and at what rate, so that it samples the incoming data
 * correctly. A receiver that is never programmed keeps the FPGA's default
 * configuration.
 *
 * The service is looked up per device, using the enumeration metadata
 * for that device. It is only relevant for devices supporting MIPI D-PHYs.
 */
class MipiDphyInterfaceV1 : public ConfigurableService<MipiDphyInterfaceV1> {
public:
    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    static constexpr const char* type_id = "mipi_dphy.v1";
    static constexpr const char* release_version = UNRELEASED;

    static std::string locator_id(const EnumerationMetadata& metadata)
    {
        return "serial=" + metadata.get<std::string>("serial_number");
    }
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * Pass as `line_rate_mbps` to program() to select the fastest line
     * rate the FPGA design supports.
     */
    static constexpr unsigned LINE_RATE_DESIGN_MAXIMUM = 0;

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    virtual ~MipiDphyInterfaceV1() = default;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Configure one D-PHY receiver for a given lane count and
     *      line rate.
     *
     * Sets the number of active CSI-2 data lanes and the data-lane
     * HS-SETTLE window, which is derived from `line_rate_mbps` so that it
     * falls within the D-PHY specification's T_HS-SETTLE limits for that
     * rate. The setting takes effect on the receiver's subsequent input;
     * call this before starting the sensor's CSI-2 transmission. It may be
     * called again to reprogram the same receiver.
     *
     * Parameters are validated before the device is modified. If an
     * error is returned for a parameter problem, the receiver is
     * unchanged.
     *
     * @param physical_mipi_interface Zero-based index of the MIPI
     *      interface to program. The number of interfaces depends on the
     *      FPGA design.
     * @param lane_count Number of CSI-2 data lanes the sensor transmits
     *      on, from 1 to 4.
     * @param line_rate_mbps Bit rate of each data lane, in Mbps, or
     *      LINE_RATE_DESIGN_MAXIMUM for the fastest rate the design
     *      supports.
     * @return HOLOLINK_MODULE_OK on success.
     *      HOLOLINK_MODULE_INVALID_PARAMETER if `physical_mipi_interface`
     *      does not exist on this design, `lane_count` is outside 1..4,
     *      `line_rate_mbps` exceeds the design's maximum, or no valid
     *      settle window can be programmed for `line_rate_mbps` (for
     *      example, a rate too low for the design to represent).
     *      HOLOLINK_MODULE_NOT_FOUND if the device has no usable D-PHY
     *      or the receiver cannot be reached. Other error statuses are
     *      returned if communicating with the device fails.
     */
    virtual hololink_module_status_t program(
        unsigned physical_mipi_interface,
        unsigned lane_count,
        unsigned line_rate_mbps)
        = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_MIPI_DPHY_HPP
