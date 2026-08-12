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

/* Programming interface for the MIPI D-PHY. A receiver that is never
 * programmed keeps the FPGA's defaults.
 */
class MipiDphyInterfaceV1 : public ConfigurableService<MipiDphyInterfaceV1> {
public:
    static constexpr const char* type_id = "mipi_dphy.v1";

    static std::string locator_id(const EnumerationMetadata& metadata)
    {
        return "serial=" + metadata.get<std::string>("serial_number");
    }

    /* Pass as line_rate_mbps for the design's fastest rate */
    static constexpr unsigned LINE_RATE_DESIGN_MAXIMUM = 0;

    virtual ~MipiDphyInterfaceV1() = default;

    /* Program the physical MIPI interface at physical_mipi_interface:
     * lane_count sets its active CSI-2 data lanes, and line_rate_mbps - the
     * per-lane bit rate - sets its data-lane settle window.
     *
     * Returns HOLOLINK_MODULE_INVALID_PARAMETER for an interface the design
     * does not have, a lane_count outside 1..4, or a rate above what the
     * design supports, and HOLOLINK_MODULE_NOT_FOUND if depencdencies are
     * not available */
    virtual hololink_module_status_t program(
        unsigned physical_mipi_interface,
        unsigned lane_count,
        unsigned line_rate_mbps)
        = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_MIPI_DPHY_HPP
