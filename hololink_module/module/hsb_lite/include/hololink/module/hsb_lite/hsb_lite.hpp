/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_HSB_LITE_HSB_LITE_HPP
#define HOLOLINK_MODULE_HSB_LITE_HSB_LITE_HPP

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/module.hpp"
#include "hololink/module/service.hpp"
#include "hololink/module/status.h"

namespace hololink::module::hsb_lite {

/* HSB-Lite-specific board surface. One supplement instance per HSB-Lite
 * board, keyed in the locator by "serial=<serial_number>". The
 * metadata-form get_service inherited from ConfigurableService caches
 * the per-board impl and runs configure(metadata) so the impl resolves
 * its underlying HololinkInterface lazily. */
class HsbLiteInterfaceV1
    : public ConfigurableService<HsbLiteInterfaceV1> {
public:
    static constexpr const char* type_id = "hsb_lite.v1";

    /* Per-board instance_id derivation used by the metadata-form
     * get_service. Throws if the metadata is missing 'serial_number'. */
    static std::string locator_id(const EnumerationMetadata& metadata)
    {
        return "serial=" + metadata.get<std::string>("serial_number");
    }

    virtual ~HsbLiteInterfaceV1() = default;

    /* Program the on-board Renesas Bajoran Lite TS1 clock generator
     * with the supplied per-register profile. Each inner vector is
     * a single I2C transaction the existing core's setup_clock issues
     * verbatim. */
    virtual hololink_module_status_t setup_clock(
        const std::vector<std::vector<uint8_t>>& clock_profile)
        = 0;

    /* Reset the board and return without waiting for it to re-announce
     * or reconfiguring HSB; the board reboots and device I/O fails until
     * it re-enumerates. Leaves recovery to the pipeline, unlike the
     * blocking HololinkInterfaceV1::reset(). Models an abrupt loss for
     * the reconnection path. */
    virtual hololink_module_status_t trigger_reset() = 0;
};

} // namespace hololink::module::hsb_lite

#endif // HOLOLINK_MODULE_HSB_LITE_HSB_LITE_HPP
