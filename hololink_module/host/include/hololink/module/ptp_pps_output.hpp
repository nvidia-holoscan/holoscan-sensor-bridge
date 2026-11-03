/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_PTP_PPS_OUTPUT_HPP
#define HOLOLINK_MODULE_PTP_PPS_OUTPUT_HPP

#include <string>

#include "enumeration_metadata.hpp"
#include "module.hpp"
#include "service.hpp"
#include "status.h"
#include "vsync.hpp"

namespace hololink::module {

/**
 * @brief Configure an output pin to pulse at a rate driven
 *      by PLL from the PTP PPS signal.
 *
 * A common use of this output is to drive a camera sensor's
 * VSYNC input: when multiple HSB units are driven from the PTP
 * of a single host, these output signals have been shown to
 * have less than 1us difference across those HSB devices.
 *
 * Typical use:
 *  -# Call enable() with the frame rate to configure the frequency;
 *      this does not yet enable the actual output signal.
 *  -# Configure each sensor for external-trigger operation.
 *  -# Once all sensors are waiting for their triggers, call start().
 *      This starts driving the actual PPS output.
 *  -# Call stop() to halt the pulses, and disable() when the output
 *     is no longer needed.
 *
 * The start(), stop() and is_enabled() operations are inherited from
 * VsyncInterfaceV1. Code that needs only those can obtain the same
 * object as a VsyncInterfaceV1, without knowing which kind of source is
 * driving the trigger.
 */
class PtpPpsOutputInterfaceV1 : public VsyncInterfaceV1,
                                public ConfigurableService<PtpPpsOutputInterfaceV1> {
public:
    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    static constexpr const char* type_id = "ptp_pps_output.v1";
    static constexpr const char* release_version = "2.7.0";

    /* ServiceAlias chain: PtpPpsOutputInterfaceV1 publishes under
     * VsyncInterfaceV1::type_id as well, so consumers fetching the
     * abstract base get the same impl instance. */
    using ServiceAlias = VsyncInterfaceV1;

    /* Hide the inherited Service<VsyncInterfaceV1>::get_service and
     * for_each_type_id (visible through the VsyncInterfaceV1 base) so
     * callers writing PtpPpsOutputInterfaceV1::get_service / ::for_each_type_id
     * reach this class's chain (which emits both type_ids), not the
     * base's. Both the instance_id and metadata forms of get_service
     * come through ConfigurableService<PtpPpsOutputInterfaceV1>. */
    using ConfigurableService<PtpPpsOutputInterfaceV1>::get_service;
    using Service<PtpPpsOutputInterfaceV1>::for_each_type_id;

    /* Per-board instance_id derivation used by the metadata-form
     * get_service. Throws if the metadata is missing 'serial_number'. */
    static std::string locator_id(const EnumerationMetadata& metadata)
    {
        return "serial=" + metadata.get<std::string>("serial_number");
    }

    virtual ~PtpPpsOutputInterfaceV1() = default;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Configure the trigger output to produce pulses at the given
     *      rate.
     *
     * After a successful call is_enabled() returns `true`. This only
     * configures the output; pulses are not produced until start() is
     * called.
     *
     * Calling this again with the frequency already configured succeeds
     * and changes nothing. Asking for a different frequency while the
     * output is enabled is rejected.
     *
     * @param frequency_hz Pulse rate, in Hz.
     * @return HOLOLINK_MODULE_OK on success.
     *      HOLOLINK_MODULE_INVALID_PARAMETER if the hardware does not
     *      support `frequency_hz`, or the output is already enabled at a
     *      different frequency.
     */
    virtual hololink_module_status_t enable(unsigned frequency_hz) = 0;

    /**
     * @brief Remove the trigger output configuration.
     *
     * Afterwards is_enabled() returns `false`. Does nothing if the output
     * is not enabled.
     *
     * @return HOLOLINK_MODULE_OK on success, or an error status.
     */
    virtual hololink_module_status_t disable() = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_PTP_PPS_OUTPUT_HPP
