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

/* PTP-PPS-driven VSYNC source. A PTP-locked pulse-per-second train
 * drives the trigger line connected to the cameras' frame-start
 * input. One instance per board, keyed by "serial=<serial_number>".
 *
 * Inherits VsyncInterfaceV1 so a consumer holding only a Vsync
 * (camera driver, future per-board controller) doesn't need to know
 * which source is wired up. Inherits ConfigurableService so callers
 * holding only an EnumerationMetadata reach it via the standard
 * metadata-form get_service. The publisher registers the impl under
 * both `vsync.v1` and `ptp_pps_output.v1` type_ids by walking the
 * ServiceAlias chain — a single publish call covers both. */
class PtpPpsOutputInterfaceV1 : public VsyncInterfaceV1,
                                public ConfigurableService<PtpPpsOutputInterfaceV1> {
public:
    static constexpr const char* type_id = "ptp_pps_output.v1";

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

    /* Program the FPGA VSYNC block for the requested pulse rate, in
     * Hz. Impls reject unsupported frequencies with
     * HOLOLINK_MODULE_INVALID_PARAMETER.
     *
     * Idempotent: a second call with the same frequency returns
     * HOLOLINK_MODULE_OK; a different frequency on an already-enabled
     * output returns HOLOLINK_MODULE_INVALID_PARAMETER. */
    virtual hololink_module_status_t enable(unsigned frequency_hz) = 0;

    /* Tear down the VSYNC configuration. Idempotent. */
    virtual hololink_module_status_t disable() = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_PTP_PPS_OUTPUT_HPP
