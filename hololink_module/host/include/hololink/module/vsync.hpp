/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_VSYNC_HPP
#define HOLOLINK_MODULE_VSYNC_HPP

#include "module.hpp"
#include "service.hpp"
#include "status.h"

namespace hololink::module {

/* Abstract base for any source that drives a board's VSYNC trigger
 * line. PTP-PPS is the source the framework ships today (see
 * PtpPpsOutputInterfaceV1 in ptp_pps_output.hpp); external trigger
 * inputs, software triggers, or other timing references can be added
 * later as additional derived interfaces without touching consumers.
 *
 * Consumers (camera drivers) only need to know whether *some* source
 * is currently driving the trigger so they can switch the sensor's
 * trigger-input mode on or off. They do not configure the source —
 * that is the producer interface's responsibility. */
class VsyncInterfaceV1 : public Service<VsyncInterfaceV1> {
public:
    static constexpr const char* type_id = "vsync.v1";

    virtual ~VsyncInterfaceV1() = default;

    /* True when this source is configured and will respond to
     * start()/stop(). Camera drivers consult this during configure()
     * to decide whether to put their sensor in external-trigger mode. */
    virtual bool is_enabled() const = 0;

    /* Begin emitting trigger pulses on the wire. Idempotent — calling
     * start() twice is safe. The application typically arranges for
     * start() to fire once after every camera has finished entering
     * external-sync waiting state, so the first pulse lands on all
     * cameras simultaneously. */
    virtual hololink_module_status_t start() = 0;

    /* Stop emitting trigger pulses. Idempotent. */
    virtual hololink_module_status_t stop() = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_VSYNC_HPP
