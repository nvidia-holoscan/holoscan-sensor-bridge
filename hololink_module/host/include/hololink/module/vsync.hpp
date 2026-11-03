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

/**
 * @brief Controls the VSYNC trigger that starts frames on a board's
 * cameras.
 *
 * VSYNC is a timing pulse train sent to the cameras' frame-start (external
 * trigger) inputs, so that the cameras begin their frames together.
 * Something has to generate those pulses; this interface is what a
 * camera driver, or an application, needs in order to use whichever
 * source is installed. PtpPpsOutputInterfaceV1 is the source currently
 * provided, and the same object is available through both interfaces.
 *
 * This interface does not configure the source: choosing the pulse rate
 * and making the source ready (so that is_enabled() is true) is done
 * through the source's own interface. Here, a camera driver checks
 * is_enabled() to decide whether to put its sensor in external-trigger
 * mode, and the application calls start() once the cameras are ready
 * and stop() when done.
 */
class VsyncInterfaceV1 : public Service<VsyncInterfaceV1> {
public:
    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    static constexpr const char* type_id = "vsync.v1";
    static constexpr const char* release_version = "2.7.0";

    virtual ~VsyncInterfaceV1() = default;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Whether the source has been configured to produce pulses.
     *
     * Camera drivers consult this while configuring, to decide whether
     * to put their sensor in external-trigger mode. start() and stop()
     * are only meaningful while this is `true`.
     *
     * @return `true` if the source is configured; `false` otherwise.
     */
    virtual bool is_enabled() const = 0;

    /**
     * @brief Begin generating trigger pulses.
     *
     * The application normally calls this once, after every camera has
     * been put into external-trigger mode and is waiting, so that the
     * first pulse reaches all of them at the same time. Calling it while
     * pulses are already being generated has no effect. If the source
     * is not enabled (see is_enabled()), no pulses are generated.
     *
     * @return HOLOLINK_MODULE_OK on success, or an error status.
     */
    virtual hololink_module_status_t start() = 0;

    /**
     * @brief Stop generating trigger pulses.
     *
     * The source remains configured, so start() can resume the pulses.
     * Calling it when no pulses are being generated has no effect.
     *
     * @return HOLOLINK_MODULE_OK on success, or an error status.
     */
    virtual hololink_module_status_t stop() = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_VSYNC_HPP
