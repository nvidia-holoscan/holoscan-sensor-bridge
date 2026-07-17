/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_PTP_PPS_OUTPUT_DEFAULT_HPP
#define HOLOLINK_MODULE_CORE_PTP_PPS_OUTPUT_DEFAULT_HPP

#include <memory>
#include <stdexcept>
#include <utility>

#include "hololink/module/ptp_pps_output.hpp"
#include "hololink/module/status.h"

#include "hololink/core/hololink.hpp"

#include "hololink_default.hpp"

namespace hololink::module::module_core {

/* PtpPpsOutputInterfaceV1 backed by the legacy hololink::Hololink's
 * built-in PtpSynchronizer (one per hololink::Hololink,
 * constructed during the Hololink ctor and reached via
 * Hololink::ptp_pps_output(freq)). */
class PtpPpsOutputV1 : public PtpPpsOutputInterfaceV1 {
public:
    PtpPpsOutputV1(std::shared_ptr<HololinkV1> owner,
        std::shared_ptr<LegacyHololinkAccess> hololink)
        : owner_(std::move(owner))
        , hololink_(std::move(hololink))
    {
        if (!owner_) {
            throw std::runtime_error(
                "While constructing PtpPpsOutputV1: "
                "owning HololinkV1 must be non-null");
        }
        if (!hololink_) {
            throw std::runtime_error(
                "While constructing PtpPpsOutputV1: "
                "backing Hololink access must be non-null");
        }
        owner_->register_associated(this);
    }

    hololink_module_status_t enable(unsigned frequency_hz) override
    {
        // ptp_pps_output(freq) caches frequency on the legacy
        // synchronizer and returns the per-Hololink singleton. The
        // legacy set_frequency throws on a different cached frequency,
        // which propagates out as fatal — matches the documented
        // INVALID_PARAMETER-on-frequency-mismatch contract without a
        // separate status return.
        std::shared_ptr<hololink::Synchronizer> sync
            = hololink_->ptp_pps_output(frequency_hz);
        if (!enabled_) {
            sync->setup();
            enabled_ = true;
        }
        return HOLOLINK_MODULE_OK;
    }

    hololink_module_status_t disable() override
    {
        if (enabled_) {
            std::shared_ptr<hololink::Synchronizer> sync
                = hololink_->ptp_pps_output(0);
            sync->shutdown();
            enabled_ = false;
        }
        return HOLOLINK_MODULE_OK;
    }

    hololink_module_status_t start() override
    {
        // VSYNC_CONTROL=1; pulses begin emitting on the GPIO. Legacy
        // start() is idempotent — safe to call from a one-shot
        // application operator after enable() has run.
        std::shared_ptr<hololink::Synchronizer> sync
            = hololink_->ptp_pps_output(0);
        sync->start();
        return HOLOLINK_MODULE_OK;
    }

    hololink_module_status_t stop() override
    {
        std::shared_ptr<hololink::Synchronizer> sync
            = hololink_->ptp_pps_output(0);
        sync->stop();
        return HOLOLINK_MODULE_OK;
    }

    bool is_enabled() const override { return enabled_; }

private:
    std::shared_ptr<HololinkV1> owner_;
    std::shared_ptr<LegacyHololinkAccess> hololink_;
    bool enabled_ = false;
};

} // namespace hololink::module::module_core

#endif // HOLOLINK_MODULE_CORE_PTP_PPS_OUTPUT_DEFAULT_HPP
