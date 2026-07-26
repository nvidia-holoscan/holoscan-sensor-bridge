/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_HSB_LITE_DEFAULT_HPP
#define HOLOLINK_MODULE_CORE_HSB_LITE_DEFAULT_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/hsb_lite/hsb_lite.hpp"
#include "hololink/module/oscillator.hpp"
#include "hololink/module/service.hpp"
#include "hololink/module/status.h"

#include <hololink/sensors/camera/imx274/renesas_bajoran_lite_ts1.hpp>

#include "hololink_default.hpp"

namespace hololink::module::module_core {

/* Canonical HsbLiteInterfaceV1 impl. Holds the per-board
 * HololinkV1 via the cache-only get_service(module, instance_id)
 * form; setup_clock dispatches through the legacy access. */
class HsbLiteV1 : public hsb_lite::HsbLiteInterfaceV1,
                  public Service<HsbLiteV1> {
public:
    static constexpr const char* type_id = "hsb_lite.module_core.v1";
    using Service<HsbLiteV1>::get_service;
    using Service<HsbLiteV1>::for_each_type_id;
    using ServiceAlias = hsb_lite::HsbLiteInterfaceV1;

    HsbLiteV1() = default;

    void configure(const EnumerationMetadata& metadata) override
    {
        const std::string hololink_id
            = HololinkInterfaceV1::locator_id(metadata);
        hololink_ = HololinkV1::get_service(
            this->module(), hololink_id.c_str());
        hololink_->register_associated(this);
    }

    hololink_module_status_t setup_clock(
        const std::vector<std::vector<uint8_t>>& clock_profile) override
    {
        auto impl = hololink_->legacy_access();
        impl->setup_clock(clock_profile);
        return HOLOLINK_MODULE_OK;
    }

    hololink_module_status_t trigger_reset() override
    {
        hololink_->legacy_access()->trigger_reset();
        return HOLOLINK_MODULE_OK;
    }

private:
    std::shared_ptr<HololinkV1> hololink_;
};

/* Canonical OscillatorInterfaceV1 impl for HSB-Lite carriers. The
 * on-board Renesas Bajoran TS1 is shared across the board's data
 * planes, so this impl is a per-board singleton: HsbLitePublisher's
 * construct_oscillator publishes the same shared_ptr under every
 * data-plane instance_id for a given serial. That sharing is what
 * makes the in-impl rate cache correct — `enable()` programs the
 * chip on the first call and short-circuits subsequent same-rate
 * calls instead of re-running setup_clock, which would otherwise
 * toggle the shared sensor-reset register 0x8 = 0x30 -> 0x3 and
 * disrupt a sibling channel that's already streaming.
 *
 * The cache is keyed to the LegacyHololinkAccess it programmed. A reconnect
 * materializes a new legacy Hololink, which enable() sees as a cache miss and
 * reprograms; a per-legacy on_reset listener additionally drops the cache when
 * a reused Hololink is reset (e.g. a later pipeline run's board bring-up wipes
 * the clock). The survivor would otherwise carry a stale committed rate across
 * a device reset and skip reprogramming. */
class HsbLiteOscillatorV1 : public OscillatorInterfaceV1,
                            public Service<HsbLiteOscillatorV1> {
public:
    static constexpr const char* type_id = "oscillator.module_core.v1";
    using Service<HsbLiteOscillatorV1>::get_service;
    using Service<HsbLiteOscillatorV1>::for_each_type_id;
    using ServiceAlias = OscillatorInterfaceV1;

    HsbLiteOscillatorV1() = default;

    void configure(const EnumerationMetadata& metadata) override
    {
        hololink_id_ = HololinkInterfaceV1::locator_id(metadata);
        hololink_ = HololinkV1::get_service(
            this->module(), hololink_id_.c_str());
        hololink_->register_associated(this);
    }

    bool enable(uint64_t clocks_per_second) override
    {
        // The committed rate is valid only for the specific
        // LegacyHololinkAccess it was programmed on. This oscillator is a
        // per-board survivor (shared by serial), so it outlives the
        // HololinkV1/legacy it configured against; configure() runs once, but a
        // reconnect (fresh HololinkV1) or a fresh pipeline run on the same board
        // (reused HololinkV1, later reset()) leaves this survivor with a stale
        // rate. Re-resolve the current board Hololink and treat a changed legacy
        // as a cache miss.
        auto hololink
            = HololinkV1::get_service(this->module(), hololink_id_.c_str());
        auto legacy = hololink->legacy_access();
        if (committed_clock_rate_.has_value()
            && committed_legacy_.lock() == legacy) {
            // Same legacy, already programmed: same rate → no-op success; a
            // different rate fails (the on-board clock generator serves one
            // rate at a time). The short-circuit avoids re-toggling the shared
            // sensor-reset register and disrupting a sibling channel already
            // streaming on this connection.
            return *committed_clock_rate_ == clocks_per_second;
        }
        legacy->setup_clock(hololink::renesas::DEVICE_CONFIGURATION);
        // Drop the cache if THIS legacy's device is reset (its
        // post_reset_configuration fires reset controllers), so a same-legacy
        // reset — e.g. a later run's board bring-up on a reused HololinkV1 —
        // reprograms the wiped clock. Registered per programmed legacy because
        // configure() (once, on the first Hololink) can't reach later ones.
        legacy->on_reset(std::make_shared<ResetClockRate>(&committed_clock_rate_));
        committed_clock_rate_ = clocks_per_second;
        committed_legacy_ = legacy;
        return true;
    }

    std::map<std::string, std::string> get_caps() override
    {
        // TODO: report the oscillator's tunable knobs (current values).
        return {};
    }

    bool set_caps(const std::map<std::string, std::string>& /*caps*/) override
    {
        // TODO: apply the requested cap updates. Return true on full
        // acceptance, false if any key is unknown or any value is out
        // of range. All-or-nothing — do not apply partial updates.
        return false;
    }

private:
    class ResetClockRate : public hololink::Hololink::ResetController {
    public:
        explicit ResetClockRate(std::optional<uint64_t>* committed)
            : committed_(committed)
        {
        }
        void reset() override { committed_->reset(); }

    private:
        std::optional<uint64_t>* committed_;
    };

    std::shared_ptr<HololinkV1> hololink_;
    std::string hololink_id_;
    std::optional<uint64_t> committed_clock_rate_;
    // The legacy Hololink committed_clock_rate_ was programmed on; a different
    // one (fresh connection) invalidates the cache.
    std::weak_ptr<LegacyHololinkAccess> committed_legacy_;
};

} // namespace hololink::module::module_core

#endif // HOLOLINK_MODULE_CORE_HSB_LITE_DEFAULT_HPP
