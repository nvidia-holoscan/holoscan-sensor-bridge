/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_HOLOLINK_HPP
#define HOLOLINK_MODULE_HOLOLINK_HPP

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "enumeration_metadata.hpp"
#include "i2c_lock.hpp"
#include "module.hpp"
#include "service.hpp"
#include "status.h"

namespace hololink::module {

// Forward decls for the template default args; the corresponding
// headers (i2c.hpp, roce_data_channel.hpp, sequencer.hpp) bring in
// the full definitions when callers need to actually instantiate
// the templates.
class I2cInterfaceV1;
class RoceDataChannelInterfaceV1;
class SequencerInterfaceV1;
class DataChannelInterfaceV1;
class PtpPpsOutputInterfaceV1;
class VsyncInterfaceV1;

/* Per-board control plane. Reached via HololinkInterfaceV1::get_service
 * with the per-board instance_id "serial=<serial_number>". The
 * metadata-form get_service inherited from ConfigurableService
 * resolves the supplement module, gets-or-constructs the per-board
 * impl, and calls configure(metadata) on it before returning.
 *
 * Per-board EnumerationMetadata cache. The metadata stashed on the
 * impl is the snapshot per-board services (I2c, Oscillator,
 * per-board supplements) read off via enumeration_metadata(). They
 * hold a shared_ptr<HololinkInterfaceV1> and consult that accessor
 * rather than caching their own copy. The per-channel counterpart
 * lives on DataChannelInterfaceV1.
 *
 * Default DataChannel. configure(metadata) constructs a single
 * DataChannelInterfaceV1 bound to this Hololink and returns it via
 * default_data_channel(). The two are configured from the same
 * metadata blob in the same call, so their per-board fields cannot
 * drift. Board-level code that needs a channel handle reads this
 * accessor; per-data-plane channels beyond the default are reached
 * through DataChannelInterfaceV1::get_service(metadata). */
class HololinkInterfaceV1
    : public ConfigurableService<HololinkInterfaceV1>,
      public std::enable_shared_from_this<HololinkInterfaceV1> {
public:
    static constexpr const char* type_id = "hololink.v1";

    /* Per-board instance_id derivation used by the metadata-form
     * get_service. Throws if the metadata is missing 'serial_number'. */
    static std::string locator_id(const EnumerationMetadata& metadata)
    {
        return "serial=" + metadata.get<std::string>("serial_number");
    }

    virtual ~HololinkInterfaceV1() = default;

    /* The per-board enumeration metadata the impl was configured
     * with — the snapshot of fields the channel(s) on this board
     * agreed on at construction time. Per-board V1 services that
     * need metadata (I2c, Oscillator, per-board supplements) read
     * through this accessor rather than holding their own copy. */
    virtual const EnumerationMetadata& enumeration_metadata() const = 0;

    /* The DataChannel constructed alongside this Hololink during
     * configure(metadata). Shares the same metadata blob the
     * Hololink was configured with, so its per-board fields agree
     * with this Hololink's by construction. Null until configure
     * has run, and null if the configure-time metadata lacked the
     * data_plane field required to identify a channel. */
    virtual std::shared_ptr<DataChannelInterfaceV1> default_data_channel() const = 0;

    // --- Device lifecycle ---
    virtual hololink_module_status_t start() = 0;
    virtual hololink_module_status_t stop() = 0;
    virtual hololink_module_status_t reset() = 0;
    virtual hololink_module_status_t configure_hsb() = 0;

    /* Invalidate this device's cached services so that, once the device
     * is rediscovered, re-resolving them yields fresh instances in sync
     * with the recovered device. Services the caller still holds stay
     * valid until dropped; the caller must re-fetch to resync. Called by
     * the reconnection path when the device is lost. */
    virtual hololink_module_status_t device_lost() = 0;

    /* Block until the board's clock is disciplined to the PTP grandmaster
     * (using a default timeout), returning true on success. Applications
     * that validate PTP-synchronized frame timestamps call this after
     * start()/reset() so the FPGA's timestamp clock shares the host's PTP
     * domain. Returns false if synchronization didn't complete in time. */
    virtual bool ptp_synchronize() = 0;

    /* RAII handle for a callback registered via on_reset(). Destroying it
     * unregisters the callback. Because the per-board HololinkInterfaceV1
     * is a process-lifetime singleton, callers (e.g. a camera) must hold
     * the handle for as long as the callback should fire and drop it when
     * they go out of scope — otherwise registrations would accumulate on
     * the singleton for the life of the process. */
    class ResetRegistration {
    public:
        virtual ~ResetRegistration() = default;
    };

    /* Register a callback fired each time the board completes a reset
     * (after the FPGA has rebooted and HSB is reconfigured). Multiple
     * callbacks may be registered; each fires once per reset(). Used by
     * sensor drivers to drop cached hardware state so the next start()
     * re-commits it. The returned handle unregisters the callback when
     * destroyed; keep it alive for as long as the callback should fire. */
    virtual std::shared_ptr<ResetRegistration> on_reset(
        std::function<void()> callback)
        = 0;

    // --- Control plane ---
    virtual hololink_module_status_t write_uint32(
        const std::vector<uint32_t>& addresses,
        const std::vector<uint32_t>& values)
        = 0;
    virtual hololink_module_status_t read_uint32(
        const std::vector<uint32_t>& addresses,
        std::vector<uint32_t>& out_values)
        = 0;
    virtual hololink_module_status_t and_uint32(uint32_t address, uint32_t mask) = 0;
    virtual hololink_module_status_t or_uint32(uint32_t address, uint32_t mask) = 0;

    /* Returns an unlocked I2cLockV1 the caller acquires through
     * std::lock_guard / std::unique_lock / std::scoped_lock. */
    virtual hololink_module_status_t i2c_lock(
        std::unique_ptr<I2cLockV1>& out_lock)
        = 0;

    // --- Child-object factories ---
    // Templated so callers can pin a future version (e.g.
    // T = RoceDataChannelInterfaceV2). The default is the explicit
    // current version and never moves on its own — upgrading is a
    // deliberate edit at the call site. The factory builds the
    // per-board instance_id via the protected *_instance_id hooks.

    template <typename T = RoceDataChannelInterfaceV1>
    std::shared_ptr<T> get_roce_data_channel(
        const EnumerationMetadata& md, bool allow_null = false)
    {
        // Metadata-form get_service: caches and configures the
        // per-(serial, data_plane) channel through ConfigurableService,
        // so callers receive a fully-materialized handle.
        return T::get_service(md, allow_null);
    }

    template <typename T = I2cInterfaceV1>
    std::shared_ptr<T> get_i2c(uint32_t bus, uint32_t address,
        bool allow_null = false)
    {
        const std::string id = i2c_instance_id(bus, address);
        return T::get_service(module(), id.c_str(), allow_null);
    }

    // --- Sequencer factories (fixed instance ids) ---
    template <typename T = SequencerInterfaceV1>
    std::shared_ptr<T> software_sequencer(bool allow_null = false)
    {
        return T::get_service(module(), "kind=software", allow_null);
    }
    template <typename T = SequencerInterfaceV1>
    std::shared_ptr<T> gpio0_sequencer(bool allow_null = false)
    {
        return T::get_service(module(), "kind=gpio0", allow_null);
    }
    template <typename T = SequencerInterfaceV1>
    std::shared_ptr<T> gpio1_sequencer(bool allow_null = false)
    {
        return T::get_service(module(), "kind=gpio1", allow_null);
    }
    template <typename T = SequencerInterfaceV1>
    std::shared_ptr<T> sif0_frame_start_sequencer(bool allow_null = false)
    {
        return T::get_service(module(), "kind=sif0_frame_start", allow_null);
    }
    template <typename T = SequencerInterfaceV1>
    std::shared_ptr<T> sif1_frame_start_sequencer(bool allow_null = false)
    {
        return T::get_service(module(), "kind=sif1_frame_start", allow_null);
    }

    /* Per-board PTP-PPS-driven VSYNC source. Returns the producer-side
     * surface (with enable/disable), so callers can configure the
     * pulse frequency. Camera drivers that consume the source as a
     * Vsync take a shared_ptr<VsyncInterfaceV1>; this factory's return
     * value upcasts implicitly. */
    template <typename T = PtpPpsOutputInterfaceV1>
    std::shared_ptr<T> ptp_pps_output(bool allow_null = false)
    {
        const std::string id = ptp_pps_output_instance_id();
        return T::get_service(module(), id.c_str(), allow_null);
    }

    /* No-op VsyncInterfaceV1 from the module. Returned by
     * applications that want a concrete Vsync handle to pass to a
     * consumer (e.g. Vb1940Cam) without wiring up a real trigger
     * source. */
    template <typename T = VsyncInterfaceV1>
    std::shared_ptr<T> null_vsync(bool allow_null = false)
    {
        const std::string id = null_vsync_instance_id();
        return T::get_service(module(), id.c_str(), allow_null);
    }

protected:
    /* Per-factory instance-id hooks. Each derived class overrides
     * these to translate factory arguments into instance-id strings
     * its module-side get_service dispatch recognizes. Every hook
     * returns an id in "name=value;name=value;..." form. Every id
     * includes a serial=<serial_number> pair — the per-board scoping
     * that disambiguates services published by two boards sharing
     * the same module. The data-channel hook returns
     * "serial=<serial_number>;data_channel=<n>"; the I2C hook
     * returns "serial=<serial_number>;bus=<bus>;address=<address>",
     * so two boards' I2C bus 1 / 0x42 do not collide. */
    virtual std::string roce_data_channel_instance_id(
        const EnumerationMetadata& md)
        = 0;
    virtual std::string i2c_instance_id(uint32_t bus, uint32_t address) = 0;

    /* Per-board instance_id hook for the PTP-PPS-driven VSYNC source.
     * Returns "serial=<serial_number>"; the source is one physical
     * FPGA resource per board, shared across every data plane. */
    virtual std::string ptp_pps_output_instance_id() = 0;

    /* Per-board instance_id hook for the null VSYNC source. The
     * module-side impl is keyed under a distinct instance_id from
     * the real PtpPpsOutput so both can coexist under the
     * "vsync.v1" type_id without collision. */
    virtual std::string null_vsync_instance_id() = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_HOLOLINK_HPP
