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

/**
 * @brief Access features on an HSB device.
 *
 * This is the application's handle to the device itself: it opens and closes
 * the control plane, drives the device into a known state, reads and writes
 * device registers, and hands out the other services that work on this
 * device -- data channels, I2C buses, sequencers and VSYNC sources.
 *
 * Fetch it with the EnumerationMetadata describing the device, then bring
 * the device up before configuring a sensor on it:
 *
 * @code
 *  auto& adapter = hololink::module::Adapter::get_adapter();
 *  auto metadata = adapter.wait_for_channel(hololink_ip, discovery_timeout);
 *  auto hololink = hololink::module::HololinkInterfaceV1::get_service(metadata);
 *  // No device I/O works until the control plane is open.
 *  hololink->start();
 *  // The framework leaves device state alone unless the application asks;
 *  // reset() drives the board into a known state.
 *  hololink->reset();
 *  // ... configure sensors, run the pipeline ...
 *  // Close connections to the device.
 *  hololink->stop();
 * @endcode
 */
class HololinkInterfaceV1
    : public ConfigurableService<HololinkInterfaceV1>,
      public std::enable_shared_from_this<HololinkInterfaceV1> {
public:
    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    static constexpr const char* type_id = "hololink.v1";
    static constexpr const char* release_version = "2.7.0";

    /* Per-board instance_id derivation used by the metadata-form
     * get_service. Throws if the metadata is missing 'serial_number'. */
    static std::string locator_id(const EnumerationMetadata& metadata)
    {
        return "serial=" + metadata.get<std::string>("serial_number");
    }

    virtual ~HololinkInterfaceV1() = default;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief The enumeration metadata this device was configured with.
     *
     * Services sharing this device read the configuration metadata from this
     * snapshot rather than keeping a copy of their own, so they are all built
     * from the same values.
     */
    virtual const EnumerationMetadata& enumeration_metadata() const = 0;

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    /* Used for testing only. */
    virtual std::shared_ptr<DataChannelInterfaceV1> default_data_channel() const = 0;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Open the control plane to the device.
     *
     * Register access, I2C traffic and anything else that reaches the device
     * works only while the control plane is open, so an application calls
     * this before configuring the device. Calling it again while it is open
     * does nothing.
     *
     * @return HOLOLINK_MODULE_OK or an error status.
     */
    virtual hololink_module_status_t start() = 0;

    /**
     * @brief Close the control plane to the device.
     *
     * Applications call this when they are done with the device; it does
     * nothing if the control plane is already closed.
     *
     * @return HOLOLINK_MODULE_OK or an error status.
     */
    virtual hololink_module_status_t stop() = 0;

    /**
     * @brief Drive the device into a known state.
     *
     * Resets the device, waits for it to come back and announce itself, then
     * applies the configuration the control plane needs. Every callback
     * registered through on_reset() fires. Applications reset a device before
     * configuring the sensors on it: the framework changes no device state on
     * its own.
     *
     * @return HOLOLINK_MODULE_OK or an error status.
     */
    virtual hololink_module_status_t reset() = 0;

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    /* Configure the device's HSB IP with various settings,
     * including enables for PTP client and IP address and UDP
     * port number for asynchronous messages.  Applications don't
     * call this; it's executed automatically on calls to start()
     * or reset().
     *
     * @return HOLOLINK_MODULE_OK or an error status.
     */
    virtual hololink_module_status_t configure_hsb() = 0;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Invalidate the services cached for this device, following loss
     *      of contact with it.
     *
     * Services built for a device accumulate device state, which goes stale
     * when the device is lost. Once an application discovers the loss it
     * calls this, and the framework drops its references to those service
     * objects. When the device comes back and is enumerated again, calls to
     * `get_service` build new instances, in sync with the recovered device.
     * The application drops its own references to the older objects; calls on
     * those stale instances fail in ways that depend on the service.
     *
     * @return HOLOLINK_MODULE_OK or an error status.
     */
    virtual hololink_module_status_t device_lost() = 0;

    /**
     * @brief Wait for the device's clock to synchronize with the PTP
     *      grandmaster.
     *
     * Blocks until the device's clock is disciplined, so that the timestamps
     * it reports with each frame share the host's PTP domain. Applications
     * that rely on those timestamps call this after start() or reset().
     *
     * @return True once the clock is synchronized; false if it did not
     * synchronize within a default timeout.
     */
    virtual bool ptp_synchronize() = 0;

    /**
     * @brief Handle to a callback registered with on_reset().
     *
     * The callback fires while the handle is held, and is unregistered when
     * the handle is destroyed. Hold it for as long as the callback should
     * fire.
     */
    class ResetRegistration {
    public:
        virtual ~ResetRegistration() = default;
    };

    /**
     * @brief Register a callback fired each time this device completes a
     *      reset.
     *
     * The callback runs once per reset(), after the device has rebooted and
     * its HSB configuration has been re-applied. Sensor drivers register here
     * to drop the hardware state they cached, so the next configuration pass
     * writes it to the device again. Any number of callbacks may be
     * registered.
     *
     * @param callback Invoked once per reset of this device.
     * @return Handle keeping the callback registered; destroying it
     * unregisters the callback.
     */
    virtual std::shared_ptr<ResetRegistration> on_reset(
        std::function<void()> callback)
        = 0;

    /**
     * @brief Write to locations within the HSB IP local bus.
     *
     * @param addresses Device addresses to write.
     * @param values The value written to each address.
     * @return HOLOLINK_MODULE_OK, or HOLOLINK_MODULE_INVALID_PARAMETER when
     * the two lists differ in length.
     */
    virtual hololink_module_status_t write_uint32(
        const std::vector<uint32_t>& addresses,
        const std::vector<uint32_t>& values)
        = 0;

    /**
     * @brief Read locations within the HSB IP local bus.
     *
     * @param addresses Device addresses to read.
     * @param[out] out_values Receives one value per address, in the order the
     * addresses were given.
     * @return HOLOLINK_MODULE_OK or an error status.
     */
    virtual hololink_module_status_t read_uint32(
        const std::vector<uint32_t>& addresses,
        std::vector<uint32_t>& out_values)
        = 0;

    /**
     * @brief Clear bits in an HSB IP local bus location.
     *
     * Reads the register, ANDs the value with `mask`, and writes it back.
     *
     * @param address Device address to update.
     * @param mask Value ANDed with the register's contents.
     * @return HOLOLINK_MODULE_OK or an error status.
     */
    virtual hololink_module_status_t and_uint32(uint32_t address, uint32_t mask) = 0;

    /**
     * @brief Set bits in an HSB IP local bus location.
     *
     * Reads the register, ORs the value with `mask`, and writes it back.
     *
     * @param address Device address to update.
     * @param mask Value ORed with the register's contents.
     * @return HOLOLINK_MODULE_OK or an error status.
     */
    virtual hololink_module_status_t or_uint32(uint32_t address, uint32_t mask) = 0;

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    /**
     * @brief Fetch the lock guarding this device's I2C controller.
     *
     * Provide access to the lock used to serialize access to I2C
     * transactions on this device; use with std::lock_guard or
     * std::unique_lock or std::scoped_lock.
     *
     * @param[out] out_lock Receives the lock handle.
     * @return HOLOLINK_MODULE_OK or an error status.
     */
    virtual hololink_module_status_t i2c_lock(
        std::unique_ptr<I2cLockV1>& out_lock)
        = 0;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    /* Used by RoceReceiver to configure registers in the HSB ROCE
     * encapsulation engine.
     * @tparam T The interface version to fetch.
     * @param md Metadata selecting and configuring the data channel.
     * @param allow_null Return nullptr rather than throwing when
     * this device publishes no such channel.
     * @return The channel, ready to use.
     */
    template <typename T = RoceDataChannelInterfaceV1>
    std::shared_ptr<T> get_roce_data_channel(
        const EnumerationMetadata& md, bool allow_null = false)
    {
        // Metadata-form get_service: caches and configures the
        // per-(serial, data_plane) channel through ConfigurableService,
        // so callers receive a fully-materialized handle.
        return T::get_service(md, allow_null);
    }
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Fetch a I2C driver object.
     *
     * The bus reaches every peripheral on it; each transaction names the
     * peripheral it addresses.
     *
     * @tparam T The interface version to fetch.
     * @param bus Which bus number within the device.
     * @param address Names the instance to fetch; a separate handle comes
     * back per value.
     * @param allow_null Return an empty pointer, rather than throwing, when
     * this device publishes no such bus.
     * @return The I2C interface for that bus.
     */
    template <typename T = I2cInterfaceV1>
    std::shared_ptr<T> get_i2c(uint32_t bus, uint32_t address,
        bool allow_null = false)
    {
        const std::string id = i2c_instance_id(bus, address);
        return T::get_service(module(), id.c_str(), allow_null);
    }

    /**
     * @brief Sequencer for use with immediately triggered
     *      sequences.
     *
     * Sequencers are implemented within the HSB IP to trigger
     * sets of HSB IP local bus transactions (e.g. to execute an
     * I2C transaction) on a specific flag (e.g. a GPIO or frame-done
     * event).  Use this sequencer when you want a sequence to run
     * immediately.
     *
     * @tparam T The interface version to fetch.
     * @param allow_null Return an empty pointer, rather than throwing, when
     * this device publishes no such sequencer.
     * @return The sequencer.
     */
    template <typename T = SequencerInterfaceV1>
    std::shared_ptr<T> software_sequencer(bool allow_null = false)
    {
        return T::get_service(module(), "kind=software", allow_null);
    }

    /**
     * @brief Sequencer to trigger a sequence when GPIO0 is active.
     *
     * @tparam T The interface version to fetch.
     * @param allow_null Return an empty pointer, rather than throwing, when
     * this device publishes no such sequencer.
     * @return The sequencer.
     */
    template <typename T = SequencerInterfaceV1>
    std::shared_ptr<T> gpio0_sequencer(bool allow_null = false)
    {
        return T::get_service(module(), "kind=gpio0", allow_null);
    }

    /**
     * @brief Sequencer to trigger a sequence when GPIO1 is active.
     *
     * @tparam T The interface version to fetch.
     * @param allow_null Return an empty pointer, rather than throwing, when
     * this device publishes no such sequencer.
     * @return The sequencer.
     */
    template <typename T = SequencerInterfaceV1>
    std::shared_ptr<T> gpio1_sequencer(bool allow_null = false)
    {
        return T::get_service(module(), "kind=gpio1", allow_null);
    }

    /**
     * @brief Sequencer to trigger a sequence when SIF0 (sensor interface
     *      0) receives the first data from a sensor following a frame end
     *      condition.
     *
     * @tparam T The interface version to fetch.
     * @param allow_null Return an empty pointer, rather than throwing, when
     * this device publishes no such sequencer.
     * @return The sequencer.
     */
    template <typename T = SequencerInterfaceV1>
    std::shared_ptr<T> sif0_frame_start_sequencer(bool allow_null = false)
    {
        return T::get_service(module(), "kind=sif0_frame_start", allow_null);
    }

    /**
     * @brief Sequencer to trigger a sequence when SIF1 (sensor interface
     *      1) receives the first data from a sensor following a frame end
     *      condition.
     *
     * @tparam T The interface version to fetch.
     * @param allow_null Return an empty pointer, rather than throwing, when
     * this device publishes no such sequencer.
     * @return The sequencer.
     */
    template <typename T = SequencerInterfaceV1>
    std::shared_ptr<T> sif1_frame_start_sequencer(bool allow_null = false)
    {
        return T::get_service(module(), "kind=sif1_frame_start", allow_null);
    }

    /**
     * @brief Fetch a PTP-based PPS output controller, for use with
     *      sensors that can be triggered externally (e.g. cameras with
     *      VSYNC inputs).
     *
     * One PTP PPS output serves the whole device, driven by the PPS the
     * device's PTP-disciplined clock produces.  Use the object returned
     * here to configure the PPS rate; this same object is usually passed
     * in to a device driver to ask it to use that as the sample trigger
     * input.
     *
     * @tparam T The interface version to fetch.
     * @param allow_null Return an empty pointer, rather than throwing, when
     * this device publishes no VSYNC source.
     * @return the PPS controller based on PTP
     */
    template <typename T = PtpPpsOutputInterfaceV1>
    std::shared_ptr<T> ptp_pps_output(bool allow_null = false)
    {
        const std::string id = ptp_pps_output_instance_id();
        return T::get_service(module(), id.c_str(), allow_null);
    }

    /**
     * @brief When a sensor takes a PPS sync input, use this instance
     *      to configure the device for free-running (unsynchronized)
     *      operation.
     *
     * @tparam T The interface version to fetch.
     * @param allow_null Return an empty pointer, rather than throwing, when
     * this device publishes no such source.
     * @return The VSYNC source.
     */
    template <typename T = VsyncInterfaceV1>
    std::shared_ptr<T> null_vsync(bool allow_null = false)
    {
        const std::string id = null_vsync_instance_id();
        return T::get_service(module(), id.c_str(), allow_null);
    }

    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
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
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_HOLOLINK_HPP
