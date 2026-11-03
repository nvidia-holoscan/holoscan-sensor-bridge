/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_OSCILLATOR_HPP
#define HOLOLINK_MODULE_OSCILLATOR_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "enumeration_metadata.hpp"
#include "module.hpp"
#include "service.hpp"

namespace hololink::module {

/**
 * @brief Configure a clock source driving a sensor.
 *
 * Sensors usually require an external reference clock before they can be
 * configured. Call enable() with the rate the sensor requires, before
 * configuring the sensor.
 */
class OscillatorInterfaceV1 : public ConfigurableService<OscillatorInterfaceV1> {
public:
    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    static constexpr const char* type_id = "oscillator.v1";
    static constexpr const char* release_version = "2.7.0";

    /* Per-data-plane instance_id derivation used by the metadata-form
     * get_service. Throws if the metadata is missing 'serial_number'
     * or 'data_plane'. */
    static std::string locator_id(const EnumerationMetadata& metadata)
    {
        return "serial=" + metadata.get<std::string>("serial_number")
            + ";data_plane=" + std::to_string(metadata.get<int64_t>("data_plane"));
    }

    virtual ~OscillatorInterfaceV1() = default;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Start the oscillator at the given output rate.
     *
     * Calling this again with the rate the oscillator is already
     * producing succeeds without disturbing the output, so it is safe
     * to call from each sensor sharing a clock generator. Where the
     * generator can supply only one rate at a time, asking for a
     * different rate than the one already running returns `false`.
     * After the device is reset the output must be started again by
     * calling enable() again.
     *
     * @param clocks_per_second Requested output frequency, in Hz.
     * @return `true` if the oscillator is now producing the requested
     *      rate. `false` if the hardware cannot produce that rate, or
     *      cannot do so while another rate is in use.
     */
    virtual bool enable(uint64_t clocks_per_second) = 0;

    /**
     * @brief Report the oscillator's adjustable settings.
     *
     * Settings are hardware specific; keys and their meaning are defined
     * by the oscillator in use. Values are strings; how numeric values
     * are encoded is also defined by the oscillator. An oscillator with
     * no adjustable settings returns an empty map.
     *
     * @return The current value of each setting, by name.
     */
    virtual std::map<std::string, std::string> get_caps() = 0;

    /**
     * @brief Change one or more of the oscillator's adjustable settings.
     *
     * The update is all-or-nothing: either every entry is applied, or
     * none is.
     *
     * @param caps New values, by setting name; use the names reported by
     *      get_caps().
     * @return `true` if every entry was applied. `false` if any name is
     *      not supported or any value is out of range for the hardware,
     *      in which case no setting is changed.
     */
    virtual bool set_caps(const std::map<std::string, std::string>& caps) = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_OSCILLATOR_HPP
