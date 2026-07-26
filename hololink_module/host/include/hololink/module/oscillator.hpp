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

/* Per-data-plane oscillator the supplement programs to drive a
 * downstream sensor's pixel clock (or whatever the data plane's
 * outbound clock domain is). Each board publishes one instance per
 * data plane under the same "serial=<serial_number>;data_plane=<n>"
 * instance_id the matching RoceDataChannelInterfaceV1 uses. The
 * metadata-form get_service inherited from ConfigurableService
 * resolves the supplement module, gets-or-constructs the per-data-plane
 * impl, and calls configure(metadata) on it before returning. */
class OscillatorInterfaceV1 : public ConfigurableService<OscillatorInterfaceV1> {
public:
    static constexpr const char* type_id = "oscillator.v1";

    /* Per-data-plane instance_id derivation used by the metadata-form
     * get_service. Throws if the metadata is missing 'serial_number'
     * or 'data_plane'. */
    static std::string locator_id(const EnumerationMetadata& metadata)
    {
        return "serial=" + metadata.get<std::string>("serial_number")
            + ";data_plane=" + std::to_string(metadata.get<int64_t>("data_plane"));
    }

    virtual ~OscillatorInterfaceV1() = default;

    /* Program the oscillator for `clocks_per_second`. Returns true
     * if the requested rate is achievable on this hardware, false
     * if the configuration is not possible (e.g. the rate is outside
     * the oscillator's range, or no integer divisor exists). */
    virtual bool enable(uint64_t clocks_per_second) = 0;

    /* Tunable knobs the oscillator exposes — implementation-defined
     * keys (e.g. "spread_spectrum", "output_swing") mapped to their
     * current values. Both keys and values are strings so the surface
     * stays free of implementation-specific types; numeric values are
     * encoded by the implementation. */
    virtual std::map<std::string, std::string> get_caps() = 0;

    /* Apply a set of cap updates. Returns true if every requested key
     * was recognized and accepted, false if any key is unknown or any
     * value is out of range for this hardware. Implementations should
     * treat the call as all-or-nothing; partial application is not
     * defined. */
    virtual bool set_caps(const std::map<std::string, std::string>& caps) = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_OSCILLATOR_HPP
