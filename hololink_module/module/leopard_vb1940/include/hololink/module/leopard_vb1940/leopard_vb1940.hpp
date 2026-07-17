/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_LEOPARD_VB1940_LEOPARD_VB1940_HPP
#define HOLOLINK_MODULE_LEOPARD_VB1940_LEOPARD_VB1940_HPP

#include <cstdint>
#include <string>

#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/module.hpp"
#include "hololink/module/service.hpp"

namespace hololink::module::leopard_vb1940 {

/* Leopard VB1940 board-specific surface. One supplement instance per
 * board, keyed in the locator by "serial=<serial_number>". The
 * metadata-form get_service inherited from ConfigurableService caches
 * the per-board impl and runs configure(metadata) so the impl resolves
 * its underlying HololinkInterface lazily.
 *
 * Board-level primitives the sensor driver (Vb1940Cam) reaches for
 * during bring-up. Each primitive is idempotent (first call does the
 * work, later calls validate compatibility, board reset clears the
 * cache) so the camera driver can invoke it from configure() without
 * coordinating with other channels on the same board. */
class LeopardVb1940InterfaceV1 : public ConfigurableService<LeopardVb1940InterfaceV1> {
public:
    static constexpr const char* type_id = "leopard_vb1940.v1";

    /* Per-board instance_id derivation used by the metadata-form
     * get_service. Throws if the metadata is missing 'serial_number'. */
    static std::string locator_id(const EnumerationMetadata& metadata)
    {
        return "serial=" + metadata.get<std::string>("serial_number");
    }

    virtual ~LeopardVb1940InterfaceV1() = default;

    /* Record that `sensor_number` is expected to be enabled on this
     * board. Does not touch hardware — purely updates the supplement's
     * `expected_sensors` mask. Idempotent (re-expecting an
     * already-expected sensor is a no-op); throws std::runtime_error
     * if sensor_number is out of range.
     *
     * Vb1940Cam::configure(mode) calls this so the supplement has the
     * full expected-sensors set before any camera's start() commits
     * the cumulative mask to the FPGA. Without this two-phase split
     * the first camera to start would write a partial mask
     * (e.g. 0x1 for sensor 0 alone), and we can't assume the
     * hardware accepts partial-mask writes — only the legacy-known
     * full atomic transition (0x8: 0x0 -> 0x3 for stereo) is safe. */
    virtual void expect_sensor(int64_t sensor_number) = 0;

    /* Bring `sensor_number` out of reset by committing the full
     * `expected_sensors` mask to the sensor-reset register. The
     * caller (Vb1940Cam::start) is responsible for having previously
     * called expect_sensor(sensor_number) on this same supplement.
     * Idempotent:
     *   - First call after a fresh board (or after a board reset
     *     cleared the cache): write the sensor-reset register to 0
     *     defensively, then write the full expected-sensors mask,
     *     then sleep ~100 ms so the sensors' PLLs stabilise.
     *   - Subsequent calls when the hardware mask already matches the
     *     expected mask (i.e. another camera on this board already
     *     committed): no-op.
     *   - Subsequent calls when the expected set has grown since the
     *     last commit (a new camera was added between commits):
     *     write the updated cumulative expected mask without zeroing
     *     first, so previously-released sensors stay released. Avoid
     *     this path when possible by funnelling all expect_sensor
     *     calls before any enable_sensor call.
     * Throws std::runtime_error if `sensor_number` is not in the
     * expected set or out of range. Implementations register an
     * on_reset listener on the underlying Hololink so a board reset
     * clears the committed-mask cache (expected-sensors persists
     * across reset — it's the application's stated intent, not the
     * hardware's state). */
    virtual void enable_sensor(int64_t sensor_number) = 0;
};

} // namespace hololink::module::leopard_vb1940

#endif // HOLOLINK_MODULE_LEOPARD_VB1940_LEOPARD_VB1940_HPP
