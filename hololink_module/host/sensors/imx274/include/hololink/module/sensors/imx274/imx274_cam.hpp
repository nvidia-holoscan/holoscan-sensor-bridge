/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_SENSORS_IMX274_CAM_HPP
#define HOLOLINK_MODULE_SENSORS_IMX274_CAM_HPP

#include <cstdint>
#include <memory>
#include <vector>

#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/i2c.hpp"
#include "hololink/module/i2c_lock.hpp"
#include "hololink/module/oscillator.hpp"

#include "hololink/module/csi_converter.hpp"

#include "hololink/module/sensors/imx274/imx274_mode.hpp"
#include "hololink/module/sensors/imx274/li_i2c_expander.hpp"

namespace hololink::module::sensors::imx274 {

/* IMX274 camera driver bound to the module V1 surface.
 *
 * Mirrors the Python `hololink_module.sensors.imx274.Imx274Cam`
 * class register-table flow but in C++. The constructor takes an
 * module `HololinkInterfaceV1` directly — no `DataChannel`
 * indirection — and uses `hololink->get_i2c(bus, address)` to reach
 * the camera + I2C expander. Two cameras share the I2C bus on an
 * HSB-Lite carrier; the per-camera `LII2CExpander` selects which
 * one the bus drives before each transaction. Construct with
 * `expander_configuration=0` for the first camera and `=1` for the
 * second. */
class Imx274Cam {
public:
    static constexpr uint32_t CAM_I2C_ADDRESS = 0b00011010;
    static constexpr uint32_t DEFAULT_CAM_I2C_BUS = 1;
    static constexpr uint32_t VERSION = 1;

    Imx274Cam(std::shared_ptr<HololinkInterfaceV1> hololink,
        std::shared_ptr<OscillatorInterfaceV1> oscillator,
        uint32_t i2c_bus = DEFAULT_CAM_I2C_BUS,
        uint32_t expander_configuration = 0);

    /* Convenience constructor that fetches HololinkInterface +
     * OscillatorInterface from the supplement-stamped metadata
     * (each resolves the supplement module from metadata through
     * the process-wide Adapter) and derives expander_configuration
     * from metadata too — prefers metadata["expander_configuration"]
     * when present, falls back to metadata["data_plane"] otherwise
     * (the HSB-Lite carrier conflates the two so the data_plane
     * index doubles as the I2C-expander output selector). */
    explicit Imx274Cam(const EnumerationMetadata& metadata,
        uint32_t i2c_bus = DEFAULT_CAM_I2C_BUS);

    /* Stamp an explicit "expander_configuration" entry into the
     * metadata so the constructor above picks it up instead of
     * falling back to "data_plane". Application code that wants to
     * override the LI I2C expander output goes through this rather
     * than mutating EnumerationMetadata fields by string key — that
     * keeps the metadata layout an Imx274Cam concern, not an
     * application concern. */
    static void use_expander_configuration(EnumerationMetadata& metadata,
        uint32_t expander_configuration);

    // Clock + lifecycle.
    void setup_clock(const std::vector<std::vector<uint8_t>>& clock_profile);
    void configure(Imx274_Mode mode);
    void start();
    void stop();
    uint32_t get_version() const { return VERSION; }

    // Per-register I/O. Each transaction acquires the per-board I2C
    // lock so two cameras sharing the bus don't tangle their packets.
    uint8_t get_register(uint16_t reg);
    void set_register(uint16_t reg, uint8_t value);

    // Mode + format helpers.
    void configure_camera(Imx274_Mode mode);
    void set_mode(Imx274_Mode mode);
    void set_exposure_reg(uint16_t value = 0x000C);
    void set_digital_gain_reg(uint16_t value = 0x0000);
    void set_analog_gain_reg(uint16_t value = 0x000C);

    // Format accessors used by the receiver / converter operators.
    PixelFormat pixel_format() const { return pixel_format_; }
    BayerFormat bayer_format() const { return BayerFormat::RGGB; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }

    /* Train the CSI converter that interprets the received image data.
     * Computes the IMX274's start-byte / line-bytes geometry from the
     * current mode (using the converter's receiver-geometry helpers) and
     * feeds it to the converter. Takes the module-native `CsiConverterV1`;
     * an application bridges its concrete converter (e.g. the legacy
     * `CsiToBayerOp`) to this interface at the application layer. */
    void configure_converter(std::shared_ptr<CsiConverterV1> converter);

    // Test pattern helpers (matches Python's API).
    void test_pattern_disable();
    void test_pattern_enable(uint8_t pattern);

private:
    std::shared_ptr<HololinkInterfaceV1> hololink_;
    std::shared_ptr<OscillatorInterfaceV1> oscillator_;
    std::shared_ptr<I2cInterfaceV1> i2c_;
    std::shared_ptr<LII2CExpander> i2c_expander_;
    uint8_t i2c_expander_configuration_;
    uint32_t i2c_bus_;

    Imx274_Mode mode_ = Imx274_Mode::Unknown;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    PixelFormat pixel_format_ = PixelFormat::RAW_10;
};

} // namespace hololink::module::sensors::imx274

#endif // HOLOLINK_MODULE_SENSORS_IMX274_CAM_HPP
