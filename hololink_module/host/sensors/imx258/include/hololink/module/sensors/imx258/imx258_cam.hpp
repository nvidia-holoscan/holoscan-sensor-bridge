/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Module-native IMX258 camera driver. Ports the Imx258Camera class of the
 * reference bring-up camera setup (which drove the FPGA I2C block through
 * direct BAR0 register writes) onto the module V1 surface, in the
 * shape of hololink_module/host/sensors/imx274/imx274_cam.hpp. */

#ifndef HOLOLINK_MODULE_SENSORS_IMX258_CAM_HPP
#define HOLOLINK_MODULE_SENSORS_IMX258_CAM_HPP

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/i2c.hpp"
#include "hololink/module/i2c_lock.hpp"

#include "hololink/module/csi_converter.hpp"

#include "hololink/module/sensors/imx258/imx258_mode.hpp"

namespace hololink::module::sensors::imx258 {

/* IMX258 camera driver bound to the module V1 surface.
 *
 * The camera sits on I2C bus 0 at 7-bit address 0x1A (unlike HSB's
 * CAM_I2C_BUS 1). Register I/O goes through `hololink->get_i2c(bus,
 * address)` under the per-board I2C lock; there is no I2C expander and
 * no oscillator -- the sensor's reference clock is board-supplied. The
 * I2C controller's 400 kHz timing is programmed by the hsb_pcie module
 * (in its start()), not by this driver. The driver is pure I2C: the
 * receiver data path is the application's concern.
 *
 * Deviation from the reference: with `verify_registers` the legacy
 * Imx258Camera only warned about readback mismatches; this driver logs
 * them and then throws from configure() so a misconfigured sensor is
 * caught before streaming starts. */
class Imx258Cam {
public:
    static constexpr uint32_t CAM_I2C_ADDRESS = IMX258_I2C_ADDRESS;
    static constexpr uint32_t VERSION = 1;

    Imx258Cam(std::shared_ptr<HololinkInterfaceV1> hololink,
        uint32_t i2c_bus = DEFAULT_CAM_I2C_BUS);

    /* Convenience constructor that fetches the HololinkInterface from the
     * supplement-stamped metadata (resolving the module from metadata
     * through the process-wide Adapter). */
    explicit Imx258Cam(const EnumerationMetadata& metadata,
        uint32_t i2c_bus = DEFAULT_CAM_I2C_BUS);

    /* Lifecycle. configure() checks the model ID, then walks the mode's
     * register table and leaves the sensor programmed but not streaming;
     * with `verify_registers` every register except REG_MODE_SELECT is
     * read back, mismatches are logged and, if any, configure throws at
     * the end (stricter than the ported reference, which only warned).
     * start()/stop() only toggle REG_MODE_SELECT so streaming begins
     * after the receiver is armed. */
    void configure(Imx258_Mode mode, bool verify_registers = false);
    void start();
    void stop();
    uint32_t get_version() const { return VERSION; }

    // Per-register I/O. Each transaction acquires the per-board I2C
    // lock. Register addresses are 16 bits, sent big-endian.
    uint8_t get_register(uint16_t reg);
    void set_register(uint16_t reg, uint8_t value);

    /* {REG_MODEL_ID_HI, REG_MODEL_ID_LO}; MODEL_ID (0x0258) when the
     * sensor answers. */
    uint16_t model_id();

    /* What the sensor itself says about streaming, independent of the FPGA:
     * REG_MODE_SELECT as latched and REG_FRAME_COUNT sampled twice
     * `sample_interval` apart. `advancing` is true when the counter moved,
     * i.e. frames are leaving the sensor over CSI whether or not anything
     * receives them. Diagnostic; costs three I2C reads. */
    struct StreamingStatus {
        uint8_t mode_select { 0 };
        uint8_t frame_count_before { 0 };
        uint8_t frame_count_after { 0 };
        bool advancing { false };
    };
    StreamingStatus streaming_status(std::chrono::milliseconds sample_interval = std::chrono::milliseconds(100));

    // Mode + format helpers.
    void set_mode(Imx258_Mode mode);
    void set_exposure_reg(uint16_t value);
    void set_analog_gain_reg(uint16_t value);

    // Format accessors used by the receiver / converter operators.
    PixelFormat pixel_format() const { return pixel_format_; }
    BayerFormat bayer_format() const { return BayerFormat::RGGB; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    uint32_t framerate() const { return framerate_; }
    uint32_t raw_start_byte() const { return RAW10_START_BYTE; }
    const char* frame_format() const { return "raw10"; }

    /* Train the CSI converter that interprets the received image data.
     * Computes the IMX258's start-byte / line-bytes geometry from the
     * current mode (using the converter's receiver-geometry helpers) and
     * feeds it to the converter. Takes the module-native `CsiConverterV1`;
     * an application bridges its concrete converter (e.g. the legacy
     * `CsiToBayerOp`) to this interface at the application layer. */
    void configure_converter(std::shared_ptr<CsiConverterV1> converter);

    /* Geometry-only core of the above for a RAW10 image of `width` x
     * `height`: start = receiver_start_byte() + RAW10_START_BYTE, line =
     * received_line_bytes(transmitted_line_bytes(RAW_10, width)). Split
     * out so the arithmetic can be exercised without a live board (the
     * constructor needs a registered I2C service). */
    static void configure_converter(CsiConverterV1& converter,
        uint32_t width, uint32_t height);

private:
    std::shared_ptr<HololinkInterfaceV1> hololink_;
    std::shared_ptr<I2cInterfaceV1> i2c_;
    uint32_t i2c_bus_;

    Imx258_Mode mode_ = Imx258_Mode::Unknown;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t framerate_ = 0;
    PixelFormat pixel_format_ = PixelFormat::RAW_10;
};

} // namespace hololink::module::sensors::imx258

#endif // HOLOLINK_MODULE_SENSORS_IMX258_CAM_HPP
