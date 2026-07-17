/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_SENSORS_VB1940_CAM_HPP
#define HOLOLINK_MODULE_SENSORS_VB1940_CAM_HPP

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/i2c.hpp"
#include "hololink/module/leopard_vb1940/leopard_vb1940.hpp"
#include "hololink/module/oscillator.hpp"
#include "hololink/module/vsync.hpp"

#include "hololink/module/csi_converter.hpp"

namespace hololink::module::sensors::vb1940 {

// The CSI format/converter types live in hololink::module::csi (they are not
// sensor types); alias them here so this driver's code names them unqualified.
using ::hololink::module::csi::BayerFormat;
using ::hololink::module::csi::CsiConverterV1;
using ::hololink::module::csi::PixelFormat;

enum class Vb1940_Mode : uint32_t {
    VB1940_MODE_2560X1984_30FPS = 0,
    VB1940_MODE_1920X1080_30FPS = 1,
    VB1940_MODE_2560X1984_30FPS_8BIT = 2,
    VB1940_MODE_2560X1984_60FPS = 3,
    Unknown = 4,
};

/* Stereo + intrinsic calibration data parsed from the on-camera
 * EEPROM. RGB and IR variants share this layout — fetch via
 * Vb1940Cam::get_rgb_calibration_data() / get_ir_calibration_data(). */
struct CalibrationData {
    enum class CameraIndex : uint8_t {
        kLeft = 0,
        kRight = 1,
    };
    // (fx, fy, cx, cy) for left, right.
    std::array<std::array<double, 4>, 2> intrinsic_parameters;
    // (k1, k2, p1, p2, k3, k4, k5, k6) for left, right.
    std::array<std::array<double, 8>, 2> distortion_parameters;
    // Rotation (Rx, Ry, Rz).
    std::array<double, 3> R;
    // Translation (Tx, Ty, Tz).
    std::array<double, 3> T;

    std::string to_string() const;
};

/* VB1940 camera driver bound to the module V1 surface.
 *
 * Ports the legacy `hololink::sensors::NativeVb1940Sensor` onto an
 * module `HololinkInterfaceV1` + `OscillatorInterfaceV1`. The
 * constructor does not need an HSB-Lite-style I2C expander; VB1940
 * boards (Leopard carrier) wire the camera bus directly. configure()
 * follows the legacy FSM dance: if the sensor is already in SW_STBY
 * it just applies the per-mode register table; otherwise it walks
 * SYSTEM_UP -> BOOT -> certificate / FWP load -> VT_PATCH -> mode
 * configuration. */
class Vb1940Cam {
public:
    /* Explicit form. `vsync` is the per-board source driving the
     * frame-start trigger line. Null is accepted; the ctor body
     * substitutes the module's NullVsync. */
    Vb1940Cam(std::shared_ptr<HololinkInterfaceV1> hololink,
        std::shared_ptr<OscillatorInterfaceV1> oscillator,
        std::shared_ptr<leopard_vb1940::LeopardVb1940InterfaceV1> leopard,
        int64_t sensor_number,
        uint32_t i2c_bus,
        std::shared_ptr<VsyncInterfaceV1> vsync = nullptr);

    /* Convenience constructor that fetches HololinkInterface +
     * OscillatorInterface + LeopardVb1940Interface from the
     * supplement-stamped metadata, reads `sensor_number` from
     * metadata["data_channel"], and reads the per-sensor `i2c_bus`
     * from metadata["i2c_bus"]. Both fields are stamped by
     * `Adapter::use_sensor(metadata, n)` — the application must
     * call it once per camera before constructing the Vb1940Cam.
     * The optional `vsync` argument carries the same per-board trigger
     * source the explicit form takes; the metadata-form does not
     * auto-fetch it — pass it explicitly when stereo synchronization
     * is desired (typically as the upcast return of
     * HololinkInterfaceV1::ptp_pps_output()). */
    explicit Vb1940Cam(const EnumerationMetadata& metadata,
        std::shared_ptr<VsyncInterfaceV1> vsync = nullptr);

    // Lifecycle.
    void configure(Vb1940_Mode mode);
    void start();
    void stop();
    uint32_t get_version() const { return 1; }

    // Exposure + analog gain tuning knobs.
    void set_exposure_reg(int32_t value = 0x0014);
    void set_analog_gain_reg(int32_t value = 0x00);

    // Format accessors used by the receiver / converter operators.
    PixelFormat pixel_format() const { return pixel_format_; }
    BayerFormat bayer_format() const { return BayerFormat::GBRG; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }

    /* Train the CSI converter that interprets the received image data.
     * VB1940 prepends 1 status line and appends 2 status lines around the
     * image data; this method computes the start-byte / line-bytes geometry
     * and feeds the module-native `CsiConverterV1`. An application bridges
     * its concrete converter (e.g. the legacy `CsiToBayerOp`) to this
     * interface at the application layer. */
    void configure_converter(std::shared_ptr<CsiConverterV1> converter);

    // Stereo + intrinsic calibration data parsed from EEPROM. RGB
    // calibration lives in pages 0-3; IR calibration in pages 4-7.
    CalibrationData get_rgb_calibration_data();
    CalibrationData get_ir_calibration_data();

private:
    // Register I/O with per-board I2C lock acquired via
    // hololink_->i2c_lock(...).
    uint8_t get_register(uint16_t reg);
    uint32_t get_register_32(uint16_t reg);
    void set_register_8(uint16_t reg, uint8_t value);
    void set_register_buffer(uint16_t reg, const std::vector<uint8_t>& data);

    void set_mode(Vb1940_Mode mode);
    void setup_clock();
    void configure_camera(Vb1940_Mode mode);

    // VB1940-specific bring-up helpers (run from configure() when the
    // sensor is not already in SW_STBY).
    void write_data_in_pages(uint32_t start_addr, const std::vector<uint8_t>& data);
    uint32_t get_device_id();
    void status_check();
    void do_secure_boot();
    void write_certificate();
    void write_fw();
    void write_vt_patch();

    // Apply a register-table sequence (writes + waits) over I2C.
    void apply_register_settings(
        const std::vector<std::pair<uint16_t, uint8_t>>& settings);

    // EEPROM access + calibration parsing.
    std::vector<uint8_t> read_eeprom_page(uint32_t page_num);
    CalibrationData parse_calibration_data(const std::vector<double>& data);
    CalibrationData read_calibration_from_eeprom(bool rgb);

    std::shared_ptr<HololinkInterfaceV1> hololink_;
    std::shared_ptr<OscillatorInterfaceV1> oscillator_;
    std::shared_ptr<leopard_vb1940::LeopardVb1940InterfaceV1> leopard_;
    std::shared_ptr<I2cInterfaceV1> i2c_;
    std::shared_ptr<I2cInterfaceV1> eeprom_i2c_;
    std::shared_ptr<VsyncInterfaceV1> vsync_;
    int64_t sensor_number_;
    uint32_t i2c_bus_;

    Vb1940_Mode mode_ = Vb1940_Mode::Unknown;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    PixelFormat pixel_format_ = PixelFormat::RAW_10;
};

} // namespace hololink::module::sensors::vb1940

#endif // HOLOLINK_MODULE_SENSORS_VB1940_CAM_HPP
