/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION
 * & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SENSORS_ADCAM_LIB_HPP
#define SENSORS_ADCAM_LIB_HPP

#define GET_MASTER_CHIP_ID_CMD 0x0112
#define GET_SLAVE_CHIP_ID_CMD 0x0116
#define GET_IMAGER_STATUS_CMD 0x0020
#define GET_MASTER_FIRMWARE_COMMAND 0x01
#define GET_SLAVE_FIRMWARE_COMMAND 0x04
#define SET_SWITCH_TO_BURST_MODE 0x0019

#define STREAM_ON_CMD 0x00AD
#define STREAM_ON_VAL 0x00C5

#define STREAM_OFF_CMD 0x000C
#define STREAM_OFF_VAL 0x0002

#define ENABLE_VAL 0x0001

#define SET_FRAME_RATE_REG 0x0022
#define SET_FRAME_RATE_VAL_30 0x001E

#define MIPI_CLK_CONTINUOUS_CMD 0x00A9
#define MIPI_OUTPUT_SPEED_CMD 0x0031
#define ADSD3500_CMD_GET_CHIP_INFO 0x0032
#define DESKEW_ENABLE_CMD 0x00AB
#define GET_IMAGER_ERROR_CMD 0x0038
#define GET_MIPI_CLK_CONTINUOUS_CMD 0x00AA

#define MIPI_SPEED_2_5_GBPS 0x0001
#define MIPI_SPEED_2_0_GBPS 0x0002
#define MIPI_SPEED_1_5_GBPS 0x0003
#define MIPI_SPEED_1GBPS 0x0004

#define LOW_BYTE_MASK 0x00FF

#include <cstdint>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <string>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/common/tools.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/logging.hpp>
#include <hololink/operators/csi_to_bayer/csi_to_bayer.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>
#include <hololink/operators/roce_receiver/roce_receiver_op.hpp>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/bayer_demosaic/bayer_demosaic.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

namespace hololink::sensors {

// ---------------------------------------------------------------------------
// Imager type values returned by ADSD3500 register 0x0032 (GET_CHIP_INFO).
// resp[0] (bits [15:8]) contains the raw imager type code.
// ---------------------------------------------------------------------------
enum class AdcamImagerType : uint16_t {
    ADSD3100 = 1,
    ADTF3066 = 2,
};

// ---------------------------------------------------------------------------
// QMP capture mode ? frame geometry and Set Imager Mode parameters.
//
// Word 2 (0xYYYY) bit layout for the ADSD3500 Set Imager Mode command:
//   Bit  0     : depth_enable        (1 = depth output on)
//   Bit  1     : data_interleaving   (always 1 in these modes)
//   Bit  2     : ab_enable           (always 1 in these modes)
//   Bit  3     : ab_averaging        (bool: 0 or 1)
//   Bits [6:4] : phase_depth_bits    (register = 6 - enum; min=0, max=6)
//                   0?0-bit  2?8-bit  3?10-bit  4?12-bit  5?14-bit  6?16-bit
//   Bits [9:7] : ab_bits             (register = 6 - enum; min=0, max=6)
//                   0?0-bit  2?8-bit  3?10-bit  4?12-bit  5?14-bit  6?16-bit
//   Bits [11:10]: confidence_bits    (enum direct; min=0, max=2)
//                   0?off(0-bit)  1?4-bit  2?8-bit
//   Bits [13:12]: output_mipi        (number of MIPI lanes: 0, 1, or 2)
// ---------------------------------------------------------------------------
struct AdcamModeConfig {
    int mode_number; // capture mode index
    // MIPI frame dimensions — used for CSI converter configuration.
    int width;  // RAW_8 bytes per MIPI line
    int height; // number of MIPI lines
    // Actual image pixel dimensions — used by unpack_kernel.
    // For QMP modes: pixel_width = width/5, pixel_height = height (no padding).
    // For MP  modes: pixel_width * pixel_height * 5 < width * height
    //                (MIPI frame has trailing zero-padding bytes).
    int pixel_width;  // actual image pixels per row
    int pixel_height; // actual image pixel rows
    // phase_depth_bits: min=0, max=6
    //   0=0-bit, 2=8-bit, 3=10-bit, 4=12-bit, 5=14-bit, 6=16-bit
    int phase_depth_bits;
    // ab_bits: min=0, max=6
    //   0=0-bit, 2=8-bit, 3=10-bit, 4=12-bit, 5=14-bit, 6=16-bit
    int ab_bits;
    // confidence_bits: min=0, max=2
    //   0=off(0-bit), 1=4-bit, 2=8-bit
    int confidence_bits;
    int ab_averaging; // bool: 0=off, 1=on
    int depth_enable; // bool: 0=off, 1=on
    int output_mipi;  // MIPI lane count (0, 1, or 2)
};

// Build the 0xYYYY word for the Set Imager Mode register from an AdcamModeConfig.
// data_interleaving (bit 1) and ab_enable (bit 2) are always asserted.
constexpr uint16_t adcam_make_mode_settings(const AdcamModeConfig &cfg) {
    uint16_t w = 0;
    w |= static_cast<uint16_t>((cfg.depth_enable & 0x1) << 0);
    w |= static_cast<uint16_t>(1 << 1); // data_interleaving
    w |= static_cast<uint16_t>(1 << 2); // ab_enable
    w |= static_cast<uint16_t>((cfg.ab_averaging & 0x1) << 3);
    w |= static_cast<uint16_t>(((6 - cfg.phase_depth_bits) & 0x7) << 4);
    w |= static_cast<uint16_t>(((6 - cfg.ab_bits) & 0x7) << 7);
    w |= static_cast<uint16_t>((cfg.confidence_bits & 0x3) << 10);
    w |= static_cast<uint16_t>((cfg.output_mipi & 0x3) << 12);
    return w;
}

// Standard capture modes for ADSD3100.
// Each entry maps a mode index to its MIPI frame geometry, pixel dimensions,
// and imager settings used to build the Set Imager Mode register word.
//
// Column layout:
//   { mode_number, mipi_w, mipi_h,  px_w,  px_h,  phase_depth_bits, ab_bits, confidence_bits, ab_averaging, depth_enable, output_mipi }
//
// Mode types:
//   MP  (Mega Pixel)       : 1024×1024 pixels, MIPI frame 3072×1707, requires 1.5 Gbps
//   QMP (Quarter Mega Pixel):  512×512  pixels, MIPI frame 2560×512,  requires 1 Gbps
//
// Frame layout (5 bytes/pixel):
//   Sub-frame 1 (SF1) = depth + confidence (3 B/pixel)
//   Sub-frame 2 (SF2) = active brightness  (2 B/pixel)
//
constexpr AdcamModeConfig adsd3100_standardModes[] = {
    // ---- MP modes (1024×1024, 2 Gbps MIPI) ----
    /* Mode 0 */ {0, 3072, 1707, 1024, 1024, 6, 6, 2, 0, 1,
                  2},                              // depth+AB+conf, no AB avg
                                                   /* Mode 1 */
    {1, 3072, 1707, 1024, 1024, 6, 6, 2, 0, 1, 2}, // depth+AB+conf, no AB avg

    // ---- QMP modes (512×512, 1 Gbps MIPI) ----
    /* Mode 2 */
    {2, 2560, 512, 512, 512, 6, 6, 2, 1, 1, 2}, // depth+AB+conf, AB avg on
                                                /* Mode 3 */
    {3, 2560, 512, 512, 512, 6, 6, 2, 1, 1, 2}, // depth+AB+conf, AB avg on
                                                /* Mode 5 */
    {5, 2560, 512, 512, 512, 6, 6, 2, 1, 1, 2}, // depth+AB+conf, AB avg on
                                                /* Mode 6 */
    {6, 2560, 512, 512, 512, 6, 6, 2, 1, 1, 2}, // depth+AB+conf, AB avg on
};

// Standard capture modes for ADTF3066.
// Entries are looked up by mode_number using adcam_find_mode() — the array is
// not contiguous (some mode numbers are absent) so direct indexing is not used.
//
// Column layout:
//   { mode_number, mipi_w, mipi_h,  px_w,  px_h,  phase_depth_bits, ab_bits, confidence_bits, ab_averaging, depth_enable, output_mipi }
//
// Mode types:
//   VGA  (Video Graphics Array) :  512×640 pixels, MIPI frame 2560×640,  1 Gbps  ? modes 0,1,7
//   QVGA (Quarter VGA)          :  256×320 pixels, MIPI frame 1280×320,  1 Gbps  ? modes 2,3,5,6,8
//
// Frame layout (5 bytes/pixel):
//   Sub-frame 1 (SF1) = depth + confidence (3 B/pixel)
//   Sub-frame 2 (SF2) = active brightness  (2 B/pixel)
//
// Imager settings (all modes): depth+AB+conf, 16-bit depth, 16-bit AB, 8-bit conf, AB avg on
//
constexpr AdcamModeConfig adtf3066_standardModes[] = {
    // ---- VGA modes (512×640, 1 Gbps MIPI) ----
    /* Mode 0 */ {0, 2560, 640, 512, 640, 6, 6, 2, 1, 1,
                  2}, // VGA   depth+AB+conf, AB avg on
                      /* Mode 1 */
    {1, 2560, 640, 512, 640, 6, 6, 2, 1, 1,
     2}, // VGA   depth+AB+conf, AB avg on
         /* Mode 7 */
    {7, 2560, 640, 512, 640, 6, 6, 2, 1, 1,
     2}, // VGA   depth+AB+conf, AB avg on

    // ---- QVGA modes (256×320, 1 Gbps MIPI) ----
    /* Mode 3 */
    {3, 1280, 320, 256, 320, 6, 6, 2, 1, 1,
     2}, // QVGA  depth+AB+conf, AB avg on
         /* Mode 6 */
    {6, 1280, 320, 256, 320, 6, 6, 2, 1, 1,
     2}, // QVGA  depth+AB+conf, AB avg on
         /* Mode 8 */
    {8, 1280, 320, 256, 320, 6, 6, 2, 1, 1,
     2}, // QVGA  depth+AB+conf, AB avg on
};

// Find the AdcamModeConfig entry with the given mode_number in a fixed-size table.
// Returns nullptr if no entry with that mode_number exists.
template <size_t N>
inline const AdcamModeConfig *adcam_find_mode(const AdcamModeConfig (&table)[N],
                                              int mode_number) {
    for (size_t i = 0; i < N; ++i) {
        if (table[i].mode_number == mode_number) {
            return &table[i];
        }
    }
    return nullptr;
}

enum class I2CExpanderOutputEN : uint8_t {
  OUTPUT_1 = 0b0001,
  OUTPUT_2 = 0b0010,
  OUTPUT_3 = 0b0100,
  OUTPUT_4 = 0b1000,
  DEFAULT  = 0b0000
};

class ADII2CExpander {
 public:
  ADII2CExpander(std::shared_ptr<hololink::Hololink> hololink_,
                 uint32_t i2c_bus,
                 uint32_t expander_addr);

  void configure(I2CExpanderOutputEN output_en = I2CExpanderOutputEN::DEFAULT);

  void set_register(uint16_t register_,
                    uint32_t value,
                    std::optional<Timeout> timeout = std::nullopt);

 private:
  std::shared_ptr<Hololink::I2c> i2c_;
  uint32_t expander_addr_;
};

class ADIGPIOCtl {
 public:
  ADIGPIOCtl(std::shared_ptr<hololink::Hololink> hololink_,
             hololink::Metadata& channel_metadata,
             uint32_t pin);

  void configure_reset_low(uint32_t pin);
  void configure_reset_high(uint32_t pin);
  bool wait_for_low_and_set_high_profile(uint32_t pin);

 private:
  std::shared_ptr<Hololink::GPIO> gpio_;
  uint32_t reset_pin_;
};

/// ADCAM (ADTF3175 / ADSD3500 controller)
class Adcam {
 public:
  static constexpr uint8_t ADCAM_I2C_BUS_ADDRESS        = 0x38;
  static constexpr uint8_t EXPANDER_0_I2C_BUS_ADDRESS   = 0x68;
  static constexpr uint8_t EXPANDER_1_I2C_BUS_ADDRESS   = 0x58;

  Adcam(std::shared_ptr<hololink::DataChannel> hololink_channel,
        uint32_t hololink_i2c_controller_address,
        hololink::Metadata& channel_metadata,
        uint32_t adcam_mode,
        uint32_t num_planes,
        uint32_t tof_fps,                
        uint32_t reset_pin,
	uint32_t metadata_sz, 
	uint16_t mipi_lane_speed);

  ~Adcam();

  // ---- High-level ops ----
  std::vector<uint8_t> get_version();
  void set_mipi();
  void set_mode();
  void read_nvm_config();
  void get_status();
  void get_only_status();
    void get_imager_type_and_ccb_version();

  int probe_adcam_adtf3175();
  std::vector<uint8_t> force_stop_burst_mode();
    bool switch_from_standard_to_burst();
    bool switch_from_burst_to_standard();
    std::vector<uint8_t>
    get_fw_version(uint8_t cmd = GET_MASTER_FIRMWARE_COMMAND);
    std::vector<uint8_t>
    get_fw_version_burst_mode(uint8_t cmd = GET_MASTER_FIRMWARE_COMMAND);
  void get_chip_status();

  void stream_on();
  void stream_off();

    bool get_ChipID(uint16_t cmd = GET_MASTER_CHIP_ID_CMD);
  void get_Status();
  std::vector<uint8_t> get_ClockContinuousMode();

  void adcam_reset_power_on();
    void adcam_hard_reset();

  void profile_fpga_perf(uint32_t pin);

  // ---- Low-level register ops ----
    bool set_register(uint16_t reg, uint8_t value);

    bool set_register16_no_response(uint16_t *register_blob);

  std::vector<uint8_t> set_register16_response(uint16_t* register_blob,
                                               size_t resp_len);

  std::vector<uint8_t> set_register16_response(
      std::shared_ptr<hololink::sensors::Adcam> adcam_inst,
      uint16_t* register_blob,
      size_t resp_len);

  // ---- Helpers ----
  static std::vector<uint16_t> bytes_to_uint16_array_be(
      const std::vector<uint8_t>& data);

  static std::vector<uint8_t> registers_to_byte_array_be(uint64_t reg_blob);
  static std::vector<uint16_t> format_registers_be(uint64_t reg_blob);

  void configure_converter(
      std::shared_ptr<hololink::csi::CsiConverter> converter);

  uint32_t get_width();
  uint32_t get_height();
  uint32_t get_abs_height();
  uint32_t get_abs_width();
  uint32_t get_mode();
  uint32_t get_numPlane();
  uint32_t get_pixel_size_bytes();
  uint32_t get_pixel_height(); // actual image pixel rows  
  uint16_t get_imager_type(); // detected imager type (ADSD3100 = 1 , ADTF3066 = 2)
  void start();
  void stop();

 private:
  static void append_u16_be(std::vector<uint8_t>& out, uint16_t v);

  void i2c_write_read(const std::vector<uint8_t>& write_bytes,
                      size_t read_byte_count,
                      std::vector<uint8_t>* out_reply);

  void i2c_write_read(
      std::shared_ptr<hololink::sensors::Adcam> adcam_inst,
      const std::vector<uint8_t>& write_bytes,
      size_t read_byte_count,
      std::vector<uint8_t>* out_reply);

 private:
  std::shared_ptr<hololink::Hololink> hololink_{nullptr};
  std::shared_ptr<Hololink::I2c> i2c_{nullptr};

  int width_{512};
  int height_{512};
  int abs_width_{512};
  int abs_height_{512};  
  int pixel_width_{5}; //each pixel size in bytes
  //int pixel_width_{512}; // actual image pixels per row — set from mode table
  //int pixel_height_{512}; // actual image pixel rows     — set from mode table  
  int pixel_format_{0};
  int test_{0};

    uint16_t imager_type_{0}; // ADSD3100 =1 , ADTF3066 = 2
  uint32_t adcam_mode_{6};
  uint32_t num_planes_{3};
  uint32_t tof_fps_{30};      

  uint32_t reset_pin_{0};
  uint32_t metadata_sz_{0};
  uint16_t mipi_lane_speed_{MIPI_SPEED_2_5_GBPS};

  // expanders + gpio
  ADII2CExpander expander0_;
  ADII2CExpander expander1_;
  ADIGPIOCtl pf_gpio_;
};

}  // namespace hololink::sensors

#endif /* SENSORS_ADCAM_LIB_HPP */
