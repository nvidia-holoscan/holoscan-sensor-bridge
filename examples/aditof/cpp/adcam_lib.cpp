/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION
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

#include "adcam_lib.hpp"

#include <algorithm>
#include <chrono>
#include <iterator>
#include <stdexcept>
#include <thread>
#include <vector>

#include <holoscan/logger/logger.hpp>

#include <hololink/core/serializer.hpp>
#include <hololink/core/timeout.hpp>

namespace hololink::sensors {
namespace {

// -----------------------------------------------------------------------------
// Internal helpers
// These helpers are file-local and do not change the public API.
// -----------------------------------------------------------------------------

/// Encode register_blob into a big-endian byte array.
///
/// Expected layout of register_blob:
///   register_blob[0] = number of 16-bit words that follow
///   register_blob[1..N] = payload words
///
/// Example:
///   {2, 0x0031, 0x0004}
/// becomes:
///   00 31 00 04
static std::vector<uint8_t> encode_register_blob(const uint16_t* register_blob) {
  if (register_blob == nullptr) {
    throw std::runtime_error("register_blob is null");
  }

  const size_t word_count = register_blob[0];
  std::vector<uint8_t> write_bytes(word_count * sizeof(uint16_t));

  core::Serializer serializer(write_bytes.data(), write_bytes.size());
  for (size_t i = 1; i <= word_count; ++i) {
    serializer.append_uint16_be(register_blob[i]);
  }

  return write_bytes;
}

/// Log first two bytes safely if available.
static void log_reply_prefix(const char* label, const std::vector<uint8_t>& reply) {
  if (reply.size() >= 2) {
    HOLOSCAN_LOG_INFO("{} bytes={} {} {}",
                      label,
                      reply.size(),
                      static_cast<uint32_t>(reply[0]),
                      static_cast<uint32_t>(reply[1]));
  } else {
    HOLOSCAN_LOG_INFO("{} bytes={}", label, reply.size());
  }
}

}  // namespace

// ============================================================================
// ADII2CExpander
// ============================================================================

ADII2CExpander::ADII2CExpander(std::shared_ptr<hololink::Hololink> hololink_,
                               uint32_t i2c_bus,
                               uint32_t expander_addr)
    : i2c_(hololink_->get_i2c(i2c_bus)), expander_addr_(expander_addr) {}

void ADII2CExpander::configure(I2CExpanderOutputEN output_en) {
  // Write a single byte that selects the enabled output bitmask.
  std::vector<uint8_t> write_bytes(1);
  core::Serializer serializer(write_bytes.data(), write_bytes.size());
  serializer.append_uint8(static_cast<uint8_t>(output_en));

  HOLOSCAN_LOG_DEBUG("ADCAM Expander Config");
  i2c_->i2c_transaction(expander_addr_, write_bytes, 0);
}

void ADII2CExpander::set_register(uint16_t register_,
                                  uint32_t value,
                                  std::optional<Timeout> timeout) {
  // Current implementation ignores the optional timeout to preserve
  // the original runtime behavior. The parameter remains for API compatibility.
  (void)timeout;

  // Format: [16-bit register address in BE] + [8-bit value]
  std::vector<uint8_t> write_bytes(3);
  core::Serializer serializer(write_bytes.data(), write_bytes.size());
  serializer.append_uint16_be(register_);
  serializer.append_uint8(static_cast<uint8_t>(value));

  HOLOSCAN_LOG_DEBUG("ADCAM Expander Config set addr=0x{:X} value=0x{:X}",
                     expander_addr_, value);

  i2c_->i2c_transaction(expander_addr_, write_bytes, 0);
}

// ============================================================================
// ADIGPIOCtl
// ============================================================================

ADIGPIOCtl::ADIGPIOCtl(std::shared_ptr<hololink::Hololink> hololink_,
                       hololink::Metadata& channel_metadata,
                       uint32_t pin)
    : reset_pin_(pin) {
  gpio_ = hololink_->get_gpio(channel_metadata);
}

void ADIGPIOCtl::configure_reset_low(uint32_t pin) {
  HOLOSCAN_LOG_DEBUG("Resetting ADCAM..set GPIO to LOW");
  gpio_->set_value(pin, gpio_->LOW);
}

void ADIGPIOCtl::configure_reset_high(uint32_t pin) {
  HOLOSCAN_LOG_DEBUG("Resetting ADCAM..set GPIO to HIGH");
  gpio_->set_value(pin, gpio_->HIGH);
}

bool ADIGPIOCtl::wait_for_low_and_set_high_profile(uint32_t pin) {
  // Tight polling loop used for timing experiments / profiling.
  constexpr int poll_interval_us = 1;         // busy-wait polling interval
  constexpr int timeout_us = 1'000'000;       // 1 second timeout

  int elapsed_us = 0;
  uint64_t loop_count = 0;

  HOLOSCAN_LOG_INFO("Profiling GPIO {} for transition", pin);

  auto t_start = std::chrono::high_resolution_clock::now();

  // Trigger the line high first (preserves original behavior).
  gpio_->set_value(pin, gpio_->HIGH);
  gpio_->set_value(pin, gpio_->LOW);
  gpio_->set_value(pin, gpio_->HIGH);

  while (elapsed_us < timeout_us) {
    ++loop_count;

    auto value = gpio_->get_value(pin);

    // Preserve original semantics: success path when sampled HIGH.
    if (value == gpio_->HIGH) {
      auto t_end = std::chrono::high_resolution_clock::now();

      // Restore line state after measurement.
      gpio_->set_value(pin, gpio_->LOW);

      auto total_us =
          std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
              .count();

      HOLOSCAN_LOG_INFO(
          "GPIO {} observed HIGH after {} us ({} loops). Restoring LOW.",
          pin,
          total_us,
          loop_count);

      return true;
    }

    // Busy-wait to avoid scheduler jitter from thread sleep.
    auto t_poll = std::chrono::high_resolution_clock::now();
    while (true) {
      auto now = std::chrono::high_resolution_clock::now();
      auto diff =
          std::chrono::duration_cast<std::chrono::microseconds>(now - t_poll)
              .count();
      if (diff >= poll_interval_us) {
        break;
      }
    }

    elapsed_us += poll_interval_us;
  }

  HOLOSCAN_LOG_WARN(
      "Timeout: GPIO {} did not reach target state within {} us ({} loops)",
      pin,
      timeout_us,
      loop_count);

  return false;
}

// ============================================================================
// Adcam
// ============================================================================

Adcam::Adcam(std::shared_ptr<hololink::DataChannel> hololink_channel,
             uint32_t hololink_i2c_controller_address,
             hololink::Metadata& channel_metadata,
             uint32_t adcam_mode,
             uint32_t num_planes,
             uint32_t tof_fps,                 
             uint32_t reset_pin,
             uint32_t metadata_sz,
             uint16_t mipi_lane_speed)
    : hololink_(hololink_channel->hololink()),
      // Preserve current behavior: use CAM_I2C_BUS here.
      // The constructor parameter is kept for API compatibility.
      i2c_(hololink_->get_i2c(hololink::CAM_I2C_BUS)),
      expander0_(hololink_,
                 hololink::CAM_I2C_BUS,
                 EXPANDER_0_I2C_BUS_ADDRESS),
      expander1_(hololink_,
                 hololink::CAM_I2C_BUS,
                 EXPANDER_1_I2C_BUS_ADDRESS),
      pf_gpio_(hololink_, channel_metadata, reset_pin),
      adcam_mode_(adcam_mode),
      num_planes_(num_planes),
      tof_fps_(tof_fps),
      reset_pin_(reset_pin),
      metadata_sz_(metadata_sz),
      mipi_lane_speed_(mipi_lane_speed) {
  (void)hololink_i2c_controller_address;

    // At construction the imager type is not yet known (requires I2C communication).
    // Search adsd3100_standardModes first, then adtf3066_standardModes, as a
    // best-effort initialisation. Call get_imager_type_and_ccb_version() after
    // hololink->start() to select the correct table and re-init geometry.
    const AdcamModeConfig *init_cfg =
        adcam_find_mode(adsd3100_standardModes, static_cast<int>(adcam_mode));
    if (!init_cfg) {
        init_cfg = adcam_find_mode(adtf3066_standardModes,
                                   static_cast<int>(adcam_mode));
    }
    if (!init_cfg) {
        throw std::runtime_error(
            fmt::format("Adcam: unsupported adcam_mode {}", adcam_mode));
    }
    width_ = init_cfg->width;
    height_ = init_cfg->height;
    //pixel_width_ = init_cfg->pixel_width;
    //pixel_height_ = init_cfg->pixel_height;
    abs_width_ = init_cfg->pixel_width;
    abs_height_ = init_cfg->pixel_height;

    //HOLOSCAN_LOG_DEBUG("[ADCAM] Constructed mode={} mipi={}x{} pixels={}x{}",
    //                   adcam_mode, width_, height_, pixel_width_,
    //                   pixel_height_);
    HOLOSCAN_LOG_DEBUG("[ADCAM] Constructed mode={} mipi={}x{} pixels={}x{}",
                       adcam_mode, width_, height_, abs_width_,
                       abs_height_);

  if (!hololink_) {
    throw std::runtime_error("Adcam: hololink is null");
  }
  if (!i2c_) {
    throw std::runtime_error("Adcam: i2c is null");
  }
#if 0
  if (adcam_mode_ >= 0 && adcam_mode_ < 2)
  {
    //Read extra one row if we are reading confidence (FW8.0 implementation)
    if (num_planes_ > 2)
    {
      height_ = 1024 +1; // One eextra line 
      pixel_width_ = 5;
    }
    else{
      height_ = 1024; // One eextra line 
      pixel_width_ = 4;
    }
    abs_height_ = 1024;
    width_ = 1024;
    abs_width_ = 1024;    
  }
  else if (adcam_mode_ >= 2 && adcam_mode_ < 7)
  {
    height_ = 512;
    width_ = 512;
    abs_height_ = 512;
    abs_width_ = 512;
    pixel_width_ = 5;
  }
#else

    pixel_width_ = 1;
#endif
  test_ = 1;
}

Adcam::~Adcam() = default;

// -----------------------------------------------------------------------------
// Version / identification
// -----------------------------------------------------------------------------

std::vector<uint8_t> Adcam::get_version() {
  HOLOSCAN_LOG_DEBUG("Fetching Chip version");

    uint16_t reg[] = {1, GET_MASTER_CHIP_ID_CMD};
  auto resp = set_register16_response(reg, 2);
  HOLOSCAN_LOG_DEBUG("Version Response = {} bytes", resp.size());

    uint16_t reg1[] = {1, GET_IMAGER_STATUS_CMD};
  resp = set_register16_response(reg1, 2);
  HOLOSCAN_LOG_DEBUG("Status Response = {} bytes", resp.size());

  return resp;
}

// -----------------------------------------------------------------------------
// Byte / register conversion helpers
// -----------------------------------------------------------------------------

std::vector<uint16_t> Adcam::bytes_to_uint16_array_be(const std::vector<uint8_t>& data) {
  if (data.size() % 2 != 0) {
    throw std::runtime_error("Data length must be even for 16-bit conversion.");
  }

  std::vector<uint16_t> out;
  out.reserve(data.size() / 2);

  for (size_t i = 0; i < data.size(); i += 2) {
    uint16_t v = (uint16_t(data[i]) << 8) | uint16_t(data[i + 1]);
    out.push_back(v);
  }

  return out;
}

std::vector<uint8_t> Adcam::registers_to_byte_array_be(uint64_t reg_blob) {
  // Match Python behavior:
  // length = ceil(bit_length / 8), minimum 2 bytes, force even length.
  int bitlen = 0;
  if (reg_blob != 0) {
    bitlen = 64 - __builtin_clzll(reg_blob);
  }

  int length = (bitlen + 7) / 8;
  if (length == 0) {
    length = 2;
  }
  if (length % 2 != 0) {
    length += 1;
  }

  std::vector<uint8_t> out(length);
  for (int i = 0; i < length; ++i) {
    int shift = (length - 1 - i) * 8;
    out[i] = static_cast<uint8_t>((reg_blob >> shift) & 0xFF);
  }

  return out;
}

std::vector<uint16_t> Adcam::format_registers_be(uint64_t reg_blob) {
  return bytes_to_uint16_array_be(registers_to_byte_array_be(reg_blob));
}

void Adcam::append_u16_be(std::vector<uint8_t>& out, uint16_t v) {
  out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
  out.push_back(static_cast<uint8_t>(v & 0xFF));
}

// -----------------------------------------------------------------------------
// I2C wrappers
// Keep both overloads for API compatibility.
// -----------------------------------------------------------------------------

void Adcam::i2c_write_read(const std::vector<uint8_t>& write_bytes,
                           size_t read_byte_count,
                           std::vector<uint8_t>* out_reply) {
  auto timeout = std::make_shared<hololink::Timeout>(30.0, 0.2);

  if (!i2c_) {
    throw std::runtime_error("i2c is null");
  }

  auto reply = i2c_->i2c_transaction(ADCAM_I2C_BUS_ADDRESS,
                                     write_bytes,
                                     read_byte_count,
                                     timeout);

  if (out_reply) {
    *out_reply = std::move(reply);
  }
}

// -----------------------------------------------------------------------------
// Low-level register writes
// -----------------------------------------------------------------------------

bool Adcam::set_register(uint16_t reg, uint8_t value) {
    HOLOSCAN_LOG_DEBUG("set_register(reg=0x{:04X}, value=0x{:02X})", reg,
                       value);

  std::vector<uint8_t> write;
  write.reserve(3);
  append_u16_be(write, reg);
  write.push_back(value);

    try {
  i2c_write_read(write, 0, nullptr);
        return true;
    } catch (const std::exception &e) {
        holoscan::log_error("[ERROR] set_register failed: {}", e.what());
        return false;
    }
}

bool Adcam::set_register16_no_response(uint16_t *register_blob) {
  try {
    auto write_bytes = encode_register_blob(register_blob);

    HOLOSCAN_LOG_DEBUG("ADCAM REGISTER NORESPREQD total_size={}", write_bytes.size());
    i2c_write_read(write_bytes, 0, nullptr);
        return true;
  } catch (const std::exception& e) {
        holoscan::log_error("[ERROR] set_register16_no_response failed: {}",
                            e.what());
        return false;
  }
}

std::vector<uint8_t> Adcam::set_register16_response(uint16_t* register_blob,
                                                    size_t resp_len) {
  std::vector<uint8_t> reply;

  try {
    auto write_bytes = encode_register_blob(register_blob);

    HOLOSCAN_LOG_DEBUG("ADCAM REGISTER RESPREQD total_size={}", write_bytes.size());
    i2c_write_read(write_bytes, resp_len, &reply);

    HOLOSCAN_LOG_DEBUG("ADCAM REGISTER REPLY bytes={}", reply.size());
  } catch (const std::exception& e) {
    holoscan::log_error("[ERROR] set_register16_response failed: {}", e.what());
  }

  return reply;
}

// -----------------------------------------------------------------------------
// High-level configuration / status methods
// -----------------------------------------------------------------------------

void Adcam::set_mipi() {
  HOLOSCAN_LOG_DEBUG("Setting MIPI speed ADCAM INST {}", width_);

  // Check current chip status before changing link settings.
    uint16_t reg0[] = {1, GET_IMAGER_STATUS_CMD};
  HOLOSCAN_LOG_DEBUG("Fetching status ={}", test_);
  auto resp = set_register16_response(reg0, 2);

  log_reply_prefix("Chip Status", resp);

  // Set MIPI lane speed / mode value as in original implementation.
    uint16_t reg[] = {2, MIPI_OUTPUT_SPEED_CMD,
                      mipi_lane_speed_}; 
    if (!set_register16_no_response(reg)) {
        HOLOSCAN_LOG_ERROR("set_mipi: failed to set MIPI lane speed");
    }

  get_status();

  // Enable deskew
  HOLOSCAN_LOG_DEBUG("Enabling deskew");
    uint16_t reg1[] = {2, DESKEW_ENABLE_CMD, ENABLE_VAL};
    if (!set_register16_no_response(reg1)) {
        HOLOSCAN_LOG_ERROR("set_mipi: failed to enable deskew");
    }
}

void Adcam::set_mode() {
    // Build the two-word Set Imager Mode command:
    //   Word 1: 0xDAXX  — XX = imager mode number (0-10)
    //   Word 2: 0xYYYY  — dynamically computed from per-mode field configuration
    auto imager_enum = static_cast<AdcamImagerType>(imager_type_);
    const char *imager_str =
        (imager_enum == AdcamImagerType::ADSD3100)   ? "ADSD3100"
        : (imager_enum == AdcamImagerType::ADTF3066) ? "ADTF3066"
                                                     : "Unknown";
    const AdcamModeConfig *cfg = nullptr;
    switch (imager_enum) {
    case AdcamImagerType::ADSD3100:
        cfg = adcam_find_mode(adsd3100_standardModes,
                              static_cast<int>(adcam_mode_));
        break;
    case AdcamImagerType::ADTF3066:
        cfg = adcam_find_mode(adtf3066_standardModes,
                              static_cast<int>(adcam_mode_));
        break;
    default:
        throw std::runtime_error(fmt::format(
            "set_mode: unsupported imager type (raw={})", imager_type_));
    }
    if (!cfg) {
        throw std::runtime_error(
            fmt::format("set_mode: mode {} not found for imager {} (raw={})",
                        adcam_mode_, imager_str, imager_type_));
    }

    uint16_t mode_reg = static_cast<uint16_t>(0xDA00 | (adcam_mode_ & 0xFF));
    uint16_t mode_setting = adcam_make_mode_settings(*cfg);

    HOLOSCAN_LOG_INFO("Setting imager mode={} reg=0x{:04X} settings=0x{:04X}",
                      adcam_mode_, mode_reg, mode_setting);
#if 0                    
  // Original register sequence preserved.
  uint16_t reg[] = {2, 0xDA06, 0x280F};

  //insert the mode value
  reg[1] &= 0xFF00;
  reg[1] |=  (adcam_mode_ & 0xFF);

  if (adcam_mode_ < 2)
  {
    //reg[2] = 0x2007;
    reg[2] = num_planes_ > 2? 0x2807: 0x2007;
  }
  else
  {
    reg[2] = num_planes_ > 2? 0x280F: 0x2007;
  }
#else
  uint16_t reg[] = {2, mode_reg, mode_setting};

#endif
  if (!set_register16_no_response(reg)) {
      HOLOSCAN_LOG_ERROR("set_mode: failed to set imager mode {}",
                         adcam_mode_);
  }

  // Set FPS.
  uint16_t reg1[] = {2, SET_FRAME_RATE_REG, SET_FRAME_RATE_VAL_30};
  reg1[2] &=  (0xFF00);
  reg1[2] |=  (tof_fps_ & 0xFF);
  set_register16_no_response(reg1);  
}

void Adcam::read_nvm_config() {
  HOLOSCAN_LOG_DEBUG("Reading NVM Config");

  std::this_thread::sleep_for(std::chrono::seconds(1));

    uint16_t reg[] = {2, SET_SWITCH_TO_BURST_MODE, 0x0000};
    if (!set_register16_no_response(reg)) {
        HOLOSCAN_LOG_ERROR("read_nvm_config: failed to switch to burst mode");
    }

  // Multi-word command preserved exactly from original code.
  uint16_t reg1[] = {4, 0xAD00, 0x2C05, 0x0000, 0x0000};
    if (!set_register16_no_response(reg1)) {
        HOLOSCAN_LOG_ERROR("read_nvm_config: failed to send NVM read command");
    }

  // Keep original note: some Python register paths used >64-bit equivalents.
  holoscan::log_warn(
      "read_nvm_config: register blobs >64-bit need a byte-array API in C++ if extended further.");

  HOLOSCAN_LOG_DEBUG("Reading NVM Config done");
}

void Adcam::get_status() {
  HOLOSCAN_LOG_DEBUG("Fetching status");

    uint16_t reg[] = {1, GET_IMAGER_STATUS_CMD};
  auto resp = set_register16_response(reg, 2);
  log_reply_prefix("Chip Status", resp);

    uint16_t reg1[] = {1, GET_IMAGER_ERROR_CMD};
  resp = set_register16_response(reg1, 2);
  log_reply_prefix("0x0038 Status", resp);
}

void Adcam::get_only_status() {
  HOLOSCAN_LOG_DEBUG("Fetching status");

    uint16_t reg[] = {1, GET_IMAGER_STATUS_CMD};
  auto resp = set_register16_response(reg, 2);
  log_reply_prefix("Chip Status", resp);
}

void Adcam::get_imager_type_and_ccb_version() {
    // Write 0x0032 to trigger the read; response is 4 bytes.
    //   Bits  [7:0]  = CCB Version  (1 = Version 0, 2 = Version 1, 3 = Version 2, 4 = Version 3)
    //   Bits [15:8]  = Imager Type  (1 = ADSD3100,  2 = ADTF3066)
    // Response is a single 16-bit word: resp[0] = CCB Version, resp[1] = Imager Type.
    uint16_t reg[] = {1, ADSD3500_CMD_GET_CHIP_INFO};
    auto resp = set_register16_response(reg, 2);

    if (resp.size() < 2) {
        HOLOSCAN_LOG_ERROR("get_imager_type_and_ccb_version: incomplete "
                           "response bytes={} (expected 2)",
                           resp.size());
        return;
    }

    // Response is a single 16-bit word in big-endian byte order:
    //   resp[0] = bits [15:8] = Imager Type
    //   resp[1] = bits  [7:0] = CCB Version
    imager_type_ = resp[0] & LOW_BYTE_MASK;         // bits [15:8]
    uint16_t ccb_version = resp[1] & LOW_BYTE_MASK; // bits [7:0]

    const char *ccb_str = (ccb_version == 1)   ? "Version 0"
                          : (ccb_version == 2) ? "Version 1"
                          : (ccb_version == 3) ? "Version 2"
                          : (ccb_version == 4) ? "Version 3"
                                               : "Unknown";

    auto imager_enum = static_cast<AdcamImagerType>(imager_type_);
    const char *imager_str =
        (imager_enum == AdcamImagerType::ADSD3100)   ? "ADSD3100"
        : (imager_enum == AdcamImagerType::ADTF3066) ? "ADTF3066"
                                                     : "Unknown";

    HOLOSCAN_LOG_INFO("Imager Type: {} (raw={}), CCB Version: {} (raw={})",
                      imager_str, imager_type_, ccb_str, ccb_version);

    // Re-initialize frame geometry from the appropriate mode table.
    const AdcamModeConfig *mode_cfg = nullptr;

    switch (imager_enum) {
    case AdcamImagerType::ADSD3100:
        // ADSD3100: MP and QMP modes in adsd3100_standardModes.
        mode_cfg = adcam_find_mode(adsd3100_standardModes,
                                   static_cast<int>(adcam_mode_));
        break;
    case AdcamImagerType::ADTF3066:
        // ADTF3066: VGA and QVGA modes in adtf3066_standardModes.
        mode_cfg = adcam_find_mode(adtf3066_standardModes,
                                   static_cast<int>(adcam_mode_));
        break;
    default:
        HOLOSCAN_LOG_ERROR(
            "get_imager_type_and_ccb_version: unsupported imager type (raw={})",
            imager_type_);
        throw std::runtime_error(
            fmt::format("Unsupported imager type (raw={})", imager_type_));
    }

    if (!mode_cfg) {
        HOLOSCAN_LOG_ERROR(
            "get_imager_type_and_ccb_version: mode {} not found for "
            "imager {}; keeping current geometry",
            adcam_mode_, imager_str);
        return;
    }

    width_ = mode_cfg->width;
    height_ = mode_cfg->height;
#if 0
    pixel_width_ = mode_cfg->pixel_width;
    pixel_height_ = mode_cfg->pixel_height;

    HOLOSCAN_LOG_INFO(
        "Mode table selected: {} — mode={} mipi={}x{} pixel={}x{}", imager_str,
        adcam_mode_, width_, height_, pixel_width_, pixel_height_);
#else
    abs_width_ = mode_cfg->pixel_width;
    abs_height_ = mode_cfg->pixel_height;

    HOLOSCAN_LOG_INFO(
        "Mode table selected: {} — mode={} mipi={}x{} pixel={}x{}", imager_str,
        adcam_mode_, width_, height_, abs_width_, abs_height_);
#endif
}

int Adcam::probe_adcam_adtf3175() {
    uint16_t reg[] = {1, GET_MASTER_CHIP_ID_CMD};

  HOLOSCAN_LOG_INFO("probe_adcam_adtf3175 status = {}", test_);
  auto resp = set_register16_response(reg, 2);

  log_reply_prefix("probe_adcam_adtf3175", resp);

  // Preserve original ID check.
  if ((resp.size() >= 2) && (resp[0] == 0x59) && (resp[1] == 0x31)) {
    return 1;
  }

  return 0;
}

std::vector<uint8_t> Adcam::force_stop_burst_mode() {
  HOLOSCAN_LOG_DEBUG("Forcing burst mode off");

  uint16_t reg[] = {4, 0xAD00, 0x0010, 0x0000, 0x0000};
    if (!set_register16_no_response(reg)) {
        HOLOSCAN_LOG_ERROR("force_stop_burst_mode: failed to stop burst mode");
    }

    uint16_t reg1[] = {1, GET_IMAGER_STATUS_CMD};
  return set_register16_response(reg1, 2);
}

bool Adcam::switch_from_standard_to_burst() {
    uint16_t reg[] = {2, SET_SWITCH_TO_BURST_MODE, 0x0000};
    if (!set_register16_no_response(reg)) {
        HOLOSCAN_LOG_ERROR(
            "switch_from_standard_to_burst: failed to switch to burst mode");
        return false;
    }
    return true;
}

bool Adcam::switch_from_burst_to_standard() {
    uint16_t reg[] = {8,      0xAD00, 0x0010, 0x0000, 0x0000,
                      0x1000, 0x0000, 0x0100, 0x0000};
    if (!set_register16_no_response(reg)) {
        HOLOSCAN_LOG_ERROR(
            "switch_from_burst_to_standard: failed to switch to standard mode");
        return false;
    }
    return true;
}

std::vector<uint8_t> Adcam::get_fw_version(uint8_t cmd) {
    HOLOSCAN_LOG_DEBUG("Fetching version");

    switch_from_standard_to_burst();

    // Read firmware version — byte 12 of payload is the firmware command (0x01=master, 0x04=slave)
    uint16_t reg3[] = {8,      0xAD00, 0x2C05, 0x0000,
                       0x0000, 0x3100, 0x0000, static_cast<uint16_t>(cmd << 8),
                       0x0000};
    auto resp = set_register16_response(reg3, 44);
  HOLOSCAN_LOG_INFO("Firmware ID bytes={}", resp.size());

    switch_from_burst_to_standard();

  return resp;
}

std::vector<uint8_t> Adcam::get_fw_version_burst_mode(uint8_t cmd) {
    // Read firmware version while already in burst mode — no mode switching.
    // Call this variant when burst mode is managed externally by the caller.
    uint16_t reg3[] = {8,      0xAD00, 0x2C05, 0x0000,
                       0x0000, 0x3100, 0x0000, static_cast<uint16_t>(cmd << 8),
                       0x0000};
    auto resp = set_register16_response(reg3, 44);
    if (resp.size() < 44) {
        HOLOSCAN_LOG_ERROR("get_fw_version_burst_mode: incomplete response "
                           "bytes={} (expected 44)",
                           resp.size());
}
    return resp;
}

void Adcam::get_chip_status() { get_status(); }

// -----------------------------------------------------------------------------
// Stream control (legacy function names preserved)
// -----------------------------------------------------------------------------

void Adcam::stream_on() {
  HOLOSCAN_LOG_DEBUG("Setting Clock continuous mode in stream_on");

    uint16_t reg[] = {2, MIPI_CLK_CONTINUOUS_CMD, ENABLE_VAL};
    if (!set_register16_no_response(reg)) {
        HOLOSCAN_LOG_ERROR("stream_on: failed to set clock continuous mode");
    }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  HOLOSCAN_LOG_DEBUG("Turning ON Streaming");

    uint16_t reg1[] = {2, STREAM_ON_CMD, STREAM_ON_VAL};
    if (!set_register16_no_response(reg1)) {
        HOLOSCAN_LOG_ERROR("stream_on: failed to enable streaming");
    }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  get_status();
}

void Adcam::stream_off() {
  HOLOSCAN_LOG_DEBUG("Turning OFF Streaming");

    uint16_t reg[] = {2, STREAM_OFF_CMD, STREAM_OFF_VAL};
    if (!set_register16_no_response(reg)) {
        HOLOSCAN_LOG_ERROR("stream_off: failed to disable streaming");
    }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  get_status();
}

bool Adcam::get_ChipID(uint16_t cmd) {
    uint16_t reg[] = {1, cmd};
  auto resp = set_register16_response(reg, 2);

  if (resp.size() >= 2) {
        HOLOSCAN_LOG_INFO("Chip ID bytes={} ID {:02X} {:02X}", resp.size(),
                      static_cast<uint32_t>(resp[0]),
                      static_cast<uint32_t>(resp[1]));
        return true;
  } else {
    HOLOSCAN_LOG_INFO("Chip ID bytes={}", resp.size());
        return false;
  }
}

void Adcam::get_Status() { get_status(); }

std::vector<uint8_t> Adcam::get_ClockContinuousMode() {
    uint16_t reg[] = {1, GET_MIPI_CLK_CONTINUOUS_CMD};
  auto resp = set_register16_response(reg, 2);

  HOLOSCAN_LOG_DEBUG("Clock continuous mode bytes={}", resp.size());
  return resp;
}

// -----------------------------------------------------------------------------
// Reset / power sequencing
// -----------------------------------------------------------------------------

void Adcam::adcam_reset_power_on() {
  HOLOSCAN_LOG_DEBUG("Resetting ADCAM");
  HOLOSCAN_LOG_INFO("Resetting ADCAM");

  // Assert reset before changing expanders / rails.
  pf_gpio_.configure_reset_low(reset_pin_);

  expander0_.set_register(0x0, 0x0);
  expander1_.set_register(0x0, 0x0);

  // Brief LED / diagnostic sequence retained from original code.
  expander0_.set_register(0x02, 0x02);
  HOLOSCAN_LOG_INFO("Check DS1 LED - it should be ON.");
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  expander0_.set_register(0x0, 0x0);
  HOLOSCAN_LOG_INFO("DS1 LED turned off");

  // Power rail / expander sequencing preserved exactly.
  expander0_.set_register(0x20, 0x20);
  expander1_.set_register(0x9, 0x9);

  expander0_.set_register(0x20, 0x20);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  expander0_.set_register(0x20, 0x20);
  expander1_.set_register(0x9, 0x9);

  expander0_.set_register(0x21, 0x21);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  expander0_.set_register(0x23, 0x23);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  expander1_.set_register(0x79, 0x79);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // Release reset.
  pf_gpio_.configure_reset_high(reset_pin_);

    HOLOSCAN_LOG_INFO("booting up ADCAM, wait for 5 seconds");
    std::this_thread::sleep_for(std::chrono::seconds(5));
}

void Adcam::adcam_hard_reset() {
  HOLOSCAN_LOG_DEBUG("ADCAM - Making Reset LOW ONLY");
  pf_gpio_.configure_reset_low(reset_pin_);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  HOLOSCAN_LOG_DEBUG("ADCAM - Making Reset HIGH ONLY");
  pf_gpio_.configure_reset_high(reset_pin_);

    HOLOSCAN_LOG_INFO("Waiting 5 secs after reset");
    std::this_thread::sleep_for(std::chrono::seconds(5));
}

void Adcam::profile_fpga_perf(uint32_t pin) {

  HOLOSCAN_LOG_DEBUG("ADCAM - Profile FPGA/GPIO path");
  HOLOSCAN_LOG_DEBUG("ADCAM - Making Reset HIGH ONLY");

  pf_gpio_.configure_reset_low(pin);

  HOLOSCAN_LOG_INFO("Waiting 1 sec before GPIO profiling");
  std::this_thread::sleep_for(std::chrono::seconds(1));

  pf_gpio_.wait_for_low_and_set_high_profile(pin);
}

// -----------------------------------------------------------------------------
// CSI / image path configuration
// -----------------------------------------------------------------------------

void Adcam::configure_converter(
    std::shared_ptr<hololink::csi::CsiConverter> converter) {
  // Determine where valid pixel payload starts in each frame buffer.
  uint32_t start_byte = converter->receiver_start_byte();

   HOLOSCAN_LOG_INFO("receiver_start_byte = {}", start_byte);

  // RAW_8 pixel format is preserved from the original implementation.
  uint32_t transmitted_line_bytes =
      converter->transmitted_line_bytes(csi::PixelFormat::RAW_8, width_*pixel_width_);

  HOLOSCAN_LOG_INFO("transmitted_line_bytes = {}", transmitted_line_bytes);

  uint32_t received_line_bytes =
      converter->received_line_bytes(transmitted_line_bytes);

  HOLOSCAN_LOG_INFO("received_line_bytes = {}", received_line_bytes);
  uint32_t embedded_data_bytes = metadata_sz_;

  start_byte += converter->received_line_bytes(embedded_data_bytes);

  converter->configure(start_byte,
                       received_line_bytes,
                       width_*pixel_width_,
                       height_,
                       csi::PixelFormat::RAW_8);
}

uint32_t Adcam::get_width() {
  return width_;
}

uint32_t Adcam::get_height() {
  return height_;
}

uint32_t Adcam::get_abs_height() {
  return abs_height_;
}

uint32_t Adcam::get_abs_width() {
  return abs_width_;
}

uint32_t Adcam::get_mode() {
  return adcam_mode_;
}

uint32_t Adcam::get_pixel_size_bytes() {
  return pixel_width_;
}
uint32_t Adcam::get_pixel_height() { return abs_height_; }


uint32_t Adcam::get_numPlane() {
  return num_planes_;
}

// -----------------------------------------------------------------------------
// Generic start / stop API
// Keep these in addition to stream_on / stream_off.
// -----------------------------------------------------------------------------

void Adcam::start(void) {
  // Set and check clock continuous mode before enabling streaming.
  HOLOSCAN_LOG_DEBUG("Setting Clock continuous mode in start");

    uint16_t reg[] = {2, MIPI_CLK_CONTINUOUS_CMD, ENABLE_VAL};
    if (!set_register16_no_response(reg)) {
        HOLOSCAN_LOG_ERROR("start: failed to set clock continuous mode");
    }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  HOLOSCAN_LOG_INFO("Turning ON Streaming check the timestamp");

    // Original start path uses STREAM_ON_CMD / STREAM_ON_VAL.
    uint16_t reg_stream_mode_on[] = {2, STREAM_ON_CMD, STREAM_ON_VAL};
    if (!set_register16_no_response(reg_stream_mode_on)) {
        HOLOSCAN_LOG_ERROR("start: failed to enable streaming");
    }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  get_status();
}

void Adcam::stop(void) {
  HOLOSCAN_LOG_INFO("Turning OFF Streaming check the timestamp");

    // Original stop path uses STREAM_OFF_CMD / STREAM_OFF_VAL.
    uint16_t reg_stream_mode_off[] = {2, STREAM_OFF_CMD, STREAM_OFF_VAL};
    if (!set_register16_no_response(reg_stream_mode_off)) {
        HOLOSCAN_LOG_ERROR("stop: failed to disable streaming");
    }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  get_status();
}

}  // namespace hololink::sensors
