/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_SENSORS_IMX274_LI_I2C_EXPANDER_HPP
#define HOLOLINK_MODULE_SENSORS_IMX274_LI_I2C_EXPANDER_HPP

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "hololink/module/i2c.hpp"
#include "hololink/module/status.h"

namespace hololink::module::sensors::imx274 {

/* Output-enable masks for the LII2CExpander on the HSB-Lite carrier.
 * The bus carries a peripheral that multiplexes the I2C master across
 * up to four downstream output channels; the driver writes one of
 * these masks before each per-camera transaction. */
enum class LIExpanderOutputEN : uint8_t {
    DEFAULT = 0b0000,
    OUTPUT_1 = 0b0001, // first camera
    OUTPUT_2 = 0b0010, // second camera
    OUTPUT_3 = 0b0100,
    OUTPUT_4 = 0b1000,
};

/* I2C expander helper bound to a V1 `I2cInterfaceV1`. Constructed
 * with the I2c handle pointing at the expander's address; the
 * `configure(output_en)` call writes a one-byte selection mask to
 * the expander, gating the bus toward exactly one camera output. */
class LII2CExpander {
public:
    static constexpr uint32_t I2C_EXPANDER_ADDRESS = 0b01110000;

    explicit LII2CExpander(std::shared_ptr<I2cInterfaceV1> i2c)
        : i2c_(std::move(i2c))
    {
        if (!i2c_) {
            throw std::runtime_error("Invalid I2c pointer passed to LII2CExpander.");
        }
    }

    void configure(uint8_t output_en)
    {
        std::vector<uint8_t> write_bytes { static_cast<uint8_t>(output_en & 0xFFU) };
        std::vector<uint8_t> read_bytes;
        const hololink_module_status_t status = i2c_->i2c_transaction(
            I2C_EXPANDER_ADDRESS, write_bytes, read_bytes);
        if (status != HOLOLINK_MODULE_OK) {
            throw std::runtime_error(
                "While selecting LI I2C expander output: i2c_transaction failed");
        }
    }

private:
    std::shared_ptr<I2cInterfaceV1> i2c_;
};

} // namespace hololink::module::sensors::imx274

#endif // HOLOLINK_MODULE_SENSORS_IMX274_LI_I2C_EXPANDER_HPP
