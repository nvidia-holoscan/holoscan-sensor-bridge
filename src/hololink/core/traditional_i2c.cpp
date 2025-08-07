/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "hololink.hpp"

#include <cassert>
#include <cstdint>

#include "logging_internal.hpp"

namespace hololink {

// control flags
constexpr uint32_t TRADITIONAL_I2C_START = 0b0000'0000'0000'0001;
constexpr uint32_t TRADITIONAL_I2C_CORE_EN = 0b0000'0000'0000'0010;
constexpr uint32_t TRADITIONAL_I2C_DONE_CLEAR = 0b0000'0000'0001'0000;
constexpr uint32_t TRADITIONAL_I2C_BUSY = 0b0000'0001'0000'0000;
constexpr uint32_t TRADITIONAL_I2C_DONE = 0b0001'0000'0000'0000;

class TraditionalI2c
    : public Hololink::I2c {
public:
    TraditionalI2c(Hololink& hololink, uint32_t i2c_address)
        : hololink_(hololink)
        , reg_control_(i2c_address + 0)
        , reg_num_bytes_(i2c_address + 4)
        , reg_clk_ctrl_(i2c_address + 8)
        , reg_data_buffer_(i2c_address + 16)
    {
    }

    bool set_i2c_clock() override
    {
        // set the clock to 400KHz (fastmode) i2c speed once at init
        const uint32_t clock = 0b0000'0101;
        return hololink_.write_uint32(reg_clk_ctrl_, clock, Timeout::i2c_timeout());
    }

    std::vector<uint8_t> i2c_transaction(uint32_t peripheral_i2c_address,
        const std::vector<uint8_t>& write_bytes, uint32_t read_byte_count,
        const std::shared_ptr<Timeout>& in_timeout,
        bool /* ignore_nak */) override
    {
        HSB_LOG_DEBUG("i2c_transaction peripheral={:#x} len(write_bytes)={} read_byte_count={}",
            peripheral_i2c_address, write_bytes.size(), read_byte_count);
        if (peripheral_i2c_address >= 0x80) {
            throw std::runtime_error(
                fmt::format("Invalid peripheral_i2c_address \"{:#x}\", has to be less than 0x80",
                    peripheral_i2c_address));
        }
        const size_t write_byte_count = write_bytes.size();
        if (write_byte_count >= 0x100) {
            throw std::runtime_error(
                fmt::format("Size of write_bytes is too large: \"{:#x}\", has to be less than 0x100",
                    write_byte_count));
        }
        if (read_byte_count >= 0x100) {
            throw std::runtime_error(fmt::format(
                "Invalid read_byte_count \"{:#x}\", has to be less than 0x80", read_byte_count));
        }
        // FPGA only has a single I2C controller, FOR ALL INSTANCES in the
        // device, we need to serialize access between all of them.
        std::lock_guard lock(i2c_lock());
        std::shared_ptr<Timeout> timeout = Timeout::i2c_timeout(in_timeout);
        // Hololink FPGA doesn't support resetting the I2C interface;
        // so the best we can do is make sure it's not busy.
        uint32_t value = hololink_.read_uint32(reg_control_, timeout);
        if (value & TRADITIONAL_I2C_BUSY) {
            throw std::runtime_error(fmt::format(
                "Unexpected I2C_BUSY bit set, reg_control={:#x}, control value={:#x}", reg_control_, value));
        }
        //
        // set the device address and enable the i2c controller
        // I2C_DONE_CLEAR -> 1
        uint32_t control = (peripheral_i2c_address << 16) | TRADITIONAL_I2C_CORE_EN | TRADITIONAL_I2C_DONE_CLEAR;
        hololink_.write_uint32(reg_control_, control, timeout);
        // I2C_DONE_CLEAR -> 0
        control = (peripheral_i2c_address << 16) | TRADITIONAL_I2C_CORE_EN;
        hololink_.write_uint32(reg_control_, control, timeout);
        // make sure DONE is 0.
        value = hololink_.read_uint32(reg_control_, timeout);
        HSB_LOG_DEBUG("control value={:#x}", value);
        assert((value & TRADITIONAL_I2C_DONE) == 0);
        // write the num_bytes
        uint32_t num_bytes = (write_byte_count << 0) | (read_byte_count << 8);
        hololink_.write_uint32(reg_num_bytes_, num_bytes, timeout);

        const size_t remaining = write_bytes.size();
        for (size_t index = 0; index < remaining; index += 4) {
            uint32_t value;
            value = write_bytes[index] << 0;
            if (index + 1 < remaining) {
                value |= write_bytes[index + 1] << 8;
            }
            if (index + 2 < remaining) {
                value |= write_bytes[index + 2] << 16;
            }
            if (index + 3 < remaining) {
                value |= write_bytes[index + 3] << 24;
            }
            // write the register and its value
            hololink_.write_uint32(reg_data_buffer_ + index, value, timeout);
        }
        while (true) {
            // start i2c transaction.
            control = (peripheral_i2c_address << 16) | TRADITIONAL_I2C_CORE_EN | TRADITIONAL_I2C_START;
            hololink_.write_uint32(reg_control_, control, timeout);
            // retry if we don't see BUSY or DONE
            value = hololink_.read_uint32(reg_control_, timeout);
            if (value & (TRADITIONAL_I2C_DONE | TRADITIONAL_I2C_BUSY)) {
                break;
            }
            if (!timeout->retry()) {
                // timed out
                HSB_LOG_DEBUG("Timed out.");
                throw TimeoutError(
                    fmt::format("i2c_transaction i2c_address={:#x}", peripheral_i2c_address));
            }
        }
        // Poll until done.  Future version will have an event packet too.
        while (true) {
            value = hololink_.read_uint32(reg_control_, timeout);
            HSB_LOG_TRACE("control={:#x}.", value);
            const uint32_t done = value & TRADITIONAL_I2C_DONE;
            if (done != 0) {
                break;
            }
            if (!timeout->retry()) {
                // timed out
                HSB_LOG_DEBUG("Timed out.");
                throw TimeoutError(
                    fmt::format("i2c_transaction i2c_address={:#x}", peripheral_i2c_address));
            }
        }

        // round up to get the whole next word
        const uint32_t word_count = (read_byte_count + 3) / 4;
        std::vector<uint8_t> r(word_count * 4);
        for (uint32_t i = 0; i < word_count; ++i) {
            value = hololink_.read_uint32(reg_data_buffer_ + (i * 4), timeout);
            r[i * 4 + 0] = (value >> 0) & 0xFF;
            r[i * 4 + 1] = (value >> 8) & 0xFF;
            r[i * 4 + 2] = (value >> 16) & 0xFF;
            r[i * 4 + 3] = (value >> 24) & 0xFF;
        }
        r.resize(read_byte_count);
        return r;
    }

    /**
     *
     */
    Hololink::NamedLock& i2c_lock()
    {
        return hololink_.i2c_lock();
    }

    std::tuple<std::vector<unsigned>, std::vector<unsigned>, unsigned> encode_i2c_request(
        Hololink::Sequencer& sequencer,
        uint32_t peripheral_i2c_address,
        const std::vector<uint8_t>& write_bytes, uint32_t read_byte_count) override
    {
        throw std::runtime_error("encode_i2c_transaction not implemented on this platform.");
    }

private:
    Hololink& hololink_;
    const uint32_t reg_control_;
    const uint32_t reg_num_bytes_;
    const uint32_t reg_clk_ctrl_;
    const uint32_t reg_data_buffer_;
};

std::shared_ptr<Hololink::I2c> get_traditional_i2c(Hololink& hololink, uint32_t i2c_address)
{
    return std::make_shared<TraditionalI2c>(hololink, i2c_address);
}

} // namespace hololink
