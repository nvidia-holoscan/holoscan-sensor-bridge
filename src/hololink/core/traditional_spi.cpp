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

// SPI control flags
constexpr uint32_t SPI_START = 0b0000'0000'0000'0001;
constexpr uint32_t SPI_BUSY = 0b0000'0001'0000'0000;
// SPI_CFG
constexpr uint32_t SPI_CFG_CPOL = 0b0000'0000'0001'0000;
constexpr uint32_t SPI_CFG_CPHA = 0b0000'0000'0010'0000;

class TraditionalSpi
    : public Hololink::Spi {
public:
    TraditionalSpi(Hololink& hololink, uint32_t spi_address, uint32_t chip_select,
        uint32_t clock_divisor, uint32_t cpol, uint32_t cpha, uint32_t width)
        : hololink_(hololink)
        , reg_control_(spi_address + 0)
        , reg_num_bytes_(spi_address + 4)
        , reg_spi_cfg_(spi_address + 8)
        , reg_num_bytes2_(spi_address + 12)
        , reg_data_buffer_(spi_address + 16)
        , spi_cfg_(spi_cfg(chip_select, clock_divisor, cpol, cpha, width))
        , turnaround_cycles_(0)
    {
    }

    static uint32_t spi_cfg(uint32_t chip_select,
        uint32_t clock_divisor, uint32_t cpol, uint32_t cpha, uint32_t width)
    {
        if (clock_divisor >= 16) {
            throw std::runtime_error(
                fmt::format("Invalid clock_divisor \"{}\", has to be less than 16", clock_divisor));
        }
        if (chip_select >= 8) {
            throw std::runtime_error(
                fmt::format("Invalid chip_select \"{}\", has to be less than 8", chip_select));
        }
        std::map<uint32_t, uint32_t> width_map {
            { 1, 0 },
            { 2, 2 << 8 },
            { 4, 3 << 8 },
        };
        // we let this next statement raise an
        // exception if the width parameter isn't
        // supported.
        uint32_t r = clock_divisor | (chip_select << 12) | width_map[width];
        if (cpol) {
            r |= SPI_CFG_CPOL;
        }
        if (cpha) {
            r |= SPI_CFG_CPHA;
        }
        return r;
    }

    Hololink::NamedLock& spi_lock()
    {
        return hololink_.spi_lock();
    }

    std::vector<uint8_t> spi_transaction(const std::vector<uint8_t>& write_command_bytes,
        const std::vector<uint8_t>& write_data_bytes, uint32_t read_byte_count,
        const std::shared_ptr<Timeout>& in_timeout) override
    {
        std::vector<uint8_t> write_bytes(write_command_bytes);
        write_bytes.insert(write_bytes.end(), write_data_bytes.begin(), write_data_bytes.end());
        const uint32_t write_command_count = write_command_bytes.size();
        if (write_command_count >= 16) { // available bits in num_bytes2
            throw std::runtime_error(
                fmt::format("Size of combined write_command_bytes and write_data_bytes is too large: "
                            "\"{}\", has to be less than 16",
                    write_command_count));
        }
        const uint32_t write_byte_count = write_bytes.size();
        const uint32_t buffer_size = 288;
        // The buffer needs to have enough space for the read bytes (command + read)
        const uint32_t read_byte_total = write_command_count + read_byte_count;
        if (read_byte_total > buffer_size) {
            throw std::runtime_error(fmt::format("Size of read is too large: "
                                                 "\"{:#x}\", has to be less than {:#x}",
                read_byte_total, buffer_size));
        }
        // FPGA only has a single SPI controller, FOR ALL INSTANCES in the
        // device, we need to serialize access between all of them.
        std::lock_guard lock(spi_lock());
        std::shared_ptr<Timeout> timeout = Timeout::spi_timeout(in_timeout);
        // Hololink FPGA doesn't support resetting the SPI interface;
        // so the best we can do is see that it's not busy.
        uint32_t value = hololink_.read_uint32(reg_control_, timeout);
        assert((value & SPI_BUSY) == 0);
        // Set the configuration
        hololink_.write_uint32(reg_spi_cfg_, spi_cfg_, timeout);
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
            hololink_.write_uint32(reg_data_buffer_ + index, value, timeout);
        }
        // write the num_bytes; note that these are 9-bit values that top
        // out at (buffer_size=288) (length checked above)
        const uint32_t num_bytes = (write_byte_count << 0) | (read_byte_count << 16);
        hololink_.write_uint32(reg_num_bytes_, num_bytes, timeout);
        assert(turnaround_cycles_ < 16);
        const uint32_t num_bytes2 = turnaround_cycles_ | (write_command_count << 8);
        hololink_.write_uint32(reg_num_bytes2_, num_bytes2, timeout);
        // start the SPI transaction.  don't retry this guy; just raise
        // an error if we don't see the ack.
        const uint32_t control = SPI_START;
        bool status = hololink_.write_uint32(reg_control_, control, timeout, false /*retry*/
        );
        if (!status) {
            throw std::runtime_error(
                fmt::format("ACK failure writing to SPI control register {:#x}.", reg_control_));
        }
        // wait until we don't see busy, which may be immediately
        while (true) {
            value = hololink_.read_uint32(reg_control_, timeout);
            const uint32_t busy = value & SPI_BUSY;
            if (busy == 0) {
                break;
            }
            if (!timeout->retry()) {
                // timed out
                HSB_LOG_DEBUG("Timed out.");
                throw TimeoutError(fmt::format("spi_transaction control={:#x}", reg_control_));
            }
        }
        // round up to get the whole next word
        std::vector<uint8_t> r(read_byte_total + 3);
        for (uint32_t i = 0; i < read_byte_total; i += 4) {
            value = hololink_.read_uint32(reg_data_buffer_ + i, timeout);
            r[i + 0] = (value >> 0) & 0xFF;
            r[i + 1] = (value >> 8) & 0xFF;
            r[i + 2] = (value >> 16) & 0xFF;
            r[i + 3] = (value >> 24) & 0xFF;
        }
        // skip over the data that we wrote out.
        r = std::vector<uint8_t>(
            r.cbegin(), r.cbegin() + read_byte_total);
        return r;
    }

private:
    Hololink& hololink_;
    const uint32_t reg_control_;
    const uint32_t reg_num_bytes_;
    const uint32_t reg_spi_cfg_;
    const uint32_t reg_num_bytes2_;
    const uint32_t reg_data_buffer_;
    const uint32_t spi_cfg_;
    uint32_t turnaround_cycles_;
};

std::shared_ptr<Hololink::Spi> get_traditional_spi(Hololink& hololink, uint32_t spi_address, uint32_t chip_select,
    uint32_t clock_divisor, uint32_t cpol, uint32_t cpha, uint32_t width)
{
    return std::make_shared<TraditionalSpi>(hololink, spi_address, chip_select, clock_divisor, cpol, cpha, width);
}

} // namespace hololink
