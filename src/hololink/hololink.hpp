/**
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 *
 * See README.md for detailed information.
 */

#ifndef SRC_HOLOLINK_HOLOLINK
#define SRC_HOLOLINK_HOLOLINK

#include <memory>
#include <mutex>
#include <stdint.h>
#include <string>
#include <tuple>
#include <vector>

#include "enumerator.hpp"
#include "metadata.hpp"
#include "timeout.hpp"

#include <hololink/native/networking.hpp>

namespace hololink {

namespace native {
    class Deserializer;
}

// SPI interfaces
constexpr uint32_t CLNX_SPI_CTRL = 0x0300'0000;
constexpr uint32_t CPNX_SPI_CTRL = 0x0300'0200;
// I2C interfaces
constexpr uint32_t BL_I2C_CTRL = 0x0400'0000;
constexpr uint32_t CAM_I2C_CTRL = 0x0400'0200;

// packet command byte
constexpr uint32_t WR_DWORD = 0x04;
constexpr uint32_t RD_DWORD = 0x14;
// request packet flag bits
constexpr uint32_t REQUEST_FLAGS_ACK_REQUEST = 0b0000'0001;
constexpr uint32_t REQUEST_FLAGS_SEQUENCE_CHECK = 0b0000'0010;
// response codes
constexpr uint32_t RESPONSE_SUCCESS = 0x00;
constexpr uint32_t RESPONSE_ERROR_GENERAL = 0x02;
constexpr uint32_t RESPONSE_INVALID_ADDR = 0x03;
constexpr uint32_t RESPONSE_INVALID_CMD = 0x04;
constexpr uint32_t RESPONSE_INVALID_PKT_LENGTH = 0x05;
constexpr uint32_t RESPONSE_INVALID_FLAGS = 0x06;
constexpr uint32_t RESPONSE_BUFFER_FULL = 0x07;
constexpr uint32_t RESPONSE_INVALID_BLOCK_SIZE = 0x08;
constexpr uint32_t RESPONSE_INVALID_INDIRECT_ADDR = 0x09;
constexpr uint32_t RESPONSE_COMMAND_TIMEOUT = 0x0A;
constexpr uint32_t RESPONSE_SEQUENCE_CHECK_FAIL = 0x0B;

// control flags
constexpr uint32_t I2C_START = 0b0000'0000'0000'0001;
constexpr uint32_t I2C_CORE_EN = 0b0000'0000'0000'0010;
constexpr uint32_t I2C_DONE_CLEAR = 0b0000'0000'0001'0000;
constexpr uint32_t I2C_BUSY = 0b0000'0001'0000'0000;
constexpr uint32_t I2C_DONE = 0b0001'0000'0000'0000;

//
constexpr uint32_t FPGA_VERSION = 0x80;
constexpr uint32_t FPGA_DATE = 0x84;
constexpr uint32_t FPGA_PTP_CTRL = 0x104;
constexpr uint32_t FPGA_PTP_CTRL_DPLL_CFG1 = 0x110;
constexpr uint32_t FPGA_PTP_CTRL_DPLL_CFG2 = 0x114;
constexpr uint32_t FPGA_PTP_SYNC_TS_0 = 0x180;
constexpr uint32_t FPGA_PTP_OFM = 0x18C;

// board IDs
constexpr uint32_t HOLOLINK_LITE_BOARD_ID = 2u;
constexpr uint32_t HOLOLINK_100G_BOARD_ID = 3u;
constexpr uint32_t MICROCHIP_POLARFIRE_BOARD_ID = 4u;
constexpr uint32_t HOLOLINK_NANO_BOARD_ID = 5u;

// Other constants
constexpr uint32_t METADATA_SIZE = 128;

class TimeoutError : public std::runtime_error {
public:
    explicit TimeoutError(const std::string& arg)
        : runtime_error(arg)
    {
    }
};

class UnsupportedVersion : public std::runtime_error {
public:
    explicit UnsupportedVersion(const std::string& arg)
        : runtime_error(arg)
    {
    }
};

/**
 * Defined in data_channel.hpp.
 */
class DataChannel;

/**
 * @brief
 *
 */
class Hololink {
    /** DataChannel calls some methods we don't want to share. */
    friend class DataChannel;

public:
    /**
     * @brief Construct a new Hololink object
     *
     * @param peer_ip
     * @param control_port
     * @param serial_number
     */
    explicit Hololink(
        const std::string& peer_ip, uint32_t control_port, const std::string& serial_number, bool sequence_number_checking);
    Hololink() = delete;

    virtual ~Hololink() = default;

    /**
     * @brief
     *
     * @param metadata
     * @return std::shared_ptr<Hololink>
     */
    static std::shared_ptr<Hololink> from_enumeration_metadata(const Metadata& metadata);

    /**
     * @brief
     */
    static void reset_framework();

    /**
     * @brief
     *
     * @param metadata
     * @return true
     * @return false
     */
    static bool enumerated(const Metadata& metadata);

    /**
     * Returns (frame_start_size, frame_end_size, line_start_size, line_end_size),
     * all are in bytes.
     */
    std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> csi_size();

    /**
     * @brief
     */
    void start();

    /**
     * @brief
     */
    void stop();

    /**
     * @brief
     */
    void reset();

    /**
     * @param timeout
     * @returns the FPGA version
     */
    uint32_t get_fpga_version(const std::shared_ptr<Timeout>& timeout = std::shared_ptr<Timeout>(), bool check_sequence = true);

    /**
     * @returns the FPGA date
     */
    uint32_t get_fpga_date();

    /**
     * @brief
     *
     * @param address
     * @param value
     * @param in_timeout
     * @param retry
     * @return true
     * @return false
     */
    bool write_uint32(uint32_t address, uint32_t value,
        const std::shared_ptr<Timeout>& in_timeout, bool retry, bool sequence_check);

    bool write_uint32(uint32_t address, uint32_t value,
        const std::shared_ptr<Timeout>& timeout, bool retry = true)
    {
        return write_uint32(address, value, timeout, retry, sequence_number_checking_);
    }

    bool write_uint32(uint32_t address, uint32_t value)
    {
        const std::shared_ptr<Timeout>& timeout = std::shared_ptr<Timeout>();
        return write_uint32(address, value, timeout);
    }

    /**
     * @brief Returns the value found at the location or calls hololink timeout if there's a
     * problem.
     *
     * @param address
     * @param in_timeout
     * @return uint32_t
     */
    uint32_t read_uint32(
        uint32_t address, const std::shared_ptr<Timeout>& in_timeout, bool check_sequence);

    uint32_t read_uint32(uint32_t address, const std::shared_ptr<Timeout>& timeout)
    {
        return read_uint32(address, timeout, sequence_number_checking_);
    }

    uint32_t read_uint32(uint32_t address)
    {
        const std::shared_ptr<Timeout>& timeout = std::shared_ptr<Timeout>();
        return read_uint32(address, timeout);
    }

    /**
     * @brief Setup the clock
     *
     * @param clock_profile
     */
    void setup_clock(const std::vector<std::vector<uint8_t>>& clock_profile);

    /**
     * Used to guarantee serialized access to I2C or SPI controllers.  The FPGA
     * only has a single I2C controller-- what looks like independent instances
     * are really just pin-muxed outputs from a single I2C controller block within
     * the device-- the same is true for SPI.
     */
    class NamedLock;
    friend class NamedLock;

    class I2c {
    public:
        /**
         * @brief Construct a new I2c object
         *
         * @param hololink
         * @param i2c_address
         */
        explicit I2c(Hololink& hololink, uint32_t i2c_address);
        I2c() = delete;

        /**
         * @brief Set the i2c clock
         *
         * @return true
         * @return false
         */
        bool set_i2c_clock();

        /**
         * @brief
         *
         * @param peripheral_i2c_address
         * @param write_bytes
         * @param read_byte_count
         * @param in_timeout
         * @return std::vector<uint8_t>
         */
        std::vector<uint8_t> i2c_transaction(uint32_t peripheral_i2c_address,
            const std::vector<uint8_t>& write_bytes, uint32_t read_byte_count,
            const std::shared_ptr<Timeout>& in_timeout = std::shared_ptr<Timeout>());

        /**
         *
         */
        NamedLock& i2c_lock()
        {
            return hololink_.i2c_lock();
        }

    private:
        Hololink& hololink_;
        const uint32_t reg_control_;
        const uint32_t reg_num_bytes_;
        const uint32_t reg_clk_ctrl_;
        const uint32_t reg_data_buffer_;
    };

    /**
     * @brief Get an I2c class instance
     *
     * @param i2c_address
     * @return std::shared_ptr<I2c>
     */
    std::shared_ptr<I2c> get_i2c(uint32_t i2c_address);

    /**
     * Return a named semaphore that guarantees singleton access
     * to the I2C controller.
     */
    NamedLock& i2c_lock();

    /**
     * @brief This class supports transactions over the SPI (Serial Periperal Interface).
     */
    class Spi {
    public:
        /**
         * @brief Construct a new Spi object
         *
         * @param hololink
         * @param address
         * @param spi_cfg
         */
        explicit Spi(Hololink& hololink, uint32_t address, uint32_t spi_cfg);
        Spi() = delete;

        /**
         * @brief
         *
         * @param write_command_bytes
         * @param write_data_bytes
         * @param read_byte_count
         * @param in_timeout
         * @return std::vector<uint8_t>
         */
        std::vector<uint8_t> spi_transaction(const std::vector<uint8_t>& write_command_bytes,
            const std::vector<uint8_t>& write_data_bytes, uint32_t read_byte_count,
            const std::shared_ptr<Timeout>& in_timeout = std::shared_ptr<Timeout>());

        NamedLock& spi_lock()
        {
            return hololink_.spi_lock();
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

    /**
     * @brief Get the spi object
     *
     * @param spi_address
     * @param chip_select
     * @param clock_divisor
     * @param cpol
     * @param cpha
     * @param width
     * @return std::shared_ptr<Spi>
     */
    std::shared_ptr<Spi> get_spi(uint32_t spi_address, uint32_t chip_select,
        uint32_t clock_divisor = 0x0F, uint32_t cpol = 1, uint32_t cpha = 1, uint32_t width = 1);

    /**
     * Return a named semaphore that guarantees singleton access
     * to the SPI controller.
     */
    NamedLock& spi_lock();

    /**
     * @brief GPIO class
     */
    class GPIO {
    public:
        /**
         * @brief Construct a new GPIO object
         *
         * @param hololink
         */
        explicit GPIO(Hololink& hololink, uint32_t gpio_pin_number);
        GPIO() = delete;

        // Direction constants
        inline static constexpr uint32_t IN = 1;
        inline static constexpr uint32_t OUT = 0;

        // Bitmask constants
        inline static constexpr uint32_t LOW = 0;
        inline static constexpr uint32_t HIGH = 1;

        // 256 pins in FPGA - range 0...255
        // Diffrent board configurations will use different pin numbers
        // Lattice 10G - 16 pins
        // Bajoran nano - 54 pins
        // Future platform up to 256 pins supported by FPGA
        inline static constexpr uint32_t GPIO_PIN_RANGE = 0x100;

        /**
         * @brief
         *
         * @param pin
         * @param direction
         */
        void set_direction(uint32_t pin, uint32_t direction);

        /**
         * @brief
         *
         * @param pin
         * @return Direction
         */
        uint32_t get_direction(uint32_t pin);

        /**
         * @brief
         *
         * @param pin
         * @param value
         */
        void set_value(uint32_t pin, uint32_t value);

        /**
         * @brief
         *
         * @param pin
         * @return uint32_t
         */
        uint32_t get_value(uint32_t pin);

        /**
         * @brief
         *
         * @param
         * @return uint32_t
         */
        uint32_t get_supported_pin_num(void);

    private:
        Hololink& hololink_;
        uint32_t gpio_pin_number_;

        static uint32_t set_bit(uint32_t value, uint32_t bit);
        static uint32_t clear_bit(uint32_t value, uint32_t bit);
        static uint32_t read_bit(uint32_t value, uint32_t bit);
    };

    /**
     * @brief Get an GPIO class instance
     *
     * @return std::shared_ptr<GPIO>
     */
    std::shared_ptr<GPIO> get_gpio(Metadata& metadata);

    /**
     * @brief
     *
     * @param request
     */
    virtual void send_control(const std::vector<uint8_t>& request);

    class ResetController {
    public:
        virtual ~ResetController();
        virtual void reset() = 0;
    };

    /**
     * Add a callback devices can use to command reset.
     */
    void on_reset(std::shared_ptr<ResetController> reset_controller);

    /**
     * Wait up to the given timeout for PTP to synchronize.
     * @returns false if no PTP sync messages are received
     * within the allowed time.
     */
    bool ptp_synchronize(const std::shared_ptr<Timeout>& timeout);

    /**
     * Tool for deserializing HSB received metadata blob.
     */
    typedef struct {
        uint32_t flags;
        uint32_t psn;
        uint32_t crc;
        uint32_t frame_number;
        // Time when the first sample data for the frame was received
        uint32_t timestamp_ns;
        uint64_t timestamp_s;
        uint64_t bytes_written;
        // Time at which the metadata packet was sent
        uint32_t metadata_ns;
        uint64_t metadata_s;
    } FrameMetadata;
    static FrameMetadata deserialize_metadata(const uint8_t* metadata_buffer, unsigned metadata_buffer_size);

protected:
    /**
     * @brief Override this guy to record timing around ACKs etc
     *
     * @param request_time
     * @param request
     * @param reply_time
     * @param reply
     */
    virtual void executed(double request_time, const std::vector<uint8_t>& request, double reply_time,
        const std::vector<uint8_t>& reply);

    /**
     * Return a filename that, for any program talking to
     * this specific device, will always produce the same
     * filename for a given value in name.  This file is
     * not guaranteed to persist past host reboot.  This is
     * useful e.g. providing locks for transactions to a
     * specific board.  Note that this works for all processes
     * on this host but isn't smart enough to share the same
     * space with other hosts.
     */
    std::string device_specific_filename(std::string name);

    /**
     * Return a named semaphore that guarantees singleton access
     * to misc Hololink device resources, across all processes
     * on the current machine.  and_uint32 and or_uint32 use this.
     */
    NamedLock& lock();

    /**
     * Clears any bits of the given memory location with
     * the bits not set in mask.
     */
    bool and_uint32(uint32_t address, uint32_t mask);

    /**
     * Sets any bits of the given memory location with
     * the bits set in the mask.
     */
    bool or_uint32(uint32_t address, uint32_t mask);

private:
    const std::string peer_ip_;
    const uint32_t control_port_;
    const std::string serial_number_;
    uint16_t sequence_ = 0x100;
    bool sequence_number_checking_ = true;

    native::UniqueFileDescriptor control_socket_;
    uint32_t version_;
    uint32_t datecode_;
    std::vector<std::shared_ptr<ResetController>> reset_controllers_;
    std::mutex execute_mutex_; // protects command/response transactions with the device.

    bool write_uint32_(uint32_t address, uint32_t value, const std::shared_ptr<Timeout>& timeout,
        bool response_expected, bool sequence_check);
    std::tuple<bool, std::optional<uint32_t>> read_uint32_(
        uint32_t address, const std::shared_ptr<Timeout>& timeout, bool sequence_check);

    void add_read_retries(uint32_t n);
    void add_write_retries(uint32_t n);

    // Note that we take the lock_guard as a parameter not because we
    // actually use it but because we rely on the caller holding a
    // mutex for us-- and forcing them to pass that in here guarantees
    // that they have it in the first place.
    std::tuple<bool, std::optional<uint32_t>, std::shared_ptr<native::Deserializer>> execute(
        uint16_t sequence, const std::vector<uint8_t>& request, std::vector<uint8_t>& reply,
        const std::shared_ptr<Timeout>& timeout, std::lock_guard<std::mutex>&);

    std::vector<uint8_t> receive_control(const std::shared_ptr<Timeout>& timeout);

    void write_renesas(I2c& i2c, const std::vector<uint8_t>& data);

    // See the comment above for execute(...) about why we take
    // std::lock_guard as a parameter here.
    uint16_t next_sequence(std::lock_guard<std::mutex>&);
};

} // namespace hololink

#endif /* SRC_HOLOLINK_HOLOLINK */
