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

#include <algorithm>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "enumerator.hpp"
#include "metadata.hpp"
#include "networking.hpp"
#include "timeout.hpp"

namespace hololink {

namespace core {
    class Deserializer;
}

// SPI interfaces
constexpr uint32_t SPI_CTRL = 0x0300'0000;
// HSB SPI connections
constexpr uint32_t RESET_SPI_BUS = 0;
constexpr uint32_t CLNX_SPI_BUS = 0;
constexpr uint32_t CPNX_SPI_BUS = 1;
// I2C interfaces
constexpr uint32_t BL_I2C_BUS = 0;
constexpr uint32_t CAM_I2C_BUS = 1;
// I2C controller addresses
constexpr uint32_t I2C_CTRL = 0x0300'0200;
constexpr uint32_t I2C_REG_CONTROL = 0;
constexpr uint32_t I2C_REG_BUS_EN = 4;
constexpr uint32_t I2C_REG_NUM_BYTES = 8;
constexpr uint32_t I2C_REG_CLK_CNT = 0xC;
constexpr uint32_t I2C_REG_STATUS = 0x80;
constexpr uint32_t I2C_REG_DATA_BUFFER = 0x100;

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

// I2C control flags
constexpr uint32_t I2C_START = 0b0000'0000'0000'0001;
constexpr uint32_t I2C_10B_ADDRESS = 0b0000'0000'0000'0010;
// I2C status flags
constexpr uint32_t I2C_BUSY = 0b0000'0000'0000'0001;
constexpr uint32_t I2C_FSM_ERR = 0b0000'0000'0000'0010;
constexpr uint32_t I2C_I2C_ERR = 0b0000'0000'0000'0100;
constexpr uint32_t I2C_I2C_NAK = 0b0000'0000'0000'1000;
constexpr uint32_t I2C_DONE = 0b0000'0000'0001'0000;

//
constexpr uint32_t HSB_IP_VERSION = 0x80;
constexpr uint32_t FPGA_DATE = 0x84;
constexpr uint32_t FPGA_PTP_CTRL = 0x104;
constexpr uint32_t FPGA_PTP_DELAY_ASYMMETRY = 0x10C;
constexpr uint32_t FPGA_PTP_CTRL_DPLL_CFG1 = 0x110;
constexpr uint32_t FPGA_PTP_CTRL_DPLL_CFG2 = 0x114;
constexpr uint32_t FPGA_PTP_CTRL_DELAY_AVG_FACTOR = 0x118;
constexpr uint32_t FPGA_PTP_SYNC_TS_0 = 0x180;
constexpr uint32_t FPGA_PTP_SYNC_STAT = 0x188;
constexpr uint32_t FPGA_PTP_OFM = 0x18C;

// VSYNC control
constexpr uint32_t VSYNC_CONTROL = 0x70000000;
constexpr uint32_t VSYNC_FREQUENCY = 0x70000004;
constexpr uint32_t VSYNC_DELAY = 0x70000008;
constexpr uint32_t VSYNC_START = 0x7000000C;
constexpr uint32_t VSYNC_EXPOSURE = 0x70000010;
constexpr uint32_t VSYNC_GPIO = 0x70000014;

// Async event messaging
constexpr uint32_t CTRL_EVENT = 0x0000'0200;
constexpr uint32_t CTRL_EVT_RISING = CTRL_EVENT + 0x04;
constexpr uint32_t CTRL_EVT_FALLING = CTRL_EVENT + 0x08;
constexpr uint32_t CTRL_EVT_CLEAR = CTRL_EVENT + 0x0C;
constexpr uint32_t CTRL_EVT_HOST_MAC_ADDR_LO = CTRL_EVENT + 0x10;
constexpr uint32_t CTRL_EVT_HOST_MAC_ADDR_HI = CTRL_EVENT + 0x14;
constexpr uint32_t CTRL_EVT_HOST_IP_ADDR = CTRL_EVENT + 0x18;
constexpr uint32_t CTRL_EVT_HOST_UDP_PORT = CTRL_EVENT + 0x1C;
constexpr uint32_t CTRL_EVT_FPGA_UDP_PORT = CTRL_EVENT + 0x20;
constexpr uint32_t CTRL_EVT_APB_INTERRUPT_EN = CTRL_EVENT + 0x24;
constexpr uint32_t CTRL_EVT_APB_TIMEOUT = CTRL_EVENT + 0x28;
constexpr uint32_t CTRL_EVT_SW_EVENT = CTRL_EVENT + 0x2C;
constexpr uint32_t CTRL_EVT_STAT = CTRL_EVENT + 0x80;
constexpr uint32_t APB_RAM = 0x2000;

// board IDs; note that these are deprecated
// and will be removed in a future release.  Use
// fpga_uuid instead.
constexpr uint32_t HOLOLINK_LITE_BOARD_ID = 2u;
constexpr uint32_t HOLOLINK_100G_BOARD_ID = 3u;
constexpr uint32_t MICROCHIP_POLARFIRE_BOARD_ID = 4u;
constexpr uint32_t HOLOLINK_NANO_BOARD_ID = 5u;
constexpr uint32_t LEOPARD_EAGLE_BOARD_ID = 7u;

// Other constants
constexpr uint32_t METADATA_SIZE = 128;
constexpr double APB_TIMEOUT_SCALE = 1.0 / 51.2e-9;
constexpr double APB_TIMEOUT_MAX = 0xFFFF / APB_TIMEOUT_SCALE;

constexpr uint16_t DATA_SOURCE_UDP_PORT = 12288;

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
 * Objects that can be attached to a Synchronizer.
 */
class Synchronizable {
public:
    Synchronizable();
    virtual ~Synchronizable() = default;
};

/**
 * Abstract base class for synchronizers that provide timing pulses
 * to various device functions (e.g. VSYNC camera inputs).
 */
class Synchronizer {
public:
    static std::shared_ptr<Synchronizer> null_synchronizer();

public:
    virtual ~Synchronizer() = default;

    /**
     * Attach a synchronizable peer to this synchronizer.
     * @param peer The peer to attach
     */
    void attach(std::shared_ptr<Synchronizable> peer);

    /**
     * Detach a synchronizable peer from this synchronizer.
     * @param peer The peer to detach
     */
    void detach(std::shared_ptr<Synchronizable> peer);

    /**
     * Set up the synchronizer. Called when the first peer is attached.
     */
    virtual void setup() = 0;

    /**
     * Shut down the synchronizer. Called when the last peer is detached.
     */
    virtual void shutdown() = 0;

    /**
     * Returns false if synchronization is not used.
     */
    virtual bool is_enabled() = 0;

protected:
    std::vector<std::shared_ptr<Synchronizable>> peers_;
    mutable std::mutex peers_mutex_;
};

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
        const std::string& peer_ip, uint32_t control_port, const std::string& serial_number, bool sequence_number_checking, bool skip_sequence_initialization = false, bool ptp_enable = true, bool vsync_enable = true);
    Hololink() = delete;

    virtual ~Hololink();

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
     * Initialize local resources necessary to communicate
     * with HSB (e.g. the control plane socket).
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
    uint32_t get_hsb_ip_version(const std::shared_ptr<Timeout>& timeout = std::shared_ptr<Timeout>(), bool sequence_check = true);

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
        uint32_t address, const std::shared_ptr<Timeout>& in_timeout, bool sequence_check);

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
    class NamedLock {
    public:
        /** Constructs a lock using the shm_open() call to access a named
         * semaphore with the given name.
         */
        NamedLock(Hololink& hololink, std::string name);
        ~NamedLock() noexcept(false);

        /**
         * Blocks until no other process owns this lock; then takes it.
         */
        void lock();

        /**
         * Unlocks this lock, allowing another process blocked in
         * a call to lock() to proceed.
         */
        void unlock();

    protected:
        int fd_;
    };
    friend class NamedLock;

    typedef enum {
        I2C_BUSY = 0,
        SW_EVENT = 2,
        SIF_0_FRAME_END = 16,
        SIF_1_FRAME_END = 17,
    } Event;

    /**
     * Support for e.g. I2C transactions synchronized with
     * frame-end.
     */
    class Sequencer {
    public:
        typedef enum {
            POLL = 0,
            WR = 1,
            RD = 2,
        } Op;

    public:
        Sequencer(unsigned limit = 0x200);

        // All ABCs have virtual destructors.
        virtual ~Sequencer();

        /**
         * Load the sequence into HSB and enable it.
         */
        virtual void enable() = 0;

        /**
         * Add a "write" command to the sequence, which
         * will write the given data to the given location.
         * @returns the index of the data to facilitate
         * dynamic updates later.
         */
        unsigned write_uint32(uint32_t address, uint32_t data);

        /**
         * Add a "read" command to the sequence, which
         * will read the memory location and store it in
         * the following sequence memory location.
         * @returns the index of the stored memory location.
         */
        unsigned read_uint32(uint32_t address, uint32_t initial_value = 0xFFFFFFFF);

        /**
         * Adds a "poll" command to the sequence, stalling
         * sequence execution until the data read from `address`,
         * anded with `mask` matches the value in `match`.  This
         * will time out if the match isn't found within the timeout
         * configured on the sequencer.
         * @returns the index of the match value.
         */
        unsigned poll(uint32_t address, uint32_t mask, uint32_t match);

        /**
         * What location was assigned to this sequence?  Raises
         * an exception if that wasn't set and if "write()"
         * hasn't been called on this object.
         */
        uint32_t location();

    protected:
        /**
         * Add a "done" to the sequence; this is necessary
         * before the sequence can be loaded into HSB memory.
         * This is to be called by enable() above.
         */
        void done();

        /**
         * Configure the sequence memory address at which
         * this buffer will be written.  Call this after
         * done() to update sequence content that need to
         * know where the sequence actually is.  You must
         * call assign_location on an instance exactly once.
         */
        void assign_location(Hololink::Event event);

        /**
         * Write the sequence to sequence memory starting
         * at the address given to assign_location.
         */
        void write(Hololink& hololink);

    protected:
        /**
         * Append the given value to the buffer;
         * @return the index where content was
         * written.
         */
        unsigned append(uint32_t content);

        /**
         * Update the command word with the given op;
         * perhaps adding another command word as necessary.
         * @returns the index of the command word.
         */
        unsigned add_command(Op op);

    protected:
        std::vector<uint32_t> buffer_;
        bool done_;
        unsigned command_index_;
        unsigned command_bit_;
        uint32_t address_;
        bool address_set_;
        bool written_;
    };
    friend class Sequencer;

    std::shared_ptr<Hololink::Sequencer> software_sequencer();

    class I2c {
    public:
        // All ABCs have virtual destructors
        virtual ~I2c() {};

        /**
         * @brief Set the i2c clock
         *
         * @return true
         * @return false
         */
        virtual bool set_i2c_clock() = 0;

        /**
         * @brief
         *
         * @param peripheral_i2c_address
         * @param write_bytes
         * @param read_byte_count
         * @param in_timeout
         * @return std::vector<uint8_t>
         */
        virtual std::vector<uint8_t> i2c_transaction(uint32_t peripheral_i2c_address,
            const std::vector<uint8_t>& write_bytes, uint32_t read_byte_count,
            const std::shared_ptr<Timeout>& in_timeout = std::shared_ptr<Timeout>(),
            bool ignore_nak = false)
            = 0;

        /**
         * @returns {write_indexes, read_indexes, status_index}
         */
        virtual std::tuple<std::vector<unsigned>, std::vector<unsigned>, unsigned> encode_i2c_request(
            Sequencer& sequencer,
            uint32_t peripheral_i2c_address,
            const std::vector<uint8_t>& write_bytes, uint32_t read_byte_count)
            = 0;
    };

    /**
     * @brief Get an I2c class instance
     *
     * @param i2c_address
     * @return std::shared_ptr<I2c>
     */
    std::shared_ptr<I2c> get_i2c(uint32_t i2c_bus, uint32_t i2c_address = I2C_CTRL);

    /**
     * Return a named semaphore that guarantees singleton access
     * to the I2C controller.
     */
    NamedLock& i2c_lock();

    /**
     * @brief This class supports transactions over the SPI (Serial Peripheral Interface).
     */
    class Spi {
    public:
        // All ABCs have virtual destructors
        virtual ~Spi() {};

        /**
         * @brief
         *
         * @param write_command_bytes
         * @param write_data_bytes
         * @param read_byte_count
         * @param in_timeout
         * @return std::vector<uint8_t>
         */
        virtual std::vector<uint8_t> spi_transaction(const std::vector<uint8_t>& write_command_bytes,
            const std::vector<uint8_t>& write_data_bytes, uint32_t read_byte_count,
            const std::shared_ptr<Timeout>& in_timeout = std::shared_ptr<Timeout>())
            = 0;
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
    std::shared_ptr<Spi> get_spi(uint32_t bus_number, uint32_t chip_select,
        uint32_t clock_divisor = 0x0F, uint32_t cpol = 1, uint32_t cpha = 1, uint32_t width = 1, uint32_t spi_address = SPI_CTRL);

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
        // Different board configurations will use different pin numbers
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
     * This is the routine which actually sends an ECB
     * request to the network.  The only real use of overriding
     * this method is for testing.
     */
    virtual void send_control(const std::vector<uint8_t>& request);

    /**
     * This is the routine which blocks until receipt of an ECB
     * reply is found, or the timeout elapses.  The only real use
     * of overriding this method is for testing.
     */
    virtual std::vector<uint8_t> receive_control(const std::shared_ptr<Timeout>& timeout);

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
     * ptp_synchronize(...) with a default timeout of 20 seconds.
     */
    bool ptp_synchronize()
    {
        float timeout_s = 20; // seconds
        auto timeout = std::make_shared<Timeout>(timeout_s);
        return ptp_synchronize(timeout);
    }

    /**
     * Get a PTP synchronizer instance.
     * @param frequency Optional frequency parameter (default: 60 Hz)
     * @returns A shared pointer to a Synchronizer instance
     */
    std::shared_ptr<Synchronizer> ptp_pps_output(unsigned frequency = 0);

    class PtpSynchronizer : public Synchronizer {
    public:
        PtpSynchronizer(Hololink&);
        void set_frequency(unsigned frequency);
        void setup() override;
        void shutdown() override;
        bool is_enabled() override;

    private:
        Hololink& hololink_;
        unsigned frequency_;
        unsigned frequency_control_;
    };

    /**
     * Tool for deserializing HSB received metadata blob.
     */
    typedef struct {
        uint32_t flags = 0;
        uint32_t psn = 0;
        uint32_t crc = 0;
        uint16_t frame_number = 0;
        // Time when the first sample data for the frame was received
        uint32_t timestamp_ns = 0;
        uint64_t timestamp_s = 0;
        uint64_t bytes_written = 0;
        // Time at which the metadata packet was sent
        uint32_t metadata_ns = 0;
        uint64_t metadata_s = 0;
    } FrameMetadata;
    static FrameMetadata deserialize_metadata(const uint8_t* metadata_buffer, unsigned metadata_buffer_size);

    /**
     * Configure the given event to run this sequence.
     * handler must point to a valid sequence.
     */
    void configure_apb_event(Event event, uint32_t handler = 0, bool rising_edge = true);

    /**
     * Clear handling for the given event.
     */
    void clear_apb_event(Event event);

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

    /**
     * Configure the HSB device by setting some addressing
     * etc within the HSB device.  This is used by start()
     * and reset().
     */
    void configure_hsb();

    /**
     * Listens to async_event_socket_.
     */
    void async_event_thread();

private:
    const std::string peer_ip_;
    const uint32_t control_port_;
    const std::string serial_number_;
    uint16_t sequence_ = 0x100;
    bool sequence_number_checking_ = true;

    core::UniqueFileDescriptor control_socket_;
    core::UniqueFileDescriptor async_event_socket_;
    uint16_t async_event_socket_port_;
    uint32_t hsb_ip_version_;
    uint32_t datecode_;
    std::vector<std::shared_ptr<ResetController>> reset_controllers_;
    std::mutex execute_mutex_; // protects command/response transactions with the device.
    uint32_t control_event_apb_interrupt_enable_cache_;
    uint32_t control_event_rising_cache_;
    uint32_t control_event_falling_cache_;
    std::thread async_event_thread_;
    uint32_t null_sequence_location_;
    bool skip_sequence_initialization_; // workaround for programming antique devices
    bool started_;
    std::shared_ptr<PtpSynchronizer> ptp_pps_output_;
    bool ptp_enable_;
    bool vsync_enable_;

    bool write_uint32_(uint32_t address, uint32_t value, const std::shared_ptr<Timeout>& timeout,
        bool response_expected, uint16_t sequence, bool sequence_check, std::lock_guard<std::mutex>&);
    std::tuple<bool, std::optional<uint32_t>> read_uint32_(
        uint32_t address, const std::shared_ptr<Timeout>& timeout, uint16_t sequence, bool sequence_check, std::lock_guard<std::mutex>&);

    void add_read_retries(uint32_t n);
    void add_write_retries(uint32_t n);

    // Note that we take the lock_guard as a parameter not because we
    // actually use it but because we rely on the caller holding a
    // mutex for us-- and forcing them to pass that in here guarantees
    // that they have it in the first place.
    std::tuple<bool, std::optional<uint32_t>, std::shared_ptr<core::Deserializer>> execute(
        uint16_t sequence, const std::vector<uint8_t>& request, std::vector<uint8_t>& reply,
        const std::shared_ptr<Timeout>& timeout, std::lock_guard<std::mutex>&);

    void write_renesas(I2c& i2c, const std::vector<uint8_t>& data);

    // See the comment above for execute(...) about why we take
    // std::lock_guard as a parameter here.
    uint16_t next_sequence(std::lock_guard<std::mutex>&);
};

// Tool to manage extending e.g. uint16_t changes
// into e.g. a uint32_t counter.  InType and OutType
// must be unsigned.
template <class OutType, class InType>
class ExtendedCounter {
public:
    ExtendedCounter()
        : current_(0)
        , previous_(0)
    {
    }

    static_assert(std::is_unsigned_v<InType> && std::is_unsigned_v<OutType>,
        "ExtendedCounter requires unsigned types");

    OutType update(InType in)
    {
        InType delta = in - previous_;
        previous_ = in;
        current_ += delta;
        return current_;
    }

protected:
    OutType current_;
    InType previous_;
};

} // namespace hololink

#endif /* SRC_HOLOLINK_HOLOLINK */
