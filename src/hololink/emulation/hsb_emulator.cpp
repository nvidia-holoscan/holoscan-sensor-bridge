/**
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
 *
 * See README.md for detailed information.
 */

#include <chrono>
#include <climits>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "hololink/core/deserializer.hpp"
#include "hololink/core/hololink.hpp"
#include "hololink/core/serializer.hpp"

#include "data_plane.hpp"
#include "hsb_emulator.hpp"
#include "i2c_interface.hpp"
#include "mem_register.hpp"
#include "net.hpp"
#include "utils.hpp"

// this is wrapping POSIX poll() flags. We read on any type of read and stop on all errors/hangups
#define POLL_RD_ANY (POLLIN | POLLPRI)
#define POLL_STOP (POLLHUP | POLLERR | POLLNVAL)

// This is part of workaround for polling on I2C transactions.
#define POLL_COUNT_TRIGGER 30
#define POLL_COUNT_EXIT 300

#define MIN_VALID_CONTROL_LENGTH 10u
// APB event constants
#define APB_SIF_0_FRAME_END_START_ADDRESS (hololink::APB_RAM + 0x100)
#define APB_SIF_1_FRAME_END_START_ADDRESS (hololink::APB_RAM + 0x200)
#define APB_SW_EVENT_START_ADDRESS (hololink::APB_RAM + 0x800)

// based on src/hololink/core/hololink.cpp
// Allocate buffers for control plane requests and replies to this
// size, which is guaranteed to be large enough for the largest
// of any of those buffers.
#define CONTROL_PACKET_SIZE 1472

enum ECB_CMD_CODE {
    WR_BYTE = 0x01,
    WR_WORD = 0x02,
    WR_DWORD = 0x04,
    WR_BLOCK = 0x09,
    RMW_BYTE = 0x0A,
    RD_BYTE = 0x11,
    RD_WORD = 0x12,
    RD_DWORD = 0x14,
    RD_BLOCK = 0x19,
    GET_INFO = 0x20,
};

enum ECB_RESPONSE_CODE {
    SUCCESS = 0x00,
    ADDRESS_ERROR = 0x03,
    COMMAND_ERROR = 0x04,
    FLAG_ERROR = 0x06,
    SEQUENCE_ERROR = 0x0B
};

namespace hololink::emulation {

struct ControlMessage {
    uint8_t cmd_code;
    uint8_t flags;
    uint16_t sequence;
    uint8_t status;
    uint8_t reserved;
    uint16_t num_addresses;
    uint32_t addresses[CONTROL_PACKET_SIZE / sizeof(uint32_t)];
    uint32_t values[CONTROL_PACKET_SIZE / sizeof(uint32_t)];
};

// returns true if the message is valid, false otherwise.
// NOTE: if false, the ControlMessage object is in an indeterminate state.
bool deserialize_control_message(hololink::core::Deserializer& deserializer, ControlMessage& message)
{
    uint16_t num_addresses = 0;
    auto ecb_status = deserializer.next_uint8(message.cmd_code)
        && deserializer.next_uint8(message.flags)
        && deserializer.next_uint16_be(message.sequence)
        && deserializer.next_uint8(message.status)
        && deserializer.next_uint8(message.reserved);
    if (!ecb_status) {
        return false;
    }
    uint32_t address = 0;
    while (num_addresses < CONTROL_PACKET_SIZE / sizeof(uint32_t) && deserializer.next_uint32_be(address)) {
        message.addresses[num_addresses] = address;
        if (!deserializer.next_uint32_be(message.values[num_addresses])) {
            message.values[num_addresses] = 0; // this is the end of a read message that didn't provide dummy data. use 0z
        }
        num_addresses++;
    }
    message.num_addresses = num_addresses;
    return true;
}

// returns 0 on failure or the number of bytes written on success
// Note that on failure, serializer and buffer contents are in indeterminate state.
size_t serialize_reply_message(hololink::core::Serializer& serializer, ControlMessage& message)
{
    auto ecb_status = serializer.append_uint8(message.cmd_code)
        && serializer.append_uint8(message.flags)
        && serializer.append_uint16_be(message.sequence)
        && serializer.append_uint8(message.status)
        && serializer.append_uint8(message.reserved);
    if (!ecb_status) {
        return 0;
    }
    for (uint16_t i = 0; i < message.num_addresses; i++) {
        if (!(serializer.append_uint32_be(message.addresses[i]) && serializer.append_uint32_be(message.values[i]))) {
            return 0;
        }
    }
    serializer.append_uint32_be(0); // latched_sequence
    return serializer.length();
}

// class that currently just handles SW_EVENT APB events.
// runs a thread with a queue of events to execute. in practice,
// for the same type of event, there is only ever one executing at a time
class APBEventHandler {
public:
    APBEventHandler(HSBEmulator& hsbemu)
        : hsbemu_(hsbemu)
    {
    }

    ~APBEventHandler()
    {
        stop();
    }

    // this starts the apb_event consumer thread, effectively starting the controller
    void start()
    {
        std::unique_lock<std::mutex> lock(apb_event_queue_mutex_);
        running_ = true;
        apb_event_thread_ = std::thread(&APBEventHandler::run, this);
    }

    void stop()
    {
        if (!is_running()) {
            return;
        }
        std::unique_lock<std::mutex> lock(apb_event_queue_mutex_);
        running_ = false;
        apb_event_queue_cv_.notify_all();
        lock.unlock(); // unlock before joining thread to avoid deadlock
        if (apb_event_thread_.joinable()) {
            apb_event_thread_.join();
        }
        lock.lock(); // this is not strictly necessary since queue_event
                     // throws on attempts to enqueue/modify the queue while
                     // APBEventHandler is in running state, but guard against
                     // future changes
        // clear the queue
        std::queue<APBEventEntry> empty;
        std::swap(apb_event_queue_, empty);
        lock.unlock();
    }

    bool is_running()
    {
        return running_;
    }

    // queue up an event to execute
    void queue_event(hololink::Hololink::Event event)
    {
        uint32_t start_address = 0;
        switch (event) {
        case hololink::Hololink::Event::SW_EVENT:
            start_address = APB_SW_EVENT_START_ADDRESS;
            break;
        case hololink::Hololink::Event::I2C_BUSY:
            break;
        case hololink::Hololink::Event::SIF_0_FRAME_END:
            start_address = APB_SIF_0_FRAME_END_START_ADDRESS;
            break;
        case hololink::Hololink::Event::SIF_1_FRAME_END:
            start_address = APB_SIF_1_FRAME_END_START_ADDRESS;
            break;
        default:
            throw std::runtime_error("APBEventHandler: invalid event " + std::to_string(static_cast<int>(event)));
        }
        std::unique_lock<std::mutex> lock(apb_event_queue_mutex_);
        if (!running_) {
            throw std::runtime_error("APBEventHandler is not running. cannot queue event");
        }
        apb_event_queue_.push(APBEventEntry { event, start_address });
        apb_event_queue_cv_.notify_one();
    }

private:
    struct APBEventEntry {
        hololink::Hololink::Event event;
        uint32_t start_address;
    };

    // WRITE CMD callbackseq_address is where the written value is read from. seq_address is advanced by the size of the address
    void sequence_write(uint32_t& seq_address, uint32_t register_address)
    {
        uint32_t value = next_value(seq_address);
        hsbemu_.write(register_address, value);
    }

    // READ CMD callback.seq_address is where the read value will be stored. seq_address is advanced by the size of the address
    void sequence_read(uint32_t& seq_address, uint32_t register_address)
    {
        uint32_t value = hsbemu_.read(register_address);
        hsbemu_.write(seq_address, value);
        (void)next_value(seq_address); // mostly to advance the seq_address. discard result for now
    }

    // POLL CMD callback. seq_address is where the match value is read from. seq_address is advanced by the size of the address
    void sequence_poll(uint32_t& seq_address, uint32_t register_address)
    {
        uint32_t match = next_value(seq_address);
        uint32_t mask = next_value(seq_address);
        // no checking for timeout yet
        while (true) {
            uint32_t value = hsbemu_.read(register_address);
            if ((value & mask) == match) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // ERROR CMD callback. seq_address is where the error value is read from. seq_address is advanced by the size of the address
    void sequence_error(uint32_t& seq_address, uint32_t register_address)
    {
        throw std::runtime_error("invalid command code '11' detected");
    }

    enum SequenceCommand : uint8_t {
        POLL = 0,
        WR = 1,
        RD = 2,
        ERROR = 3,
    };
    static constexpr uint8_t SEQUENCE_COMMAND_SIZE_BITS_ = 2;
    static constexpr uint8_t SEQUENCE_COMMAND_MASK_ = (1 << SEQUENCE_COMMAND_SIZE_BITS_) - 1;
    static constexpr uint32_t DONE_COMMAND_ADDRESS_ = 0xFFFF'FFFF;
    static constexpr uint32_t APB_EVENT_BUFFER_SIZE_ = 0x100;
    typedef void (APBEventHandler::*apb_event_cmd_callback_ty)(uint32_t& seq_address, uint32_t register_address);
    static constexpr apb_event_cmd_callback_ty APB_EVENT_CMD_CALLBACKS_[] = {
        [POLL] = &APBEventHandler::sequence_poll,
        [WR] = &APBEventHandler::sequence_write,
        [RD] = &APBEventHandler::sequence_read,
        [ERROR] = &APBEventHandler::sequence_error,
    };

    // seq_index, cmd_index, and cmd_bit are all the next values to consume from the sequence_values vector
    // returns the next command code and if necessary, updates the indices to the next value to consume
    uint8_t next_cmd(uint32_t& cmd_word, uint32_t& address, uint8_t& cmd_bit)
    {
        if (cmd_bit >= sizeof(cmd_word) * CHAR_BIT) {
            cmd_bit = 0;
            cmd_word = next_value(address);
        }

        uint8_t cmd = (cmd_word >> cmd_bit) & SEQUENCE_COMMAND_MASK_;
        cmd = (cmd > ERROR) ? ERROR : cmd;
        cmd_bit += SEQUENCE_COMMAND_SIZE_BITS_;
        return cmd;
    }

    // returns the value at the address and advances the address by the size of the address
    uint32_t next_value(uint32_t& address)
    {
        uint32_t value = hsbemu_.read(address);
        address += sizeof(address); // advance next address
        return value;
    }

    void execute_sequence(const APBEventEntry& entry)
    {
        // sequeunce values is a byte stream of 32-bit values of format:
        // CMD = 2-bit command code
        // ADDR = 32-bit address.
        // DATA = 32-bit data
        // CMD=WR consumes 2 values (ADDR, DATA), CMD=RD consumes 2 values (ADDR, DATA), CMD=POLL consumes 3 values (ADDR, DATA, DATA)
        // CMD=0 with data to consume is an error, CMD=RD with only one DATA value is a DONE command
        // ADDR/DATA prefixed by (CMD) shows the command that owns it
        // with "[]" delineating 32-bit boundaries, an example sequence will look like:
        // [RWRWRRPWRWWWWWRR]
        //          [(R)ADDR][(R)DATA][(R)ADDR][(R)DATA][(W)ADDR][(W)DATA][(W)ADDR][(W)DATA]
        //          [(W)ADDR][(W)DATA][(W)ADDR][(W)DATA][(W)ADDR][(W)DATA][(R)ADDR][(R)DATA]
        //          [(W)ADDR][(W)DATA][(P)ADDR][(P)DATA][(P)DATA][(R)ADDR][(R)DATA][(R)ADDR][(R)DATA]
        //          [(W)ADDR][(W)DATA][(R)ADDR][(R)DATA][(W)ADDR][(W)DATA][(R)ADDR][(R)DATA]
        // [00000000000000RW]
        //          [(W)ADDR][(W)DATA][(R)DATA] // terminal DONE command only has a single value to read
        //
        uint32_t seq_address = entry.start_address;
        uint8_t cmd_bit = 0;
        uint32_t cmd_word = next_value(seq_address);
        int count = 0;
        while (true) { // should just be true
            uint8_t cmd_code = next_cmd(cmd_word, seq_address, cmd_bit);
            // the next value is already an address
            uint32_t address = next_value(seq_address);
            // check for a DONE command identified as the -1/0xFFFF'FFFF address
            if (address == DONE_COMMAND_ADDRESS_) {
                return;
            }

            auto callback = APB_EVENT_CMD_CALLBACKS_[cmd_code];
            // all 3 commands + error have a valid callback. Any code not 00/01/10 results in an exception
            (this->*callback)(seq_address, address);
            count++;
        }
    }

    // consume from apb_event_queue_ and execute the sequence
    void run()
    {
        while (running_) {
            std::unique_lock<std::mutex> lock(apb_event_queue_mutex_);
            apb_event_queue_cv_.wait(lock, [this] { return !apb_event_queue_.empty() || !running_; });
            if (!running_) {
                break;
            }
            APBEventEntry entry = apb_event_queue_.front();
            apb_event_queue_.pop();
            lock.unlock(); // allow other threads to write to queue
            execute_sequence(entry);
        }
    }
    std::mutex apb_event_queue_mutex_;
    std::condition_variable apb_event_queue_cv_;
    std::queue<APBEventEntry> apb_event_queue_;
    std::atomic<bool> running_ { false };
    std::thread apb_event_thread_;
    HSBEmulator& hsbemu_;
};

I2CController::I2CController(HSBEmulator& hsb_emulator, uint32_t controller_address)
    : registers_(hsb_emulator.get_memory())
    , controller_address_(controller_address)
    , status_address_(controller_address + hololink::I2C_REG_STATUS)
    , bus_en_address_(controller_address + hololink::I2C_REG_BUS_EN)
    , num_bytes_address_(controller_address + hololink::I2C_REG_NUM_BYTES)
    , clk_cnt_address_(controller_address + hololink::I2C_REG_CLK_CNT)
    , data_buffer_address_(controller_address + hololink::I2C_REG_DATA_BUFFER)
{
}

I2CController::~I2CController()
{
    stop();
}

void I2CController::start()
{
    std::unique_lock<std::mutex> lock(i2c_mutex_);

    // start the peripherals
    for (auto& bus_map : i2c_bus_map_) {
        for (auto& peripheral : bus_map.second) {
            peripheral.second->start();
        }
    }

    running_ = true;
    i2c_thread_ = std::thread(&I2CController::run, this);
}

void I2CController::execute(uint32_t value)
{

    std::unique_lock<std::mutex> lock(i2c_mutex_);
    if (!running_) {
        throw std::runtime_error("I2CController: cannot execute i2c transaction while controller is not running");
    }
    uint32_t status = registers_->read(status_address_);
    uint16_t cmd = value & 0xFFFF;

    if (hololink::I2C_BUSY == status) {
        throw std::runtime_error("I2CController: cannot execute i2c transaction while controller is busy. CMD=" + std::to_string(cmd) + " I2C_STATUS=" + std::to_string(status));
    }
    cmd_ = cmd;
    peripheral_address_ = value >> 16;
    // short circuit setting I2C_CTRL to logic LOW.
    // If you don't do this, there will be retries on poll in APB thread that can timeout i2c_transactions or other ECB packet retries
    if (0 == cmd_) {
        registers_->write(status_address_, I2C_IDLE);
        return;
    }
    i2c_cv_.notify_one();
}

void I2CController::stop()
{
    if (!is_running()) {
        return;
    }
    std::unique_lock<std::mutex> lock(i2c_mutex_);
    running_ = false;
    i2c_cv_.notify_one();
    lock.unlock(); // unlock before joining thread to avoid deadlock
    if (i2c_thread_.joinable()) {
        i2c_thread_.join();
    }
    // stop the peripherals, which may be no-op. Do not monitor for completion.
    for (auto& bus_map : i2c_bus_map_) {
        for (auto& peripheral : bus_map.second) {
            peripheral.second->stop();
        }
    }
}

bool I2CController::is_running()
{
    return running_;
}

void I2CController::attach_i2c_peripheral(uint32_t bus_address, uint8_t peripheral_address, I2CPeripheral* peripheral)
{
    std::unique_lock<std::mutex> lock(i2c_mutex_);
    if (running_) {
        throw std::runtime_error("I2CController: cannot attach peripherals while running");
    }

    // get peripheral map for the bus and default initialize if using bus address that is not in the map
    std::unordered_map<uint8_t, I2CPeripheral*>& i2c_peripheral_map = i2c_bus_map_[bus_address];

    auto it = i2c_peripheral_map.find(peripheral_address);
    if (it != i2c_peripheral_map.end()) {
        throw std::runtime_error("I2CController: peripheral address " + std::to_string(peripheral_address) + " is already in use");
    }
    i2c_peripheral_map[peripheral_address] = peripheral;
}

void I2CController::i2c_execute()
{
    // runs under the lock from the run() thread when this is executed
    registers_->write(status_address_, hololink::I2C_BUSY);
    // get the transaction size
    uint32_t num_bytes = registers_->read(num_bytes_address_);
    uint16_t num_bytes_write = static_cast<uint16_t>(num_bytes & 0xFFFF);
    uint16_t num_bytes_read = static_cast<uint16_t>((num_bytes >> 16) & 0xFFFF);
    // allocate buffers for i2c transaction
    std::vector<uint8_t> write_bytes(num_bytes_write);
    std::vector<uint8_t> read_bytes(num_bytes_read, 0);
    std::vector<uint32_t> read_values((num_bytes_read + sizeof(uint32_t) - 1) / sizeof(uint32_t), 0);
    I2CPeripheral* peripheral = nullptr; // nullptr peripherals will no-op i2c_transactions. this is to allow stateless camera emulation

    // find the peripheral. treat nullptr as a special case for a peripheral that consumes all data and returns 0s (user does not add any peripherals)
    // this will only work for stateless sensors
    uint32_t bus_address = registers_->read(bus_en_address_);
    std::unordered_map<uint8_t, I2CPeripheral*>& i2c_peripheral_map = i2c_bus_map_[bus_address];
    auto it = i2c_peripheral_map.find(peripheral_address_);
    if (it != i2c_peripheral_map.end()) {
        // peripheral will only receive i2c_transactions if it is registered
        peripheral = it->second;
    }

    // read from the i2c_buffer and copy to the i2c buffers.
    // exercise reads whether peripheral address has receiver or not to ensure expected read actions are taken
    const uint32_t num_write_values = (num_bytes_write + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    std::vector<uint32_t> write_values = registers_->read_range(data_buffer_address_, num_write_values);
    memcpy(write_bytes.data(), write_values.data(), num_bytes_write);

    I2CStatus status = I2CStatus::I2C_STATUS_SUCCESS;
    // send transaction to the peripheral
    // to allow for stateless sensors/no i2c peripheral associated, nullptr is allowed and will no-op the i2c_transaction() call
    if (peripheral) {
        status = peripheral->i2c_transaction(peripheral_address_, write_bytes, read_bytes);
    }
    if (status != I2CStatus::I2C_STATUS_SUCCESS) {
        // TODO: print messages based on status and write an error to the FSM status registers
        fprintf(stderr, "I2CController: i2c_transaction reports non-success status. %d\n", status);
        registers_->write(status_address_, hololink::I2C_I2C_ERR);
        return; // wrote error to I2C. Don't read data back
    }

    // guard against empty vectors where .data() is nullptr, violating memcpy requirements
    if (num_bytes_read) {
        memcpy(read_values.data(), read_bytes.data(), num_bytes_read);
        registers_->write_range(data_buffer_address_, read_values);
    }

    // update status register to done
    registers_->write(status_address_, hololink::I2C_DONE);

    // reset the command and peripheral address for next transaction
    cmd_ = 0;
    peripheral_address_ = 0;
}

// runs under the lock from the start() thread when this is executed
void I2CController::run()
{
    while (running_) {
        std::unique_lock<std::mutex> lock(i2c_mutex_);
        i2c_cv_.wait(lock, [this] { return !running_ || cmd_ != 0; });
        if (!running_) {
            break;
        }
        i2c_execute();
    }
}

const char* validate_configuration(const HSBConfiguration& config)
{
    if (config.sensor_count > MAX_SENSORS) {
        return "sensor_count exceeds MAX_SENSORS";
    }
    if (config.data_plane_count > MAX_DATA_PLANES) {
        return "data_plane_count exceeds MAX_DATA_PLANES";
    }
    if (config.sifs_per_sensor > MAX_SIFS_PER_SENSOR) {
        return "sifs_per_sensor exceeds MAX_SIFS_PER_SENSOR";
    }
    if (config.sensor_count * config.sifs_per_sensor > 32) {
        return "sifs count exceeds MAX_SIFS";
    }
    return nullptr;
}

// built-in I2C peripheral for Renesas I2C devices
class RenesasI2CPeripheral : public I2CPeripheral {
public:
    static constexpr uint8_t PERIPHERAL_ADDRESS = 0x09;
    RenesasI2CPeripheral() = default;

    I2CStatus i2c_transaction(uint8_t peripheral_address, const std::vector<uint8_t>& write_bytes, std::vector<uint8_t>& read_bytes) override
    {
        return I2CStatus::I2C_STATUS_SUCCESS;
    }

    void attach_to_i2c(I2CController& i2c_controller, uint8_t i2c_bus_address) override
    {
        i2c_controller.attach_i2c_peripheral(i2c_bus_address, PERIPHERAL_ADDRESS, this);
    }
};

HSBEmulator::HSBEmulator()
    : HSBEmulator(HSB_EMULATOR_CONFIG)
{
}

HSBEmulator::HSBEmulator(const HSBConfiguration& config)
    : configuration_(config)
{
    const char* config_error = validate_configuration(config);
    if (config_error) {
        throw std::runtime_error(config_error);
    }
    registers_ = std::make_shared<MemRegister>(configuration_);
    apb_event_handler_ = std::make_unique<APBEventHandler>(*this);
    I2CController* i2c_controller = new I2CController(*this, hololink::I2C_CTRL);
    i2c_controller_ = std::unique_ptr<I2CController>(i2c_controller);

    // attach peripherals after the i2c_controller is initialized
    renesas_i2c_ = std::make_unique<RenesasI2CPeripheral>();
    renesas_i2c_->attach_to_i2c(this->get_i2c(hololink::I2C_CTRL), hololink::BL_I2C_BUS);
}

HSBEmulator::~HSBEmulator()
{
}

void HSBEmulator::add_data_plane(DataPlane& data_plane)
{
    data_plane_list_.push_back(&data_plane);
}

void HSBEmulator::start()
{
    if (is_running()) {
        return;
    }

    i2c_controller_->start();
    apb_event_handler_->start();

    for (auto& data_plane : data_plane_list_) {
        data_plane->start();
    }
    // start the control plane thread
    running_ = true;
    control_thread_ = std::thread(&HSBEmulator::control_listen, this);

    // wait for the HSBEmulator control thread and all DataPlanes to start before returning
    while (!is_running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

/* this will block until all registered DataPlanes have stopped */
void HSBEmulator::stop()
{
    // if not running, do nothing...idempotent operation
    if (!is_running()) {
        return;
    }

    for (auto& data_plane : data_plane_list_) {
        data_plane->stop();
    }

    apb_event_handler_->stop();
    i2c_controller_->stop();
    // stop the control plane thread and wait for data planes to stop
    running_ = false;
    if (control_thread_.joinable()) {
        control_thread_.join();
    }
    while (is_running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    } // wait for data planes to stop
}

// returns true if any DataPlane is running or control plane thread, false otherwise
bool HSBEmulator::is_running()
{
    for (auto& data_plane : data_plane_list_) {
        if (data_plane->is_running()) {
            return true;
        }
    }
    if (apb_event_handler_->is_running()) {
        return true;
    }
    if (i2c_controller_->is_running()) {
        return true;
    }
    // only control plane left
    return running_;
}

void HSBEmulator::control_listen()
{
    int control_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (control_socket < 0) {
        fprintf(stderr, "Failed to create control UDP socket: %d - %s\n", errno, strerror(errno));
        throw std::runtime_error("Failed to create data channel");
    }
    uint16_t next_sequence = 0x100; // hololink.hpp starts here

    // Enable address reuse
    int reuse = 1;
    if (setsockopt(control_socket, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        fprintf(stderr, "Failed to set reuse address on control socket...socket will be set up with this option disabled: %d - %s\n", errno, strerror(errno));
    }

    // bind to all interfaces
    struct sockaddr_in control_addr = {
        .sin_family = AF_INET,
        .sin_port = htons(CONTROL_UDP_PORT),
        .sin_addr = {
            .s_addr = INADDR_ANY,
        }
    };
    if (bind(control_socket, (struct sockaddr*)&control_addr, sizeof(control_addr)) < 0) {
        fprintf(stderr, "Failed to bind control socket: %d - %s\n", errno, strerror(errno));
        close(control_socket);
        throw std::runtime_error("Failed to bind control socket");
    }

    struct pollfd fds[1] = { { .fd = control_socket, .events = POLL_RD_ANY, .revents = 0 } };
    while (running_ && !(fds[0].revents & POLLHUP)) {
        // poll on control socket for any events
        switch (poll(fds, 1, CONTROL_INTERVAL_MSEC)) {
        case 0:
            break;
        case -1:
            fprintf(stderr, "poll() failed. disconnecting: %d - %s\n", errno, strerror(errno));
            goto disconnect;
            break;
        default:
            if ((fds[0].revents & POLL_RD_ANY)) {
                struct sockaddr_in host_addr;
                socklen_t host_addr_len = sizeof(host_addr);

                // received and deserialize the control message buffer
                uint8_t message_buffer[CONTROL_PACKET_SIZE];
                uint16_t message_buffer_length = sizeof(message_buffer);
                const ssize_t message_length = recvfrom(control_socket, message_buffer, message_buffer_length, 0, (struct sockaddr*)&host_addr, &host_addr_len);

                if (message_length < 0) {
                    fprintf(stderr, "recvfrom failed: %d - %s\n", errno, strerror(errno));
                    continue;
                } else if (message_length < MIN_VALID_CONTROL_LENGTH) {
                    fprintf(stderr, "incomplete message received: %zu - %u\n", message_length, MIN_VALID_CONTROL_LENGTH);
                    continue;
                }

                // deserialize the control message
                struct ControlMessage message;
                hololink::core::Deserializer deserializer(message_buffer, message_length);
                if (!deserialize_control_message(deserializer, message)) {
                    fprintf(stderr, "deserialize_control_message failed on command %u\n", message.cmd_code);
                    continue;
                }

                // check sequence
                if (!(message.sequence == next_sequence)) {
                    //  perform sequence check handling if enabled
                    if (message.flags & hololink::REQUEST_FLAGS_SEQUENCE_CHECK) {
                        fprintf(stderr, "sequence check failed on command %u\n", message.cmd_code);
                        message.status = ECB_RESPONSE_CODE::SEQUENCE_ERROR;
                        handle_reply_message(message, message_buffer, message_buffer_length, control_socket, &host_addr, host_addr_len);
                        continue;
                    }
                }
                // regardless of REQUEST_FLAGS_SEQUENCE_CHECK, set the next sequence to an increment of previous sequence value
                next_sequence = message.sequence + 1;

                switch (message.cmd_code) {
                case ECB_CMD_CODE::RD_BYTE:
                case ECB_CMD_CODE::RD_WORD:
                case ECB_CMD_CODE::RD_DWORD:
                case ECB_CMD_CODE::RD_BLOCK:
                    handle_read_message(message, control_socket, &host_addr, host_addr_len);
                    break;
                case ECB_CMD_CODE::WR_BYTE:
                case ECB_CMD_CODE::WR_WORD:
                case ECB_CMD_CODE::WR_DWORD:
                case ECB_CMD_CODE::WR_BLOCK: {
                    handle_write_message(message, control_socket, &host_addr, host_addr_len);
                    break;
                }
                default: {
                    message.status = ECB_RESPONSE_CODE::COMMAND_ERROR;
                    break;
                }
                }

                handle_reply_message(message, message_buffer, message_buffer_length, control_socket, &host_addr, host_addr_len);

                // handle any errors or hangups
            } else if ((fds[0].revents & POLL_STOP)) {
                if ((fds[0].revents & POLLHUP)) {
                    fprintf(stderr, "connection hangup\n");
                } else if ((fds[0].revents & POLLERR)) {
                    fprintf(stderr, "poll error\n");
                } else if ((fds[0].revents & POLLNVAL)) {
                    fprintf(stderr, "invalid control socket\n");
                }
                goto disconnect;
            }
            break;
        }
    }
disconnect: // not really a connected socket, just a label to break out of the loop and cleanup
    close(control_socket);
}

void HSBEmulator::write(uint32_t address, uint32_t value)
{
    registers_->write(address, value);
    switch (address) {

    // TODO: each of these addresses should be a registered controller with separate handlers
    case hololink::CTRL_EVT_SW_EVENT:
        if (value == 1) {
            apb_event_handler_->queue_event(hololink::Hololink::Event::SW_EVENT);
        }
        break;
    case hololink::SPI_CTRL:
        handle_spi_control_write(address, value);
        break;
    case hololink::I2C_CTRL:
        i2c_controller_->execute(value);
        break;
    case hololink::FPGA_PTP_CTRL:
        if (value == 0) {
            this->write(hololink::FPGA_PTP_SYNC_STAT, 0);
        } else if (value == 0x3) {
            this->write(hololink::FPGA_PTP_SYNC_STAT, 0xF);
        }
        break;
    default:
        break;
    }
}

uint32_t HSBEmulator::read(uint32_t address)
{
    return registers_->read(address);
}

I2CController& HSBEmulator::get_i2c(uint32_t controller_address)
{
    return *i2c_controller_;
}

void HSBEmulator::handle_spi_control_write(uint32_t address, uint32_t value)
{
    switch (value) {
    case SPI_START:
        this->write(SPI_STATUS, SPI_DONE);
        break;
    case 0:
        this->write(SPI_STATUS, SPI_IDLE);
        break;
    default:
        break;
    }
}

// detect_poll() and handle_poll() are a workaround for handling i2c_transaction
// sequences.
// detect if host side application is expecting a response
// returns true if polling on the same address, false otherwise
bool HSBEmulator::detect_poll(uint32_t address)
{
    if (address != last_read_address_) {
        poll_count_ = 0;
        last_read_address_ = address;
        return false;
    }
    poll_count_++;
    if (poll_count_ >= POLL_COUNT_TRIGGER) {
        return true;
    }
    return false;
}

// if a poll is detected write to the status register that the transaction type is done
void HSBEmulator::handle_poll()
{
    if (poll_count_ >= POLL_COUNT_EXIT) {
        fprintf(stderr, "exceeded maximum polling...shutting down\n");
        raise(SIGINT);
    }
    switch (last_read_address_) {
    case hololink::SPI_CTRL:
        this->write(SPI_STATUS, SPI_DONE);
        break;
    case hololink::FPGA_PTP_SYNC_STAT:
        // 0xF is a locally defined constant (name: SYNCHRONIZED) in Hololink::p2p_synchronize()
        this->write(hololink::FPGA_PTP_SYNC_STAT, 0xF);
        break;
    default:
        break;
    }
}

void HSBEmulator::handle_reply_message(ControlMessage& message, uint8_t* message_buffer, size_t message_length, int control_socket, struct sockaddr_in* host_addr, socklen_t host_addr_len)
{
    if (message.flags & hololink::REQUEST_FLAGS_ACK_REQUEST) {
        hololink::core::Serializer serializer(message_buffer, message_length);
        if (!(message_length = serialize_reply_message(serializer, message))) {
            fprintf(stderr, "serialize_reply_message failed. no response sent\n");
        }

        if (sendto(control_socket, message_buffer, message_length, 0, (struct sockaddr*)host_addr, host_addr_len) < 0) {
            struct in_addr addr = ((struct sockaddr_in*)host_addr)->sin_addr;
            fprintf(stderr, "sendto in handle_reply_message failed: %d - %s - host_addr: %s\n", errno, strerror(errno), inet_ntoa(addr));
        }
    }
}

void HSBEmulator::handle_read_message(ControlMessage& message, int control_socket, struct sockaddr_in* host_addr, socklen_t host_addr_len)
{
    for (uint16_t i = 0; i < message.num_addresses; i++) {
        // handle polling before read as handle may affect read value
        if (detect_poll(message.addresses[i])) {
            handle_poll();
        }
        message.values[i] = this->read(message.addresses[i]);
    }

    message.cmd_code = 0x80 | message.cmd_code;
    message.status = ECB_RESPONSE_CODE::SUCCESS;
}

void HSBEmulator::handle_write_message(ControlMessage& message, int control_socket, struct sockaddr_in* host_addr, socklen_t host_addr_len)
{
    for (uint16_t i = 0; i < message.num_addresses; i++) {
        this->write(message.addresses[i], message.values[i]);
    }
    last_read_address_ = message.addresses[message.num_addresses - 1]; // needed for polling

    message.cmd_code = 0x80 | message.cmd_code;
    message.status = ECB_RESPONSE_CODE::SUCCESS;
}

}
