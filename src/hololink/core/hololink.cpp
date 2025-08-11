/*
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
 */

#include "hololink.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <semaphore.h>
#include <sys/socket.h>
#include <sys/stat.h>

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <thread>

#include "arp_wrapper.hpp"
#include "csi_controller.hpp"
#include "deserializer.hpp"
#include "logging_internal.hpp"
#include "metadata.hpp"
#include "serializer.hpp"

namespace hololink {

// SPI control flags
constexpr uint32_t SPI_START = 0b0000'0000'0000'0001;
// SPI status flags
constexpr uint32_t SPI_BUSY = 0b0000'0000'0000'0001;
[[maybe_unused]] constexpr uint32_t SPI_FSM_ERR = 0b0000'0000'0000'0010;
constexpr uint32_t SPI_DONE = 0b0000'0000'0001'0000;
// SPI_CFG
constexpr uint32_t SPI_CFG_CPOL = 0b0000'0000'0001'0000;
constexpr uint32_t SPI_CFG_CPHA = 0b0000'0000'0010'0000;

namespace {

    static std::map<std::string, std::shared_ptr<Hololink>> hololink_by_serial_number;

    // GPIO Registers
    // bitmask 0:1F, each bit corresponds to a GPIO pin
    // GPIO_OUTPUT_BASE_REGISTER    - W   - set output pin values
    // GPIO_DIRECTION_BASE_REGISTER - R/W - set/read GPIO pin direction
    // GPIO_STATUS_BASE_REGISTER    - R   - read input GPIO value
    //
    // FPGA can support up to 256 GPIO pins that are spread
    // on 8 OUTPUT/DIRECTION/STATUS registers.
    // For each type of register, the address offset is 4:
    // OUTPUT registers are:    0x0C(base),0x10,0x14,0x18....0x28
    // DIRECTION registers are: 0x2C(base),0x20,0x24,0x28....0x38
    // STATUS registers are:    0x8C(base),0x90,0x94,0x98....0xA8
    constexpr uint32_t GPIO_OUTPUT_BASE_REGISTER = 0x0000'000C;
    constexpr uint32_t GPIO_DIRECTION_BASE_REGISTER = 0x0000'002C;
    constexpr uint32_t GPIO_STATUS_BASE_REGISTER = 0x0000'008C;
    constexpr uint32_t GPIO_REGISTER_ADDRESS_OFFSET = 0x0000'0004;

    static char const* response_code_description(uint32_t response_code)
    {
        switch (response_code) {
        case RESPONSE_SUCCESS:
            return "RESPONSE_SUCCESS";
        case RESPONSE_ERROR_GENERAL:
            return "RESPONSE_ERROR_GENERAL";
        case RESPONSE_INVALID_ADDR:
            return "RESPONSE_INVALID_ADDR";
        case RESPONSE_INVALID_CMD:
            return "RESPONSE_INVALID_CMD";
        case RESPONSE_INVALID_PKT_LENGTH:
            return "RESPONSE_INVALID_PKT_LENGTH";
        case RESPONSE_INVALID_FLAGS:
            return "RESPONSE_INVALID_FLAGS";
        case RESPONSE_BUFFER_FULL:
            return "RESPONSE_BUFFER_FULL";
        case RESPONSE_INVALID_BLOCK_SIZE:
            return "RESPONSE_INVALID_BLOCK_SIZE";
        case RESPONSE_INVALID_INDIRECT_ADDR:
            return "RESPONSE_INVALID_INDIRECT_ADDR";
        case RESPONSE_COMMAND_TIMEOUT:
            return "RESPONSE_COMMAND_TIMEOUT";
        case RESPONSE_SEQUENCE_CHECK_FAIL:
            return "RESPONSE_SEQUENCE_CHECK_FAIL";
        default:
            return "(unknown)";
        }
    }

    // Allocate buffers for control plane requests and replies to this
    // size, which is guaranteed to be large enough for the largest
    // of any of those buffers.
    constexpr uint32_t CONTROL_PACKET_SIZE = 20;

} // anonymous namespace

Hololink::Hololink(
    const std::string& peer_ip, uint32_t control_port, const std::string& serial_number, bool sequence_number_checking, bool skip_sequence_initialization, bool ptp_enable, bool vsync_enable)
    : peer_ip_(peer_ip)
    , control_port_(control_port)
    , serial_number_(serial_number)
    , sequence_number_checking_(sequence_number_checking)
    , execute_mutex_()
    , control_event_apb_interrupt_enable_cache_(0)
    , control_event_rising_cache_(0)
    , control_event_falling_cache_(0)
    , async_event_thread_()
    , null_sequence_location_(0)
    , skip_sequence_initialization_(skip_sequence_initialization)
    , started_(false)
    , ptp_pps_output_(std::make_shared<PtpSynchronizer>(this[0]))
    , ptp_enable_(ptp_enable)
    , vsync_enable_(vsync_enable)
{
}

Hololink::~Hololink()
{
    // In case the user didn't call this directly
    stop();
}

/*static*/ std::shared_ptr<Hololink> Hololink::from_enumeration_metadata(const Metadata& metadata)
{
    auto serial_number = metadata.get<std::string>("serial_number");
    if (!serial_number) {
        throw std::runtime_error("Metadata has no \"serial_number\"");
    }

    std::shared_ptr<Hololink> r;

    auto it = hololink_by_serial_number.find(serial_number.value());
    if (it == hololink_by_serial_number.cend()) {
        auto peer_ip = metadata.get<std::string>("peer_ip");
        if (!peer_ip) {
            throw std::runtime_error("Metadata has no \"peer_ip\"");
        }
        auto control_port = metadata.get<int64_t>("control_port");
        if (!control_port) {
            throw std::runtime_error("Metadata has no \"control_port\"");
        }

        auto opt_sequence_number_checking = metadata.get<int64_t>("sequence_number_checking");
        bool sequence_number_checking = (opt_sequence_number_checking != 0);

        // Note that this will change to use UUIDs; board_id is deprecated and will be removed.
        // This particular section of code relies on the 100G configuration reporting a v1
        // BOOTP message, which includes board_id, so that value shows up here in enumeration
        // metadata.  Any device that shows up with a _v2 won't publish a board_id.
        bool skip_sequence_initialization = false;
        auto board_id = metadata.get<int64_t>("board_id");
        if (board_id && (board_id.value() == HOLOLINK_100G_BOARD_ID)) {
            skip_sequence_initialization = true;
        }
        bool ptp_enable = true;
        auto opt_ptp_enable = metadata.get<int64_t>("ptp_enable");
        if (opt_ptp_enable) {
            ptp_enable = opt_ptp_enable.value() != 0;
        }
        bool vsync_enable = true;
        auto opt_vsync_enable = metadata.get<int64_t>("vsync_enable");
        if (opt_vsync_enable) {
            vsync_enable = opt_vsync_enable.value() != 0;
        }
        r = std::make_shared<Hololink>(
            peer_ip.value(), control_port.value(), serial_number.value(), sequence_number_checking, skip_sequence_initialization, ptp_enable, vsync_enable);
        hololink_by_serial_number[serial_number.value()] = r;
    } else {
        r = it->second;
    }

    return r;
}

/*static*/ void Hololink::reset_framework()
{
    auto it = hololink_by_serial_number.begin();
    while (it != hololink_by_serial_number.end()) {
        HSB_LOG_INFO("Removing hololink \"{}\"", it->first);
        it = hololink_by_serial_number.erase(it);
    }
}

/*static*/ bool Hololink::enumerated(const Metadata& metadata)
{
    if (!metadata.get<std::string>("serial_number")) {
        return false;
    }
    if (!metadata.get<std::string>("peer_ip")) {
        return false;
    }
    if (!metadata.get<int64_t>("control_port")) {
        return false;
    }
    return true;
}

void Hololink::start()
{
    if (started_) {
        throw std::runtime_error("Hololink is already started.");
    }

    control_socket_.reset(socket(AF_INET, SOCK_DGRAM, 0));
    if (!control_socket_) {
        throw std::runtime_error("Failed to create socket");
    }
    // We listen to async event packets on this port.
    async_event_socket_.reset(socket(AF_INET, SOCK_DGRAM, 0));
    if (!async_event_socket_) {
        throw std::runtime_error("Failed to create async event socket");
    }
    // Assign a UDP port number
    struct sockaddr_in async_event_address { }; // fills with 0s
    async_event_address.sin_family = AF_INET;
    async_event_address.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(async_event_socket_.get(), (struct sockaddr*)&async_event_address, sizeof(async_event_address)) < 0) {
        throw std::runtime_error("Failed to bind async event socket");
    }
    // Get the actual port number
    socklen_t async_event_address_len = sizeof(async_event_address);
    if (getsockname(async_event_socket_.get(), (struct sockaddr*)&async_event_address, &async_event_address_len) < 0) {
        throw std::runtime_error("Failed to get async event port");
    }
    async_event_socket_port_ = ntohs(async_event_address.sin_port);
    async_event_thread_ = std::thread(&Hololink::async_event_thread, this);

    //
    configure_hsb();

    started_ = true;
}

void Hololink::stop()
{
    if (!started_) {
        return;
    }
    control_socket_.reset();
    // Wake up async_event_thread in a way that
    // makes it exit
    shutdown(async_event_socket_.get(), SHUT_RDWR);
    async_event_socket_.reset();
    async_event_thread_.join();
    started_ = false;
}

void Hololink::reset()
{
    std::shared_ptr<Spi> spi = get_spi(RESET_SPI_BUS, /*chip_select*/ 0, /*prescaler*/ 15,
        /*cpol*/ 0, /*cpha*/ 1, /*width*/ 1);

    //
    std::vector<uint8_t> write_command_bytes { 0x01, 0x07 };
    std::vector<uint8_t> write_data_bytes { 0x0C };
    uint32_t read_byte_count = 0;
    spi->spi_transaction(write_command_bytes, write_data_bytes, read_byte_count);
    //
    write_uint32(0x8, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    //
    write_data_bytes = { 0x0F };
    spi->spi_transaction(write_command_bytes, write_data_bytes, read_byte_count);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    //
    write_uint32(0x8, 0x3);
    try {
        // Because this drives the unit to reset,
        // we won't get a reply.
        write_uint32(0x4, 0x8, nullptr, /*retry*/ false);
    } catch (const std::exception& e) {
        HSB_LOG_INFO("ignoring error {}.", e.what());
    }

    // Now wait for the device to come back up.
    // This guy raises an exception if we're not found;
    // this can happen if set-ip is used in one-time
    // mode.
    Metadata channel_metadata = Enumerator::find_channel(peer_ip_, std::make_shared<Timeout>(30.f));

    // When the connection was lost, the host flushes its ARP cache.
    // Because ARP requests are slow, let's just set the ARP cache here,
    // because we know the MAC ID and the IP address of the system that
    // just enumerated.  This avoids timeouts when we try fetching the FPGA
    // version ID while the kernel is waiting for ARP to be updated.
    std::string interface = channel_metadata.get<std::string>("interface").value();
    std::string client_ip_address = channel_metadata.get<std::string>("client_ip_address").value();
    std::string mac_id = channel_metadata.get<std::string>("mac_id").value();
    hololink::core::ArpWrapper::arp_set(control_socket_.get(), interface.c_str(), client_ip_address.c_str(), mac_id.c_str());

    // Necessary for e.g. I2C or SPI to work.
    configure_hsb();

    // Now go through and reset all registered clients.
    for (std::shared_ptr<ResetController> reset_controller : reset_controllers_) {
        reset_controller->reset();
    }
}

uint32_t Hololink::get_hsb_ip_version(const std::shared_ptr<Timeout>& timeout, bool sequence_check)
{
    const uint32_t version = read_uint32(HSB_IP_VERSION, timeout, sequence_check);
    return version;
}

uint32_t Hololink::get_fpga_date()
{
    const uint32_t date = read_uint32(FPGA_DATE);
    return date;
}

bool Hololink::write_uint32(
    uint32_t address, uint32_t value, const std::shared_ptr<Timeout>& in_timeout, bool retry, bool sequence_check)
{
    uint32_t count = 0;
    std::exception_ptr eptr;
    std::shared_ptr<Timeout> timeout = Timeout::default_timeout(in_timeout);
    bool current_sequence_check = sequence_check;
    try {
        // HSB only supports a single command/response at a time--
        // in other words we need to inhibit other threads from sending
        // a command until we receive the response for the current one.
        std::lock_guard lock(execute_mutex_);
        const uint16_t sequence = next_sequence(lock);
        while (true) {
            count += 1;
            bool status = write_uint32_(address, value, timeout, retry, sequence, current_sequence_check, lock);
            if (status) {
                return status;
            }
            if (!retry) {
                break;
            }
            if (!timeout->retry()) {
                throw TimeoutError(
                    fmt::format("write_uint32 address={:#x} value={:#x}", address, value));
            }
            current_sequence_check = false;
        }
    } catch (...) {
        eptr = std::current_exception();
    }

    assert(count > 0);
    add_write_retries(count - 1);

    if (eptr) {
        std::rethrow_exception(eptr);
    }

    return false;
}

bool Hololink::write_uint32_(uint32_t address, uint32_t value,
    const std::shared_ptr<Timeout>& timeout, bool response_expected, uint16_t sequence, bool sequence_check, std::lock_guard<std::mutex>& lock)
{
    HSB_LOG_DEBUG("write_uint32(address={:#x}, value={:#x})", address, value);
    if ((address & 3) != 0) {
        throw std::runtime_error(
            fmt::format("Invalid address \"{:#x}\", has to be a multiple of four", address));
    }
    // BLOCKING on ack or timeout
    // This routine serializes a write_uint32 request
    // and forwards it to the device.

    // Serialize
    std::vector<uint8_t> request(CONTROL_PACKET_SIZE);
    core::Serializer serializer(request.data(), request.size());
    uint8_t flags = REQUEST_FLAGS_ACK_REQUEST;
    if (sequence_check) {
        flags |= REQUEST_FLAGS_SEQUENCE_CHECK;
    }
    if (!(serializer.append_uint8(WR_DWORD) && serializer.append_uint8(flags)
            && serializer.append_uint16_be(sequence) && serializer.append_uint8(0) // reserved
            && serializer.append_uint8(0) // reserved
            && serializer.append_uint32_be(address) && serializer.append_uint32_be(value))) {
        throw std::runtime_error("Unable to serialize");
    }
    request.resize(serializer.length());

    std::vector<uint8_t> reply(CONTROL_PACKET_SIZE);
    auto [status, optional_response_code, deserializer] = execute(sequence, request, reply, timeout, lock);
    if (!status) {
        // timed out
        return false;
    }
    if (optional_response_code != RESPONSE_SUCCESS) {
        if (!optional_response_code.has_value()) {
            if (response_expected) {
                HSB_LOG_ERROR(
                    "write_uint32 address={:#X} value={:#X} response_code=None", address, value);
                return false;
            }
        }
        uint32_t response_code = optional_response_code.value();
        throw std::runtime_error(
            fmt::format("write_uint32 address={:#X} value={:#X} response_code={:#X}({})", address,
                value, response_code, response_code_description(response_code)));
    }
    return true;
}

uint32_t Hololink::read_uint32(uint32_t address, const std::shared_ptr<Timeout>& in_timeout, bool sequence_check)
{
    uint32_t count = 0;
    std::exception_ptr eptr;
    std::shared_ptr<Timeout> timeout = Timeout::default_timeout(in_timeout);
    bool current_sequence_check = sequence_check;
    try {
        // HSB only supports a single command/response at a time--
        // in other words we need to inhibit other threads from sending
        // a command until we receive the response for the current one.
        std::lock_guard lock(execute_mutex_);
        const uint16_t sequence = next_sequence(lock);
        while (true) {
            count += 1;
            auto [status, value] = read_uint32_(address, timeout, sequence, current_sequence_check, lock);
            if (status) {
                return value.value();
            }
            if (!timeout->retry()) {
                throw TimeoutError(fmt::format("read_uint32 address={:#x}", address));
            }
            current_sequence_check = false;
        }
    } catch (...) {
        eptr = std::current_exception();
    }

    assert(count > 0);
    add_read_retries(count - 1);

    if (eptr) {
        std::rethrow_exception(eptr);
    }

    return 0;
}

std::tuple<bool, std::optional<uint32_t>> Hololink::read_uint32_(
    uint32_t address, const std::shared_ptr<Timeout>& timeout, uint16_t sequence, bool sequence_check, std::lock_guard<std::mutex>& lock)
{
    HSB_LOG_DEBUG("read_uint32(address={:#x})", address);
    if ((address & 3) != 0) {
        throw std::runtime_error(
            fmt::format("Invalid address \"{:#x}\", has to be a multiple of four", address));
    }
    // BLOCKING on ack or timeout
    // This routine serializes a read_uint32 request
    // and forwards it to the device.

    // Serialize
    std::vector<uint8_t> request(CONTROL_PACKET_SIZE);
    core::Serializer serializer(request.data(), request.size());
    uint8_t flags = REQUEST_FLAGS_ACK_REQUEST;
    if (sequence_check) {
        flags |= REQUEST_FLAGS_SEQUENCE_CHECK;
    }
    if (!(serializer.append_uint8(RD_DWORD) && serializer.append_uint8(flags)
            && serializer.append_uint16_be(sequence) && serializer.append_uint8(0) // reserved
            && serializer.append_uint8(0) // reserved
            && serializer.append_uint32_be(address))) {
        throw std::runtime_error("Unable to serialize");
    }
    request.resize(serializer.length());
    HSB_LOG_TRACE("read_uint32: {}....{}", request, sequence);

    std::vector<uint8_t> reply(CONTROL_PACKET_SIZE);
    auto [status, optional_response_code, deserializer] = execute(sequence, request, reply, timeout, lock);
    if (!status) {
        // timed out
        return { false, {} };
    }
    if (optional_response_code != RESPONSE_SUCCESS) {
        uint32_t response_code = optional_response_code.value();
        throw std::runtime_error(
            fmt::format("read_uint32 response_code={}({})", response_code, response_code_description(response_code)));
    }
    uint8_t reserved;
    uint32_t response_address;
    uint32_t value;
    uint16_t latched_sequence;
    if (!(deserializer->next_uint8(reserved) /* reserved */
            && deserializer->next_uint32_be(response_address) /* address */
            && deserializer->next_uint32_be(value)
            && deserializer->next_uint16_be(latched_sequence))) {
        throw std::runtime_error("Unable to deserialize");
    }
    assert(response_address == address);
    HSB_LOG_DEBUG("read_uint32(address={:#x})={:#x}", address, value);
    return { true, value };
}

uint16_t Hololink::next_sequence(std::lock_guard<std::mutex>&)
{
    uint16_t r = sequence_;
    sequence_ = sequence_ + 1;
    return r;
}

std::tuple<bool, std::optional<uint32_t>, std::shared_ptr<core::Deserializer>> Hololink::execute(
    uint16_t sequence, const std::vector<uint8_t>& request, std::vector<uint8_t>& reply,
    const std::shared_ptr<Timeout>& timeout, std::lock_guard<std::mutex>&)
{
    HSB_LOG_TRACE("Sending request={}", request);
    double request_time = Timeout::now_s();

    send_control(request);
    while (true) {
        reply = receive_control(timeout);
        double reply_time = Timeout::now_s();
        executed(request_time, request, reply_time, reply);
        if (reply.empty()) {
            // timed out
            return { false, {}, nullptr };
        }
        auto deserializer = std::make_shared<core::Deserializer>(reply.data(), reply.size());
        uint8_t dummy;
        uint16_t reply_sequence = 0;
        uint8_t response_code = 0;
        if (!(deserializer->next_uint8(dummy) /* reply_cmd_code */
                && deserializer->next_uint8(dummy) /* reply_flags */
                && deserializer->next_uint16_be(reply_sequence)
                && deserializer->next_uint8(response_code))) {
            throw std::runtime_error("Unable to deserialize");
        }
        HSB_LOG_TRACE("reply reply_sequence={} response_code={}({}) sequence={}", reply_sequence,
            response_code, response_code_description(response_code), sequence);
        if (sequence == reply_sequence) {
            return { true, response_code, deserializer };
        }
    }
}

void Hololink::send_control(const std::vector<uint8_t>& request)
{
    HSB_LOG_TRACE(
        "_send_control request={} peer_ip={} control_port={}", request, peer_ip_, control_port_);
    sockaddr_in address {};
    address.sin_family = AF_INET;
    address.sin_port = htons(control_port_);
    if (inet_pton(AF_INET, peer_ip_.c_str(), &address.sin_addr) == 0) {
        throw std::runtime_error(fmt::format("Failed to convert address {}", peer_ip_));
    }

    if (sendto(control_socket_.get(), request.data(), request.size(), 0, (sockaddr*)&address,
            sizeof(address))
        < 0) {
        throw std::runtime_error(
            fmt::format("sendto failed with errno={}: \"{}\"", errno, strerror(errno)));
    }
}

std::vector<uint8_t> Hololink::receive_control(const std::shared_ptr<Timeout>& timeout)
{
    timeval timeout_value {};
    while (true) {
        // if there had been a timeout provided check if it had expired, if not set the timeout of
        // the select() call to the remaining time
        timeval* select_timeout;
        if (timeout) {
            if (timeout->expired()) {
                return {};
            }
            // convert floating point time to integer seconds and microseconds
            const double trigger_s = timeout->trigger_s();
            double integral = 0.f;
            double fractional = std::modf(trigger_s, &integral);
            timeout_value.tv_sec = (__time_t)(integral);
            timeout_value.tv_usec
                = (__suseconds_t)(fractional * std::micro().den / std::micro().num);
            select_timeout = &timeout_value;
        } else {
            select_timeout = nullptr;
        }

        fd_set r;
        FD_ZERO(&r);
        FD_SET(control_socket_.get(), &r);
        fd_set x = r;
        int result = select(control_socket_.get() + 1, &r, nullptr, &x, select_timeout);
        if (result == -1) {
            throw std::runtime_error(
                fmt::format("select failed with errno={}: \"{}\"", errno, strerror(errno)));
        }
        if (FD_ISSET(control_socket_.get(), &x)) {
            HSB_LOG_ERROR("Error reading enumeration sockets.");
            return {};
        }
        if (result == 0) {
            // Timed out
            continue;
        }

        std::vector<uint8_t> received(hololink::core::UDP_PACKET_SIZE);
        sockaddr_in peer_address {};
        peer_address.sin_family = AF_UNSPEC;
        socklen_t peer_address_len = sizeof(peer_address);
        ssize_t received_bytes;
        do {
            received_bytes = recvfrom(control_socket_.get(), received.data(), received.size(), 0,
                (sockaddr*)&peer_address, &peer_address_len);
            if (received_bytes == -1) {
                throw std::runtime_error(
                    fmt::format("recvfrom failed with errno={}: \"{}\"", errno, strerror(errno)));
            }
        } while (received_bytes <= 0);

        received.resize(received_bytes);
        return received;
    }
}

void Hololink::executed(double request_time, const std::vector<uint8_t>& request, double reply_time,
    const std::vector<uint8_t>& reply)
{
    HSB_LOG_TRACE("Got reply={}", reply);
}

void Hololink::add_read_retries(uint32_t n) { }

void Hololink::add_write_retries(uint32_t n) { }

void Hololink::write_renesas(I2c& i2c, const std::vector<uint8_t>& data)
{
    HSB_LOG_TRACE("write_renesas data={}", data);
    uint32_t read_byte_count = 0;
    constexpr uint32_t RENESAS_I2C_ADDRESS = 0x09;
    std::vector<uint8_t> reply = i2c.i2c_transaction(RENESAS_I2C_ADDRESS, data, read_byte_count);
    HSB_LOG_TRACE("reply={}.", reply);
}

void Hololink::setup_clock(const std::vector<std::vector<uint8_t>>& clock_profile)
{
    // set the clock driver.
    std::shared_ptr<Hololink::I2c> i2c = get_i2c(BL_I2C_BUS);
    i2c->set_i2c_clock();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    //
    for (auto&& data : clock_profile) {
        write_renesas(*i2c, data);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // enable the clock synthesizer and output
    write_uint32(0x8, 0x30);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // enable camera power.
    write_uint32(0x8, 0x03);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    i2c->set_i2c_clock();
}

class Spi
    : public Hololink::Spi {
public:
    Spi(Hololink& hololink, uint32_t bus_number, uint32_t chip_select,
        uint32_t prescaler, uint32_t cpol, uint32_t cpha, uint32_t width,
        uint32_t spi_address)
        : hololink_(hololink)
        , reg_control_(spi_address + 0)
        , reg_bus_en_(spi_address + 4)
        , reg_num_bytes_(spi_address + 8)
        , reg_spi_mode_(spi_address + 0xC)
        , spi_mode_(spi_mode(prescaler, cpol, cpha, width))
        , reg_num_cmd_bytes_(spi_address + 0x10)
        , reg_status_(spi_address + 0x80)
        , reg_data_buffer_(spi_address + 0x100)
        , bus_number_(bus_number)
        , turnaround_cycles_(0)
    {
    }

    static uint32_t spi_mode(uint32_t prescaler, uint32_t cpol, uint32_t cpha, uint32_t width)
    {
        if (prescaler >= 16) {
            throw std::runtime_error(
                fmt::format("Invalid prescaler \"{}\", has to be less than 16", prescaler));
        }
        std::map<uint32_t, uint32_t> width_map {
            { 1, 0 },
            { 2, 1 << 8 },
            { 4, 2 << 8 },
        };
        // we let this next statement raise an
        // exception if the width parameter isn't
        // supported.
        uint32_t r = prescaler | width_map[width];
        if (cpol) {
            r |= SPI_CFG_CPOL;
        }
        if (cpha) {
            r |= SPI_CFG_CPHA;
        }
        return r;
    }

    void write_uint32(uint32_t address, uint32_t value, const std::shared_ptr<Timeout>& timeout)
    {
        bool ok = hololink_.write_uint32(address, value, timeout);
        if (!ok) {
            throw std::runtime_error(
                fmt::format("ACK failure writing SPI register {:#x}.", address));
        }
    }

    uint32_t read_uint32(uint32_t address, const std::shared_ptr<Timeout>& timeout)
    {
        return hololink_.read_uint32(address, timeout);
    }

    Hololink::NamedLock& spi_lock()
    {
        return hololink_.spi_lock();
    }

    std::vector<uint8_t> spi_transaction(const std::vector<uint8_t>& write_command_bytes,
        const std::vector<uint8_t>& write_data_bytes, uint32_t read_byte_count,
        const std::shared_ptr<Timeout>& in_timeout) override
    {
        const uint32_t write_command_count = write_command_bytes.size();
        if (write_command_count >= 16) { // available bits in num_bytes2
            throw std::runtime_error(
                fmt::format("Size of combined write_command_bytes and write_data_bytes is too large: "
                            "\"{}\", has to be less than 16",
                    write_command_count));
        }
        std::vector<uint8_t> write_bytes(write_command_bytes);
        write_bytes.insert(write_bytes.end(), write_data_bytes.begin(), write_data_bytes.end());
        const uint32_t write_byte_count = write_bytes.size();
        const uint32_t buffer_size = 256;
        // The buffer needs to have enough space for the written bytes (command + write)
        if (write_byte_count > buffer_size) {
            throw std::runtime_error(fmt::format("Size of write is too large: "
                                                 "\"{:#x}\", has to be less than {:#x}",
                write_byte_count, buffer_size));
        }
        // The buffer needs to have enough space for the read bytes (command + read)
        const uint32_t read_byte_total = write_command_count + read_byte_count;
        if (read_byte_total > buffer_size) {
            throw std::runtime_error(fmt::format("Size of read is too large: "
                                                 "\"{:#x}\", has to be less than {:#x}",
                read_byte_total, buffer_size));
        }
        // Valid "turnaround_cycles_"?
        if (turnaround_cycles_ >= 256) {
            throw std::runtime_error(fmt::format("turnaround_cycles_={} must be less than 256. ", turnaround_cycles_));
        }
        // Serialize access to the SPI controller.
        std::lock_guard lock(spi_lock());
        std::shared_ptr<Timeout> timeout = Timeout::spi_timeout(in_timeout);
        // Hololink FPGA doesn't support resetting the SPI interface;
        // so the best we can do is see that it's not busy.
        uint32_t value = read_uint32(reg_status_, timeout);
        if (value & SPI_BUSY) {
            // This would be because either the SPI port in the FPGA is broken
            // or another program is using this device.
            throw std::runtime_error("Unexpected state; SPI port is busy.");
        }
        // Clear the start bit in case a program crashed and didn't reset this.
        write_uint32(reg_control_, 0, timeout);
        // We should only read 0 in SPI_DONE
        value = read_uint32(reg_status_, timeout);
        if (value & SPI_DONE) {
            throw std::runtime_error("Unexpected state; SPI port indicates invalid DONE.");
        }
        // Set the configuration
        write_uint32(reg_spi_mode_, spi_mode_, timeout);
        // Set the buffer
        const size_t remaining = write_bytes.size();
        for (size_t index = 0; index < remaining; index += 4) {
            value = write_bytes[index] << 0;
            if ((index + 1) < remaining) {
                value |= (write_bytes[index + 1] << 8);
            }
            if ((index + 2) < remaining) {
                value |= (write_bytes[index + 2] << 16);
            }
            if ((index + 3) < remaining) {
                value |= (write_bytes[index + 3] << 24);
            }
            write_uint32(reg_data_buffer_ + index, value, timeout);
        }
        // write the num_bytes; note that these are 9-bit values that top
        // out at (buffer_size=256) (length checked above)
        const uint32_t num_bytes = (write_byte_count << 0) | (read_byte_count << 16);
        write_uint32(reg_num_bytes_, num_bytes, timeout);
        const uint32_t num_cmd_bytes = turnaround_cycles_ | (write_command_count << 8);
        write_uint32(reg_num_cmd_bytes_, num_cmd_bytes, timeout);
        write_uint32(reg_bus_en_, bus_number_, timeout);
        // start the SPI transaction.
        write_uint32(reg_control_, SPI_START, timeout);
        // wait until we see done, which may be immediately
        while (true) {
            value = read_uint32(reg_status_, timeout);
            if (value & SPI_DONE) {
                break;
            }
            if (!timeout->retry()) {
                // timed out
                HSB_LOG_DEBUG("Timed out.");
                throw TimeoutError(fmt::format("spi_transaction control={:#x}", reg_control_));
            }
        }
        // <WORKAROUND> Data doesn't show up until we clear done
        // Clear the done bit.
        hololink_.write_uint32(reg_control_, 0);
        // </WORKAROUND>
        // round up to get the whole next word
        std::vector<uint8_t> r(read_byte_total + 3);
        for (uint32_t i = 0; i < read_byte_total; i += 4) {
            value = read_uint32(reg_data_buffer_ + i, timeout);
            r[i + 0] = (value >> 0) & 0xFF;
            r[i + 1] = (value >> 8) & 0xFF;
            r[i + 2] = (value >> 16) & 0xFF;
            r[i + 3] = (value >> 24) & 0xFF;
        }
        r = std::vector<uint8_t>(
            r.cbegin(), r.cbegin() + read_byte_total);
        // Clear the done bit.
        hololink_.write_uint32(reg_control_, 0, timeout);
        return r;
    }

private:
    Hololink& hololink_;
    const uint32_t reg_control_;
    const uint32_t reg_bus_en_;
    const uint32_t reg_num_bytes_;
    const uint32_t reg_spi_mode_;
    const uint32_t spi_mode_;
    const uint32_t reg_num_cmd_bytes_;
    const uint32_t reg_status_;
    const uint32_t reg_data_buffer_;
    uint32_t bus_number_;
    uint32_t turnaround_cycles_;
};

std::shared_ptr<Hololink::Spi> Hololink::get_spi(uint32_t bus_number, uint32_t chip_select,
    uint32_t prescaler, uint32_t cpol, uint32_t cpha, uint32_t width, uint32_t spi_address)
{
    return std::make_shared<hololink::Spi>(*this, bus_number, chip_select, prescaler, cpol, cpha, width, spi_address);
}

class SoftwareSequencer
    : public Hololink::Sequencer {
public:
    SoftwareSequencer(Hololink& hololink)
        : Hololink::Sequencer()
        , hololink_(hololink)
    {
    }

    void enable() override
    {
        done();
        auto event = Hololink::Event::SW_EVENT;
        assign_location(event);
        write(hololink_);
        hololink_.configure_apb_event(event, location());
    }

public:
    Hololink& hololink_;
};

std::shared_ptr<Hololink::Sequencer> Hololink::software_sequencer()
{
    auto r = std::make_shared<SoftwareSequencer>(this[0]);
    return r;
}

class I2c
    : public Hololink::I2c {

public:
    I2c(Hololink& hololink, uint32_t i2c_bus, uint32_t i2c_address)
        : hololink_(hololink)
        , reg_control_(i2c_address + 0)
        , reg_bus_en_(i2c_address + 4)
        , reg_num_bytes_(i2c_address + 8)
        , reg_clk_cnt_(i2c_address + 0xC)
        , reg_status_(i2c_address + 0x80)
        , reg_data_buffer_(i2c_address + 0x100)
        , bus_en_(i2c_bus)
    {
    }

    void write_uint32(uint32_t address, uint32_t value, const std::shared_ptr<Timeout>& timeout)
    {
        bool ok = hololink_.write_uint32(address, value, timeout);
        if (!ok) {
            throw std::runtime_error(
                fmt::format("ACK failure writing I2C register {:#x}.", address));
        }
    }

    uint32_t read_uint32(uint32_t address, const std::shared_ptr<Timeout>& timeout)
    {
        return hololink_.read_uint32(address, timeout);
    }

    void write_uint32(uint32_t address, uint32_t value)
    {
        bool ok = hololink_.write_uint32(address, value);
        if (!ok) {
            throw std::runtime_error(
                fmt::format("ACK failure writing I2C register {:#x}.", address));
        }
    }

    uint32_t read_uint32(uint32_t address)
    {
        return hololink_.read_uint32(address);
    }

    bool set_i2c_clock() override
    {
        // set the clock to 400KHz (fastmode) i2c speed once at init
        const uint32_t clock = 0x19;
        write_uint32(reg_clk_cnt_, clock, Timeout::i2c_timeout());
        return true;
    }

    // 8-bit peripheral_i2c_address
    std::vector<uint8_t> i2c_transaction(uint32_t peripheral_i2c_address,
        const std::vector<uint8_t>& write_bytes, uint32_t read_byte_count,
        const std::shared_ptr<Timeout>& in_timeout,
        bool ignore_nak) override
    {
        HSB_LOG_DEBUG("i2c_transaction(peripheral_i2c_address={:#x}, write_byte.size={}, read_byte_count={}.",
            peripheral_i2c_address, write_bytes.size(), read_byte_count);
        auto sequencer = hololink_.software_sequencer();
        auto [write_indexes, read_indexes, status_index] = encode_i2c_request(
            *sequencer, peripheral_i2c_address, write_bytes, read_byte_count);
        std::lock_guard lock(i2c_lock());
        sequencer->enable();
        hololink_.configure_apb_event(Hololink::Event::I2C_BUSY);
        write_uint32(CTRL_EVT_SW_EVENT, 1);
        write_uint32(CTRL_EVT_SW_EVENT, 0);
        uint32_t status_cache = sequencer->location() + status_index * 4;
        std::shared_ptr<Timeout> timeout = Timeout::i2c_timeout(in_timeout);
        if (timeout->trigger_s() > APB_TIMEOUT_MAX) {
            write_uint32(CTRL_EVT_APB_TIMEOUT,
                static_cast<uint32_t>(APB_TIMEOUT_MAX * APB_TIMEOUT_SCALE));
        } else {
            write_uint32(CTRL_EVT_APB_TIMEOUT,
                static_cast<uint32_t>(timeout->trigger_s() * APB_TIMEOUT_SCALE));
        }
        // Poll until done.  Future version will have an event packet too.
        uint32_t value;
        while (true) {
            value = hololink_.read_uint32(status_cache, timeout);
            HSB_LOG_DEBUG("status_cache={:#x}.", value);
            if (value & I2C_DONE) {
                break;
            }
            if (!timeout->retry()) {
                // timed out
                HSB_LOG_DEBUG("Timed out.");
                throw TimeoutError(
                    fmt::format("i2c_transaction i2c_address={:#x}", peripheral_i2c_address));
            }
        }
        hololink_.clear_apb_event(Hololink::Event::I2C_BUSY);
        // Check for errors.
        if (value & I2C_FSM_ERR) {
            throw std::runtime_error("I2C port indicates I2C_FSM_ERR.");
        }
        if (value & I2C_I2C_ERR) {
            throw std::runtime_error("I2C port indicates I2C_I2C_ERR.");
        }
        std::vector<uint8_t> r;
        if (value & I2C_I2C_NAK) {
            if (!ignore_nak) {
                throw std::runtime_error("I2C port indicates I2C_I2C_NAK.");
            }
            HSB_LOG_DEBUG("Ignoring I2C NAK.");
            return r;
        }
        // round up to get the whole next word
        const uint32_t word_count = (read_byte_count + 3) / 4;
        // we should have one read_index for each word
        assert(word_count == read_indexes.size());
        r.reserve(word_count * 4);
        for (const auto& item : read_indexes) {
            uint32_t address = sequencer->location() + item * 4;
            value = hololink_.read_uint32(address, timeout);
            r.push_back((value >> 0) & 0xFF);
            r.push_back((value >> 8) & 0xFF);
            r.push_back((value >> 16) & 0xFF);
            r.push_back((value >> 24) & 0xFF);
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

    /**
     * @returns {write_indexes, read_indexes, status_index}
     */
    std::tuple<std::vector<unsigned>, std::vector<unsigned>, unsigned> encode_i2c_request(
        Hololink::Sequencer& sequencer,
        uint32_t peripheral_i2c_address,
        const std::vector<uint8_t>& write_bytes, uint32_t read_byte_count) override
    {
        if (peripheral_i2c_address > 0x7F) {
            throw std::runtime_error(
                fmt::format("Invalid peripheral_i2c_address={:#x}, only 7-bit addresses are supported", peripheral_i2c_address));
        }
        unsigned write_byte_count = write_bytes.size();
        if (write_byte_count > 0x100) {
            throw std::runtime_error(
                fmt::format("Write buffer size={:#x} is too large.", write_byte_count));
        }
        if (read_byte_count > 0x100) {
            throw std::runtime_error(
                fmt::format("Read buffer size={:#x} is too large.", read_byte_count));
        }
        // Clear the control register; this lowers done (which should be done already)
        sequencer.write_uint32(reg_control_, 0);
        // Write the buffer
        unsigned words = write_byte_count / 4; // round down
        uint32_t b = 0;
        std::vector<unsigned> write_indexes;
        write_indexes.reserve(words + 1); // round up
        for (unsigned i = 0; i < words; i++) {
            uint32_t value = write_bytes[b];
            value |= write_bytes[b + 1] << 8;
            value |= write_bytes[b + 2] << 16;
            value |= write_bytes[b + 3] << 24;
            unsigned index = sequencer.write_uint32(reg_data_buffer_ + b, value);
            b += 4;
            write_indexes.push_back(index);
        }
        unsigned remaining = write_byte_count & 3;
        if (remaining) {
            uint32_t value = write_bytes[b];
            if (remaining > 1) {
                value |= write_bytes[b + 1] << 8;
            }
            if (remaining > 2) {
                value |= write_bytes[b + 2] << 16;
            }
            unsigned index = sequencer.write_uint32(reg_data_buffer_ + b, value);
            write_indexes.push_back(index);
        }
        uint32_t num_bytes = (write_byte_count << 0) | (read_byte_count << 16);
        sequencer.write_uint32(reg_num_bytes_, num_bytes);
        sequencer.write_uint32(reg_bus_en_, bus_en_);
        uint32_t control = (peripheral_i2c_address << 16) | I2C_START;
        // When microcode execution gets here, I2C starts.
        sequencer.write_uint32(reg_control_, control);
        // Put in a POLL instruction to wait for it to be done.
        sequencer.poll(reg_status_, I2C_DONE, I2C_DONE);
        // Cache the status register; this is where we look to see that we're done.
        uint32_t initial_value = 0;
        unsigned status_index = sequencer.read_uint32(reg_status_, initial_value);
        // Show how to fetch the result
        std::vector<unsigned> read_indexes;
        words = (read_byte_count + 3) / 4;
        read_indexes.reserve(words);
        b = 0;
        for (unsigned i = 0; i < words; i++) {
            unsigned index = sequencer.read_uint32(reg_data_buffer_ + b);
            read_indexes.push_back(index);
            b += 4;
        }
        // Finally, write a 0 to the control register.
        sequencer.write_uint32(reg_control_, 0);
        return { write_indexes, read_indexes, status_index };
    }

private:
    Hololink& hololink_;
    const uint32_t reg_control_;
    const uint32_t reg_bus_en_;
    const uint32_t reg_num_bytes_;
    const uint32_t reg_clk_cnt_;
    const uint32_t reg_status_;
    const uint32_t reg_data_buffer_;
    const uint32_t bus_en_;
};

std::shared_ptr<Hololink::I2c> Hololink::get_i2c(uint32_t i2c_bus, uint32_t i2c_address)
{
    return std::make_shared<hololink::I2c>(*this, i2c_bus, i2c_address);
}

std::shared_ptr<Hololink::GPIO> Hololink::get_gpio(Metadata& metadata)
{
    // How many I/O pins are configured?  Note that this will crash
    // if the board doesn't have any.
    auto gpio_pin_count_value = metadata.get<int64_t>("gpio_pin_count");
    if (!gpio_pin_count_value) {
        throw std::runtime_error("GPIO is not supported on this Hololink board!");
    }
    int gpio_pin_count = gpio_pin_count_value.value();
    // Upper bound check is against an arbitrary limit; adjust this if
    // we actually see configurations that warrant it.
    if ((gpio_pin_count < 0) || (gpio_pin_count > 1000)) {
        throw std::runtime_error(fmt::format("Invalid gpio_pin_count={}", gpio_pin_count));
    }
    auto r = std::make_shared<Hololink::GPIO>(*this, static_cast<uint32_t>(gpio_pin_count));
    return r;
}

Hololink::GPIO::GPIO(Hololink& hololink, uint32_t gpio_pin_number)
    : hololink_(hololink)
{
    if (gpio_pin_number > GPIO_PIN_RANGE) {
        HSB_LOG_ERROR("Number of GPIO pins requested={} exceeds system limits={}", gpio_pin_number, GPIO_PIN_RANGE);
        throw std::runtime_error(fmt::format("Number of GPIO pins requested={} exceeds system limits={}", gpio_pin_number, GPIO_PIN_RANGE));
    }

    gpio_pin_number_ = gpio_pin_number;
}

void Hololink::GPIO::set_direction(uint32_t pin, uint32_t direction)
{
    if (pin < gpio_pin_number_) {

        uint32_t register_address = GPIO_DIRECTION_BASE_REGISTER + ((pin / 32) * GPIO_REGISTER_ADDRESS_OFFSET);
        uint32_t pin_bit = pin % 32; // map 0-255 to 0-31

        // Read direction register
        uint32_t reg_val = hololink_.read_uint32(register_address);

        // modify direction pin value
        if (direction == IN) {
            reg_val = set_bit(reg_val, pin_bit);
        } else if (direction == OUT) {
            reg_val = clear_bit(reg_val, pin_bit);
        } else {
            // raise exception
            throw std::runtime_error(fmt::format("GPIO:{},invalid direction:{}", pin, direction));
        }

        // write back modified value
        hololink_.write_uint32(register_address, reg_val);

        HSB_LOG_DEBUG("GPIO:{},set to direction:{}", pin, direction);
        return;
    }

    // raise exception
    throw std::runtime_error(fmt::format("GPIO:{},invalid pin", pin));
}

uint32_t Hololink::GPIO::get_direction(uint32_t pin)
{
    if (pin < gpio_pin_number_) {

        uint32_t register_address = GPIO_DIRECTION_BASE_REGISTER + ((pin / 32) * GPIO_REGISTER_ADDRESS_OFFSET);
        uint32_t pin_bit = pin % 32; // map 0-255 to 0-31

        uint32_t reg_val = hololink_.read_uint32(register_address);
        return read_bit(reg_val, pin_bit);
    }

    // raise exception
    throw std::runtime_error(fmt::format("GPIO:{},invalid pin", pin));
}

void Hololink::GPIO::set_value(uint32_t pin, uint32_t value)
{
    if (pin < gpio_pin_number_) {
        // make sure this is an output pin
        const uint32_t direction = get_direction(pin);

        uint32_t status_register_address = GPIO_STATUS_BASE_REGISTER + ((pin / 32) * GPIO_REGISTER_ADDRESS_OFFSET); // read from status
        uint32_t output_register_address = GPIO_OUTPUT_BASE_REGISTER + ((pin / 32) * GPIO_REGISTER_ADDRESS_OFFSET); // write to output
        uint32_t pin_bit = pin % 32; // map 0-255 to 0-31

        if (direction == OUT) {
            // Read output register values
            uint32_t reg_val = hololink_.read_uint32(status_register_address);

            // Modify pin in the register
            if (value == HIGH) {
                reg_val = set_bit(reg_val, pin_bit);
            } else if (value == LOW) {
                reg_val = clear_bit(reg_val, pin_bit);
            } else {
                // raise exception
                throw std::runtime_error(fmt::format("GPIO:{},invalid value:{}", pin, value));
            }

            // write back modified value
            hololink_.write_uint32(output_register_address, reg_val);

            HSB_LOG_DEBUG("GPIO:{},set to value:{}", pin, value);
            return;
        } else {
            // raise exception
            throw std::runtime_error(
                fmt::format("GPIO:{},trying to write to an input register!", pin));
        }
    }

    // raise exception
    throw std::runtime_error(fmt::format("GPIO:{},invalid pin", pin));
}

uint32_t Hololink::GPIO::get_value(uint32_t pin)
{
    if (pin < gpio_pin_number_) {

        uint32_t register_address = GPIO_STATUS_BASE_REGISTER + ((pin / 32) * GPIO_REGISTER_ADDRESS_OFFSET);
        uint32_t pin_bit = pin % 32; // map 0-255 to 0-31

        const uint32_t reg_val = hololink_.read_uint32(register_address);
        return read_bit(reg_val, pin_bit);
    }
    // raise exception
    throw std::runtime_error(fmt::format("GPIO:{},invalid pin", pin));
}

uint32_t Hololink::GPIO::get_supported_pin_num(void)
{
    return gpio_pin_number_;
}

/*static*/ uint32_t Hololink::GPIO::set_bit(uint32_t value, uint32_t bit)
{
    return value | (1 << bit);
}
/*static*/ uint32_t Hololink::GPIO::clear_bit(uint32_t value, uint32_t bit)
{
    return value & ~(1 << bit);
}
/*static*/ uint32_t Hololink::GPIO::read_bit(uint32_t value, uint32_t bit)
{
    return (value >> bit) & 0x1;
}

/** Constructs a lock using the shm_open() call to access a named
 * semaphore with the given name.
 */
Hololink::NamedLock::NamedLock(Hololink& hololink, std::string name)
    : fd_(-1)
{
    // We use lockf on this file as our interprocess locking
    // mechanism; that way if this program exits unexpectedly
    // we don't leave the lock held.  (An earlier implementation
    // using shm_open didn't guarantee releasing the lock if we exited due
    // to the user pressing control/C.)
    std::string formatted_name = hololink.device_specific_filename(name);
    int permissions = 0666; // make sure other processes can write
    fd_ = open(formatted_name.c_str(), O_WRONLY | O_CREAT, permissions);
    if (fd_ < 0) {
        throw std::runtime_error(
            fmt::format("open({}, ...) failed with errno={}: \"{}\"", formatted_name, errno, strerror(errno)));
    }
}

Hololink::NamedLock::~NamedLock() noexcept(false)
{
    int r = close(fd_);
    if (r != 0) {
        throw std::runtime_error(
            fmt::format("close failed with errno={}: \"{}\"", errno, strerror(errno)));
    }
}

void Hololink::NamedLock::lock()
{
    // Block until we're the owner.
    int r = lockf(fd_, F_LOCK, 0);
    if (r != 0) {
        throw std::runtime_error(
            fmt::format("lockf failed with errno={}: \"{}\"", errno, strerror(errno)));
    }
}

void Hololink::NamedLock::unlock()
{
    // Let another process take ownership.
    int r = lockf(fd_, F_ULOCK, 0);
    if (r != 0) {
        throw std::runtime_error(
            fmt::format("lockf failed with errno={}: \"{}\"", errno, strerror(errno)));
    }
}

Hololink::NamedLock& Hololink::i2c_lock()
{
    static NamedLock lock(this[0], "hololink-i2c-lock");
    return lock;
}

Hololink::NamedLock& Hololink::spi_lock()
{
    static NamedLock lock(this[0], "hololink-spi-lock");
    return lock;
}

Hololink::NamedLock& Hololink::lock()
{
    static NamedLock lock(this[0], "hololink-lock");
    return lock;
}

void Hololink::on_reset(std::shared_ptr<Hololink::ResetController> reset_controller)
{
    reset_controllers_.push_back(reset_controller);
}

std::string Hololink::device_specific_filename(std::string name)
{
    // Create a directory, if necessary, with our serial number.
    auto path = std::filesystem::temp_directory_path();
    path.append("hololink");
    path.append(serial_number_);
    if (!std::filesystem::exists(path)) {
        if (!std::filesystem::create_directories(path)) {
            throw std::runtime_error(
                fmt::format("create_directory({}) failed with errno={}: \"{}\"", std::string(path), errno, strerror(errno)));
        }
    }
    path.append(name);
    return std::string(path);
}

Hololink::ResetController::~ResetController()
{
}

bool Hololink::ptp_synchronize(const std::shared_ptr<Timeout>& timeout)
{
    constexpr uint32_t SYNCHRONIZED = 0xF;
    while (true) {
        auto value = read_uint32(FPGA_PTP_SYNC_STAT);
        if ((value & SYNCHRONIZED) == SYNCHRONIZED) {
            break;
        }
        if (timeout->expired()) {
            HSB_LOG_ERROR("ptp_synchronize timed out; value={:#X}.", value);
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    // Time is sync'd now.
    return true;
}

std::shared_ptr<Synchronizer> Hololink::ptp_pps_output(unsigned frequency)
{
    ptp_pps_output_->set_frequency(frequency);
    return ptp_pps_output_;
}

void Synchronizer::attach(std::shared_ptr<Synchronizable> peer)
{
    bool was_empty = false;
    { // lock scoping
        std::lock_guard<std::mutex> lock(peers_mutex_);
        // Don't attach more than once.
        if (std::find(peers_.cbegin(), peers_.cend(), peer) != peers_.cend()) {
            throw std::runtime_error("Synchronizer::attach called on the same Synchronizable more than once.");
        }
        was_empty = peers_.empty();
        peers_.push_back(peer);
    }
    // Call setup when the first peer is attached
    if (was_empty) {
        setup();
    }
}

void Synchronizer::detach(std::shared_ptr<Synchronizable> peer)
{
    bool now_empty = false;
    { // lock scoping
        std::lock_guard<std::mutex> lock(peers_mutex_);
        auto it = std::find(peers_.begin(), peers_.end(), peer);
        if (it != peers_.end()) {
            peers_.erase(it);
            now_empty = peers_.empty();
        }
    }
    // Call shutdown when the last peer is detached
    if (now_empty) {
        shutdown();
    }
}

class NullSynchronizer : public Synchronizer {
public:
    NullSynchronizer()
    {
    }

    void setup() override
    {
        /* ignored */
    }

    void shutdown() override
    {
        /* ignored */
    }

    bool is_enabled() override
    {
        return false;
    }
};

/* static */ std::shared_ptr<Synchronizer> Synchronizer::null_synchronizer()
{
    static auto null_synchronizer = std::make_shared<NullSynchronizer>();
    return null_synchronizer;
}

Hololink::FrameMetadata Hololink::deserialize_metadata(const uint8_t* metadata_buffer, unsigned metadata_buffer_size)
{
    hololink::core::Deserializer deserializer(metadata_buffer, metadata_buffer_size);
    FrameMetadata r = {}; // fill with 0s
    uint16_t ignored = 0;
    if (!(deserializer.next_uint32_be(r.flags)
            && deserializer.next_uint32_be(r.psn)
            && deserializer.next_uint32_be(r.crc)
            && deserializer.next_uint64_be(r.timestamp_s)
            && deserializer.next_uint32_be(r.timestamp_ns)
            && deserializer.next_uint64_be(r.bytes_written)
            && deserializer.next_uint16_be(ignored)
            && deserializer.next_uint16_be(r.frame_number)
            && deserializer.next_uint64_be(r.metadata_s)
            && deserializer.next_uint32_be(r.metadata_ns))) {
        throw std::runtime_error(fmt::format("Buffer underflow in metadata"));
    }
    HSB_LOG_TRACE("flags={:#x} psn={:#x} crc={:#x} timestamp_s={:#x} timestamp_ns={:#x} bytes_written={:#x} frame_number={:#x}",
        r.flags, r.psn, r.crc, r.timestamp_s, r.timestamp_ns, r.bytes_written, r.frame_number);
    return r;
}

bool Hololink::and_uint32(uint32_t address, uint32_t mask)
{
    std::lock_guard lock(this->lock());
    uint32_t value = read_uint32(address);
    value &= mask;
    return write_uint32(address, value);
}

bool Hololink::or_uint32(uint32_t address, uint32_t mask)
{
    std::lock_guard lock(this->lock());
    uint32_t value = read_uint32(address);
    value |= mask;
    return write_uint32(address, value);
}

class NullSequencer
    : public Hololink::Sequencer {
public:
    NullSequencer(Hololink& hololink)
        : Hololink::Sequencer()
        , hololink_(hololink)
    {
    }

    void enable() override
    {
        done();
        // NOTE @see Sequencer::assign_location has knowledge
        // of what's going on here.
        address_ = APB_RAM + 0x80;
        address_set_ = true;
        write(hololink_);
    }

public:
    Hololink& hololink_;
};

void Hololink::configure_hsb()
{
    // ARP packets are slow, so allow for more timeout on this initial read.
    auto get_hsb_ip_version_timeout = std::make_shared<Timeout>(30.f, 0.2f);

    // Because we're at the start of our session with HSB, let's reset it to
    // use the sequence number that we have from our constructor.  Following
    // this, unless the user specifies otherwise, we'll always check the
    // sequence number on every transaction-- which will trigger a fault if
    // another program goes in and does any sort of control-plane transaction.
    // Note that when a control plane request triggers a fault, the actual
    // command is ignored.
    bool sequence_check = false;
    hsb_ip_version_ = get_hsb_ip_version(get_hsb_ip_version_timeout, sequence_check);
    datecode_ = get_fpga_date();
    HSB_LOG_INFO("HSB IP version={:#x} datecode={:#x}", hsb_ip_version_, datecode_);

    // Disable VSYNC
    if (vsync_enable_) {
        write_uint32(VSYNC_CONTROL, 0);
    }

    // Enable PTP.  This feature may not be instantiated
    // within the FPGA, which would lead to an exception
    // in the first write_uint32-- which we can ignore.
    if (ptp_enable_) {
        write_uint32(FPGA_PTP_CTRL_DPLL_CFG1, 0x2);
        write_uint32(FPGA_PTP_CTRL_DPLL_CFG2, 0x2);
        write_uint32(FPGA_PTP_CTRL_DELAY_AVG_FACTOR, 0x3);
        write_uint32(FPGA_PTP_DELAY_ASYMMETRY, 0x33);
        write_uint32(FPGA_PTP_CTRL, 0x3); // dpll en
    }

    // PLACATE device programming; if we're an older FPGA--which only
    // happens during programming-- then we'll get errors on these.  Programming
    // tools know how to load up an older I2C driver so none of this is
    // used in that case.
    if (hsb_ip_version_ >= 0x2503) {
        // Clear any current events
        control_event_apb_interrupt_enable_cache_ = 0;
        write_uint32(CTRL_EVT_APB_INTERRUPT_EN, control_event_apb_interrupt_enable_cache_);
        control_event_rising_cache_ = 0;
        write_uint32(CTRL_EVT_RISING, control_event_rising_cache_);
        control_event_falling_cache_ = 0;
        write_uint32(CTRL_EVT_FALLING, control_event_falling_cache_);
        write_uint32(CTRL_EVT_CLEAR, 0xFFFF'FFFF);
        write_uint32(CTRL_EVT_CLEAR, 0x0000'0000);
        write_uint32(CTRL_EVT_APB_TIMEOUT, 0x3937); // 750us (Timeout/51.2ns=reg)
        write_uint32(CTRL_EVT_SW_EVENT, 0);
        // Set up the message destination address--
        // use local broadcast for this.
        std::vector<uint8_t> local_ip = { 255, 255, 255, 255 };
        std::vector<uint8_t> local_mac = { 255, 255, 255, 255, 255, 255 };
        uint32_t mac_addr_lo = ((local_mac[2] << 24)
            | (local_mac[3] << 16)
            | (local_mac[4] << 8)
            | (local_mac[5] << 0));
        uint32_t mac_addr_hi = (local_mac[0] << 8) | (local_mac[1] << 0);
        write_uint32(CTRL_EVT_HOST_MAC_ADDR_LO, mac_addr_lo);
        write_uint32(CTRL_EVT_HOST_MAC_ADDR_HI, mac_addr_hi);
        uint32_t host_ip = ((local_ip[0] << 24)
            | (local_ip[1] << 16)
            | (local_ip[2] << 8)
            | (local_ip[3] << 0));
        write_uint32(CTRL_EVT_HOST_IP_ADDR, host_ip);
        write_uint32(CTRL_EVT_HOST_UDP_PORT, async_event_socket_port_);
        write_uint32(CTRL_EVT_FPGA_UDP_PORT, 8432);

        // Workaround for HSB100G running FW versions dated earlier than
        // 06/04/2025 which do not have valid APG_RAM + 0x80 address for
        // sequencing
        if (!skip_sequence_initialization_) {
            // Configure a NULL sequence, which is a HSB IP sequence
            // that doesn't do anything.  This provides a reasonable
            // vector for unused events.
            NullSequencer null_sequencer(this[0]);
            null_sequencer.enable();
            null_sequence_location_ = null_sequencer.location();
        }
    }
}

void Hololink::configure_apb_event(Event event, uint32_t handler, bool rising_edge)
{
    clear_apb_event(event);

    // Set the pointer
    if (handler == 0) {
        write_uint32(APB_RAM + event * 4, null_sequence_location_);
    } else {
        write_uint32(APB_RAM + event * 4, handler);
    }
    // Trigger on the event
    uint32_t mask = 1 << event;
    if (rising_edge) {
        control_event_rising_cache_ |= mask;
        write_uint32(CTRL_EVT_RISING, control_event_rising_cache_);
    } else {
        control_event_falling_cache_ |= mask;
        write_uint32(CTRL_EVT_FALLING, control_event_falling_cache_);
    }
    control_event_apb_interrupt_enable_cache_ |= mask;
    write_uint32(CTRL_EVT_APB_INTERRUPT_EN, control_event_apb_interrupt_enable_cache_);
}

void Hololink::clear_apb_event(Hololink::Event event)
{
    // Quiesce the event
    uint32_t mask = 1 << event;
    if (control_event_apb_interrupt_enable_cache_ & mask) {
        control_event_apb_interrupt_enable_cache_ &= ~mask;
        write_uint32(CTRL_EVT_APB_INTERRUPT_EN, control_event_apb_interrupt_enable_cache_);
    }
    if (control_event_rising_cache_ & mask) {
        control_event_rising_cache_ &= ~mask;
        write_uint32(CTRL_EVT_RISING, control_event_rising_cache_);
    }
    if (control_event_falling_cache_ & mask) {
        control_event_falling_cache_ &= ~mask;
        write_uint32(CTRL_EVT_FALLING, control_event_falling_cache_);
    }
    write_uint32(CTRL_EVT_CLEAR, 1 << event);
    write_uint32(CTRL_EVT_CLEAR, 0x0000'0000);
}

void Hololink::async_event_thread()
{
    HSB_LOG_TRACE("async_event_thread starting.");
    int async_event_socket = async_event_socket_.get();
    while (true) {
        std::vector<uint8_t> received(hololink::core::UDP_PACKET_SIZE);
        sockaddr_in peer_address {};
        peer_address.sin_family = AF_UNSPEC;
        socklen_t peer_address_len = sizeof(peer_address);
        ssize_t received_bytes;
        while (true) {
            received_bytes = recvfrom(async_event_socket, received.data(), received.size(), 0,
                (sockaddr*)&peer_address, &peer_address_len);
            if (received_bytes > 0) {
                break;
            }
            if (received_bytes == 0) {
                // Socket was closed.
                HSB_LOG_DEBUG("async_event_thread shutdown.");
                return;
            }
            if (received_bytes != -1) {
                // This is outside specification for recvfrom
                throw std::runtime_error(fmt::format("async_event_thread recvfrom received_bytes={} errno={}", received_bytes, errno));
            }
            // In this case, errno is set.
            if (errno == EAGAIN) {
                continue;
            }
            if (errno == EWOULDBLOCK) {
                continue;
            }
            HSB_LOG_ERROR("async_event_thread errno={}.", errno);
            throw std::runtime_error(fmt::format("async_event_thread recvfrom errno={}", errno));
        }

        received.resize(received_bytes);
        char peer_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &peer_address.sin_addr, peer_ip, sizeof(peer_ip));
        //
        hololink::core::Deserializer deserializer(received.data(), received.size());
        uint32_t interrupt_active = 0;
        uint32_t interrupt_state = 0;
        uint64_t timestamp_s = 0;
        uint32_t timestamp_ns = 0;
        if (!(deserializer.next_uint32_be(interrupt_active)
                && deserializer.next_uint32_be(interrupt_state)
                && deserializer.next_uint48_be(timestamp_s)
                && deserializer.next_uint32_be(timestamp_ns))) {
            throw std::runtime_error(fmt::format("Buffer underflow in async event message"));
        }
        HSB_LOG_DEBUG("async_event_thread received {} bytes from={} interrupt_active={:#x} interrupt_state={:#x} timestamp_s={}, timestamp_ns={}.", received.size(), peer_ip, interrupt_active, interrupt_state, timestamp_s, timestamp_ns);
    }
}

/**
 * Support for e.g. I2C transactions synchronized with
 * frame-end.
 */
Hololink::Sequencer::Sequencer(unsigned limit)
    : buffer_()
    , done_(false)
    , command_index_(0)
    , command_bit_(0)
    , address_(0)
    , address_set_(false)
    , written_(false)
{
    buffer_.reserve(limit);
    // Start with a 0 in the first command register.
    command_index_ = append(0);
}

// ABCs require virtual constructors.
Hololink::Sequencer::~Sequencer()
{
}

void Hololink::Sequencer::done()
{
    /*
     * Note that this RD command is ignored because
     * the FFFF_FFFF done flag overrides it... but
     * we do need to have a command word in the corner
     * case where our sequence to this point ended
     * exactly on a command-word boundary
     */
    add_command(RD);
    append(0xFFFF'FFFF);
    done_ = true;
}

unsigned Hololink::Sequencer::write_uint32(uint32_t address, uint32_t data)
{
    add_command(WR);
    append(address);
    unsigned r = append(data);
    return r;
}

unsigned Hololink::Sequencer::read_uint32(uint32_t address, uint32_t initial_value)
{
    add_command(RD);
    append(address);
    unsigned r = append(initial_value);
    return r;
}

unsigned Hololink::Sequencer::poll(uint32_t address, uint32_t mask, uint32_t match)
{
    add_command(POLL);
    append(address);
    // yes, order is different than the parameter list
    unsigned r = append(match);
    append(mask);
    return r;
}

void Hololink::Sequencer::assign_location(Hololink::Event event)
{
    if (!done_) {
        throw std::runtime_error(fmt::format("Call done() on a sequencer before calling assign_location()."));
    }
    if (address_set_) {
        throw std::runtime_error(fmt::format("Sequencer address is already set to {:#x}", address_));
    }
    uint32_t address = 0;
    switch (event) {
        // NOTE @see NullSequencer which has a hard-coded address of 0x80.

    case Hololink::Event::SIF_0_FRAME_END:
        address = APB_RAM + 0x100;
        break;
    case Hololink::Event::SIF_1_FRAME_END:
        address = APB_RAM + 0x200;
        break;
    case Hololink::Event::SW_EVENT:
        address = APB_RAM + 0x800;
        break;
    default:
        throw std::runtime_error(fmt::format("Sequencer assign_location with unexpected event={}", static_cast<int>(event)));
    }
    address_ = address;
    address_set_ = true;
}

void Hololink::Sequencer::write(Hololink& hololink)
{
    if (!address_set_) {
        throw std::runtime_error(fmt::format("Sequencer address not set; call assign_location() before calling write()."));
    }
    unsigned address = address_;
    for (const auto& item : buffer_) {
        hololink.write_uint32(address, item);
        address += 4;
    }
    written_ = true;
}

unsigned Hololink::Sequencer::append(uint32_t content)
{
    if (done_) {
        throw std::runtime_error("Cannot append more data to a sequence that was marked as done.");
    }
    unsigned r = buffer_.size();
    buffer_.push_back(content);
    return r;
}

unsigned Hololink::Sequencer::add_command(Hololink::Sequencer::Op op)
{
    if (done_) {
        throw std::runtime_error("Cannot append commands to a sequence that was marked as done.");
    }
    // each op is 2 bits
    uint32_t content = static_cast<uint32_t>(op);
    if ((op & ~3) != 0) {
        throw std::runtime_error(fmt::format("Invalid op value {:#x}.", content));
    }
    // are we in an invalid state?
    if (command_bit_ > 32) {
        throw std::runtime_error(fmt::format("Invalid command_bit_={:#x}.", command_bit_));
    }
    //
    if (command_bit_ == 32) {
        command_index_ = append(0);
        command_bit_ = 0;
    }
    //
    buffer_[command_index_] |= (op << command_bit_);
    command_bit_ += 2;
    return command_index_;
}

uint32_t Hololink::Sequencer::location()
{
    if (!written_) {
        throw std::runtime_error("Location isn't valid until a sequence is written; see write().");
    }
    // This next state should be impossible-- you have to have address_set_ in
    // order to set written_ -- but let's be defensive anyway.
    if (!address_set_) {
        throw std::runtime_error(fmt::format("Sequencer address must be set; see assign_location()."));
    }
    return address_;
}

Synchronizable::Synchronizable() = default;

Hololink::PtpSynchronizer::PtpSynchronizer(Hololink& hololink)
    : hololink_(hololink)
    , frequency_(0)
    , frequency_control_(0)
{
}

void Hololink::PtpSynchronizer::set_frequency(unsigned frequency)
{
    // The first time we're set up, we require they provide
    // a frequency value.
    if (frequency_ == 0) {
        // VSYNC Frequency: 0=10Hz, 1=30Hz, 2=60Hz, 3=90Hz, 4=120Hz
        switch (frequency) {
        case 10:
            frequency_control_ = 0;
            break;
        case 30:
            frequency_control_ = 1;
            break;
        case 60:
            frequency_control_ = 2;
            break;
        case 90:
            frequency_control_ = 3;
            break;
        case 120:
            frequency_control_ = 4;
            break;
        default:
            throw std::runtime_error(
                fmt::format("Invalid PtpSynchronizer frequency={}.", frequency));
        }
        frequency_ = frequency;
        return;
    }

    if (frequency == 0) {
        return;
    }

    if (frequency != frequency_) {
        throw std::runtime_error(
            fmt::format("PtpSynchronizer frequency is already={}, can't change it to {}.", frequency_, frequency));
    }
}

void Hololink::PtpSynchronizer::setup()
{
    HSB_LOG_INFO("PtpSynchronizer: Setting up PTP synchronization with frequency {} Hz", frequency_);

    // Configure FPGA registers for VSYNC timing
    hololink_.write_uint32(VSYNC_CONTROL, 0); // Disable VSYNC
    hololink_.write_uint32(VSYNC_FREQUENCY, frequency_control_); // Set frequency control
    hololink_.write_uint32(VSYNC_START, 0); // VSYNC Start Value (For Active Low, use 0x1)
    hololink_.write_uint32(VSYNC_DELAY, 0); // VSYNC delay (ns)
    hololink_.write_uint32(VSYNC_EXPOSURE, 0xF4240); // Exposure time (1ms)
    hololink_.write_uint32(VSYNC_CONTROL, 1); // Enable VSYNC
    hololink_.write_uint32(VSYNC_GPIO, 0xF); // Set GPIO as OUT
}

void Hololink::PtpSynchronizer::shutdown()
{
    HSB_LOG_INFO("PtpSynchronizer: Shutting down PTP synchronization");

    // Disable VSYNC
    hololink_.write_uint32(VSYNC_CONTROL, 0);
}

bool Hololink::PtpSynchronizer::is_enabled()
{
    return true;
}

} // namespace hololink
