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
#include <netinet/in.h>
#include <sys/socket.h>

#include <cassert>
#include <cstdint>
#include <thread>

#include <hololink/native/deserializer.hpp>
#include <hololink/native/serializer.hpp>
#include <holoscan/logger/logger.hpp>

#include "metadata.hpp"

namespace hololink {

namespace {

    static std::map<std::string, std::shared_ptr<Hololink>> hololink_by_serial_number;

    // SPI control flags
    constexpr uint32_t SPI_START = 0b0000'0000'0000'0001;
    constexpr uint32_t SPI_BUSY = 0b0000'0001'0000'0000;
    // SPI_CFG
    constexpr uint32_t SPI_CFG_CPOL = 0b0000'0000'0001'0000;
    constexpr uint32_t SPI_CFG_CPHA = 0b0000'0000'0010'0000;

    // GPIO Registers
    // bitmask 0:F, ecah bit correspondes to a GPIO pin
    // GPIO_OUTPUT_REGISTER    - W   - set output pin values
    // GPIO_DIRECTION_REGISTER - R/W - set/read GPIO pin direction
    // GPIO_STATUS_REGISTER    - R   - read input GPIO value
    constexpr uint32_t GPIO_OUTPUT_REGISTER = 0x0000'000C;
    constexpr uint32_t GPIO_DIRECTION_REGISTER = 0x0000'002C;
    constexpr uint32_t GPIO_STATUS_REGISTER = 0x0000'008C;

} // anonymous namespace

Hololink::Hololink(
    const std::string& peer_ip, uint32_t control_port, const std::string& serial_number)
    : peer_ip_(peer_ip)
    , control_port_(control_port)
    , serial_number_(serial_number)
{
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

        r = std::make_shared<Hololink>(
            peer_ip.value(), control_port.value(), serial_number.value());

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
        HOLOSCAN_LOG_INFO("Removing hololink \"{}\"", it->first);
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

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> Hololink::csi_size()
{
    // CSI-2 spec 9.1
    const uint32_t frame_start_size = 4;
    const uint32_t frame_end_size = 4;
    const uint32_t line_start_size = 4;
    const uint32_t line_end_size = 2;
    return { frame_start_size, frame_end_size, line_start_size, line_end_size };
}

void Hololink::start()
{
    control_socket_.reset(socket(AF_INET, SOCK_DGRAM, 0));
    if (!control_socket_) {
        throw std::runtime_error("Failed to create socket");
    }
    // ARP packets are slow, so allow for more timeout on this initial read.
    auto get_fpga_version_timeout = std::make_shared<Timeout>(30.f, 0.2f);
    version_ = get_fpga_version(get_fpga_version_timeout);
    datecode_ = get_fpga_date();
    HOLOSCAN_LOG_INFO("FPGA version={:#x} datecode={:#x}", version_, datecode_);
}

void Hololink::stop() { control_socket_.reset(); }

void Hololink::reset()
{
    std::shared_ptr<Spi> spi = get_spi(CLNX_SPI_CTRL, /*chip_select*/ 0, /*clock_divisor*/ 15,
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
        HOLOSCAN_LOG_INFO("ignoring error {}.", e.what());
    }

    // Now wait for the device to come back up.
    // This guy raises an exception if we're not found;
    // this can happen if set-ip is used in one-time
    // mode.
    Enumerator::find_channel(peer_ip_, std::make_shared<Timeout>(30.f));

    // ARP packets are slow, so allow for more timeout on this initial read.
    auto get_fpga_version_timeout = std::make_shared<Timeout>(30.f, 0.2f);
    uint32_t version = get_fpga_version(get_fpga_version_timeout);
    HOLOSCAN_LOG_INFO("version={:#x}", version);
}

uint32_t Hololink::get_fpga_version(const std::shared_ptr<Timeout>& timeout)
{
    const uint32_t version = read_uint32(FPGA_VERSION, timeout);
    return version;
}

uint32_t Hololink::get_fpga_date()
{
    const uint32_t date = read_uint32(FPGA_DATE);
    return date;
}

bool Hololink::write_uint32(
    uint32_t address, uint32_t value, const std::shared_ptr<Timeout>& in_timeout, bool retry)
{
    uint32_t count = 0;
    std::exception_ptr eptr;
    std::shared_ptr<Timeout> timeout = Timeout::default_timeout(in_timeout);
    try {
        while (true) {
            count += 1;
            bool status = write_uint32_(address, value, timeout, retry /*response_expected*/);
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
    const std::shared_ptr<Timeout>& timeout = std::shared_ptr<Timeout>(), bool response_expected)
{
    HOLOSCAN_LOG_DEBUG("write_uint32(address={:#x}, value={:#x})", address, value);
    if ((address & 3) != 0) {
        throw std::runtime_error(
            fmt::format("Invalid address \"{:#x}\", has to be a multiple of four", address));
    }
    // BLOCKING on ack or timeout
    // This routine serializes a write_uint32 request
    // and forwards it to the device.
    const uint16_t sequence = next_sequence();
    // Serialize
    std::vector<uint8_t> request(20);
    native::Serializer serializer(request.data(), request.size());
    if (!(serializer.append_uint8(WR_DWORD) && serializer.append_uint8(REQUEST_FLAGS_ACK_REQUEST)
            && serializer.append_uint16_be(sequence) && serializer.append_uint8(0) // reserved
            && serializer.append_uint8(0) // reserved
            && serializer.append_uint32_be(address) && serializer.append_uint32_be(value))) {
        throw std::runtime_error("Unable to serialize");
    }
    request.resize(serializer.length());

    std::vector<uint8_t> reply(20);
    auto [status, response_code, deserializer] = execute(sequence, request, reply, timeout);
    if (!status) {
        // timed out
        return false;
    }
    if (response_code != RESPONSE_SUCCESS) {
        if (!response_code.has_value()) {
            if (response_expected) {
                HOLOSCAN_LOG_ERROR(
                    "write_uint32 address={:#X} value={:#X} response_code=None", address, value);
                return false;
            }
        }
        throw std::runtime_error(
            fmt::format("write_uint32 address={:#X} value={:#X} response_code={:#X}", address,
                value, response_code.value()));
    }
    return true;
}

uint32_t Hololink::read_uint32(uint32_t address, const std::shared_ptr<Timeout>& in_timeout)
{
    uint32_t count = 0;
    std::exception_ptr eptr;
    std::shared_ptr<Timeout> timeout = Timeout::default_timeout(in_timeout);
    try {
        while (true) {
            count += 1;
            auto [status, value] = read_uint32_(address, timeout);
            if (status) {
                return value.value();
            }
            if (!timeout->retry()) {
                throw TimeoutError(fmt::format("read_uint32 address={:#x}", address));
            }
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
    uint32_t address, const std::shared_ptr<Timeout>& timeout)
{
    HOLOSCAN_LOG_DEBUG("read_uint32(address={:#x})", address);
    if ((address & 3) != 0) {
        throw std::runtime_error(
            fmt::format("Invalid address \"{:#x}\", has to be a multiple of four", address));
    }
    // BLOCKING on ack or timeout
    // This routine serializes a read_uint32 request
    // and forwards it to the device.
    uint16_t sequence = next_sequence();
    // Serialize
    std::vector<uint8_t> request(20);
    native::Serializer serializer(request.data(), request.size());
    if (!(serializer.append_uint8(RD_DWORD) && serializer.append_uint8(REQUEST_FLAGS_ACK_REQUEST)
            && serializer.append_uint16_be(sequence) && serializer.append_uint8(0) // reserved
            && serializer.append_uint8(0) // reserved
            && serializer.append_uint32_be(address))) {
        throw std::runtime_error("Unable to serialize");
    }
    request.resize(serializer.length());
    HOLOSCAN_LOG_TRACE("read_uint32: {}....{}", request, sequence);

    std::vector<uint8_t> reply(20);
    auto [status, response_code, deserializer] = execute(sequence, request, reply, timeout);
    if (!status) {
        // timed out
        return { false, {} };
    }
    if (response_code != RESPONSE_SUCCESS) {
        throw std::runtime_error(
            fmt::format("read_uint32 response_code={}", response_code.value()));
    }
    uint8_t reserved;
    uint32_t response_address;
    uint32_t value;
    if (!(deserializer->next_uint8(reserved) /* reserved */
            && deserializer->next_uint32_be(response_address) /* address */
            && deserializer->next_uint32_be(value))) {
        throw std::runtime_error("Unable to deserialize");
    }
    assert(response_address == address);
    HOLOSCAN_LOG_DEBUG("read_uint32(address={:#x})={:#x}", address, value);
    return { true, value };
}

uint16_t Hololink::next_sequence()
{
    uint16_t r = sequence_;
    sequence_ = sequence_ + 1;
    return r;
}

std::tuple<bool, std::optional<uint32_t>, std::shared_ptr<native::Deserializer>> Hololink::execute(
    uint16_t sequence, const std::vector<uint8_t>& request, std::vector<uint8_t>& reply,
    const std::shared_ptr<Timeout>& timeout)
{
    HOLOSCAN_LOG_TRACE("Sending request={}", request);
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
        auto deserializer = std::make_shared<native::Deserializer>(reply.data(), reply.size());
        uint8_t dummy;
        uint16_t reply_sequence = 0;
        uint8_t response_code = 0;
        if (!(deserializer->next_uint8(dummy) /* reply_cmd_code */
                && deserializer->next_uint8(dummy) /* reply_flags */
                && deserializer->next_uint16_be(reply_sequence)
                && deserializer->next_uint8(response_code))) {
            throw std::runtime_error("Unable to deserialize");
        }
        HOLOSCAN_LOG_TRACE("reply reply_sequence={} response_code={} sequence={}", reply_sequence,
            response_code, sequence);
        if (sequence == reply_sequence) {
            return { true, response_code, deserializer };
        }
    }
}

void Hololink::send_control(const std::vector<uint8_t>& request)
{
    HOLOSCAN_LOG_TRACE(
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
            if (errno == EINTR) {
                // retry
                continue;
            }
            throw std::runtime_error(
                fmt::format("select failed with errno={}: \"{}\"", errno, strerror(errno)));
        }
        if (FD_ISSET(control_socket_.get(), &x)) {
            HOLOSCAN_LOG_ERROR("Error reading enumeration sockets.");
            return {};
        }
        if (result == 0) {
            // Timed out
            continue;
        }

        std::vector<uint8_t> received(8192);
        sockaddr_in peer_address {};
        peer_address.sin_family = AF_UNSPEC;
        socklen_t peer_address_len = sizeof(peer_address);
        ssize_t received_bytes;
        do {
            received_bytes = recvfrom(control_socket_.get(), received.data(), received.size(), 0,
                (sockaddr*)&peer_address, &peer_address_len);
            if ((received_bytes == -1) && (errno != EINTR)) {
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
    HOLOSCAN_LOG_TRACE("Got reply={}", reply);
}

void Hololink::add_read_retries(uint32_t n) { }

void Hololink::add_write_retries(uint32_t n) { }

void Hololink::write_renesas(I2c& i2c, const std::vector<uint8_t>& data)
{
    HOLOSCAN_LOG_TRACE("write_renesas data={}", data);
    uint32_t read_byte_count = 0;
    constexpr uint32_t RENESAS_I2C_ADDRESS = 0x09;
    std::vector<uint8_t> reply = i2c.i2c_transaction(RENESAS_I2C_ADDRESS, data, read_byte_count);
    HOLOSCAN_LOG_TRACE("reply={}.", reply);
}

void Hololink::setup_clock(const std::vector<std::vector<uint8_t>>& clock_profile)
{
    // set the clock driver.
    std::shared_ptr<Hololink::I2c> i2c = get_i2c(BL_I2C_CTRL);
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

std::shared_ptr<Hololink::I2c> Hololink::get_i2c(uint32_t i2c_address)
{
    auto r = std::make_shared<Hololink::I2c>(*this, i2c_address);
    return r;
}

std::shared_ptr<Hololink::Spi> Hololink::get_spi(uint32_t spi_address, uint32_t chip_select,
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
    uint32_t spi_cfg = clock_divisor | (chip_select << 12) | width_map[width];
    if (cpol) {
        spi_cfg |= SPI_CFG_CPOL;
    }
    if (cpha) {
        spi_cfg |= SPI_CFG_CPHA;
    }
    return std::make_shared<Hololink::Spi>(*this, spi_address, spi_cfg);
}

std::shared_ptr<Hololink::GPIO> Hololink::get_gpio()
{
    auto r = std::make_shared<Hololink::GPIO>(*this);
    return r;
}

Hololink::I2c::I2c(Hololink& hololink, uint32_t i2c_address)
    : hololink_(hololink)
    , reg_control_(i2c_address + 0)
    , reg_num_bytes_(i2c_address + 4)
    , reg_clk_ctrl_(i2c_address + 8)
    , reg_data_buffer_(i2c_address + 16)
{
}

bool Hololink::I2c::set_i2c_clock()
{
    // set the clock to 400KHz (fastmode) i2c speed once at init
    const uint32_t clock = 0b0000'0101;
    return hololink_.write_uint32(reg_clk_ctrl_, clock, Timeout::i2c_timeout());
}

std::vector<uint8_t> Hololink::I2c::i2c_transaction(uint32_t peripheral_i2c_address,
    const std::vector<uint8_t>& write_bytes, uint32_t read_byte_count,
    const std::shared_ptr<Timeout>& in_timeout)
{
    HOLOSCAN_LOG_DEBUG("i2c_transaction peripheral={:#x} len(write_bytes)={} read_byte_count={}",
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
    std::shared_ptr<Timeout> timeout = Timeout::i2c_timeout(in_timeout);
    // Hololink FPGA doesn't support resetting the I2C interface;
    // so the best we can do is make sure it's not busy.
    uint32_t value = hololink_.read_uint32(reg_control_, timeout);
    assert((value & I2C_BUSY) == 0);
    //
    // set the device address and enable the i2c controller
    // I2C_DONE_CLEAR -> 1
    uint32_t control = (peripheral_i2c_address << 16) | I2C_CORE_EN | I2C_DONE_CLEAR;
    hololink_.write_uint32(reg_control_, control, timeout);
    // I2C_DONE_CLEAR -> 0
    control = (peripheral_i2c_address << 16) | I2C_CORE_EN;
    hololink_.write_uint32(reg_control_, control, timeout);
    // make sure DONE is 0.
    value = hololink_.read_uint32(reg_control_, timeout);
    HOLOSCAN_LOG_DEBUG("control value={:#x}", value);
    assert((value & I2C_DONE) == 0);
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
        control = (peripheral_i2c_address << 16) | I2C_CORE_EN | I2C_START;
        hololink_.write_uint32(reg_control_, control, timeout);
        // retry if we don't see BUSY or DONE
        value = hololink_.read_uint32(reg_control_, timeout);
        if (value & (I2C_DONE | I2C_BUSY)) {
            break;
        }
        if (!timeout->retry()) {
            // timed out
            HOLOSCAN_LOG_DEBUG("Timed out.");
            throw TimeoutError(
                fmt::format("i2c_transaction i2c_address={:#x}", peripheral_i2c_address));
        }
    }
    // Poll until done.  Future version will have an event packet too.
    while (true) {
        value = hololink_.read_uint32(reg_control_, timeout);
        HOLOSCAN_LOG_TRACE("control={:#x}.", value);
        const uint32_t done = value & I2C_DONE;
        if (done != 0) {
            break;
        }
        if (!timeout->retry()) {
            // timed out
            HOLOSCAN_LOG_DEBUG("Timed out.");
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

Hololink::Spi::Spi(Hololink& hololink, uint32_t address, uint32_t spi_cfg)
    : hololink_(hololink)
    , reg_control_(address + 0)
    , reg_num_bytes_(address + 4)
    , reg_spi_cfg_(address + 8)
    , reg_num_bytes2_(address + 12)
    , reg_data_buffer_(address + 16)
    , spi_cfg_(spi_cfg)
    , turnaround_cycles_(0)
{
}

std::vector<uint8_t> Hololink::Spi::spi_transaction(const std::vector<uint8_t>& write_command_bytes,
    const std::vector<uint8_t>& write_data_bytes, uint32_t read_byte_count,
    const std::shared_ptr<Timeout>& in_timeout)
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
    // Because the controller always records ingress data,
    // whether we're transmitting or receiving, we get a copy
    // of the written data in the buffer on completion--
    // which means the buffer has to have enough space for
    // both the egress and ingress data.
    const uint32_t buffer_count = write_byte_count + read_byte_count;
    if (buffer_count >= buffer_size) {
        throw std::runtime_error(fmt::format("Size of combined write and read size is too large: "
                                             "\"{:#x}\", has to be less than {:#x}",
            buffer_count, buffer_size));
    }
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
            HOLOSCAN_LOG_DEBUG("Timed out.");
            throw TimeoutError(fmt::format("spi_transaction control={:#x}", reg_control_));
        }
    }
    // round up to get the whole next word
    std::vector<uint8_t> r(buffer_count + 3);
    // no need to re-read the transmitted data
    uint32_t start_byte_offset = write_byte_count;
    // but we can only read words; so back up to the word boundary
    start_byte_offset &= ~3;
    for (uint32_t i = start_byte_offset; i < buffer_count; i += 4) {
        value = hololink_.read_uint32(reg_data_buffer_ + i, timeout);
        r[i + 0] = (value >> 0) & 0xFF;
        r[i + 1] = (value >> 8) & 0xFF;
        r[i + 2] = (value >> 16) & 0xFF;
        r[i + 3] = (value >> 24) & 0xFF;
    }
    // skip over the data that we wrote out.
    r = std::vector<uint8_t>(
        r.cbegin() + write_byte_count, r.cbegin() + write_byte_count + read_byte_count);
    return r;
}

Hololink::GPIO::GPIO(Hololink& hololink)
    : hololink_(hololink)
{
}

void Hololink::GPIO::set_direction(uint32_t pin, uint32_t direction)
{
    if (pin < GPIO_PIN_RANGE) {

        // Read direction register
        uint32_t reg_val = hololink_.read_uint32(GPIO_DIRECTION_REGISTER);

        // modify direction pin value
        if (direction == IN) {
            reg_val = set_bit(reg_val, pin);
        } else if (direction == OUT) {
            reg_val = clear_bit(reg_val, pin);
        } else {
            // raise exception
            throw std::runtime_error(fmt::format("GPIO:{},invalid direction:{}", pin, direction));
        }

        // write back modified value
        hololink_.write_uint32(GPIO_DIRECTION_REGISTER, reg_val);
        HOLOSCAN_LOG_INFO("GPIO:{},set to direction:{}", pin, direction);
        return;
    }

    // raise exception
    throw std::runtime_error(fmt::format("GPIO:{},invalid pin", pin));
}

uint32_t Hololink::GPIO::get_direction(uint32_t pin)
{
    if (pin < GPIO_PIN_RANGE) {
        uint32_t reg_val = hololink_.read_uint32(GPIO_DIRECTION_REGISTER);
        return read_bit(reg_val, pin);
    }

    // raise exception
    throw std::runtime_error(fmt::format("GPIO:{},invalid pin", pin));
}

void Hololink::GPIO::set_value(uint32_t pin, uint32_t value)
{
    if (pin < GPIO_PIN_RANGE) {
        // make sure this is an output pin
        const uint32_t direction = get_direction(pin);

        if (direction == OUT) {
            // Read output register values
            uint32_t reg_val = hololink_.read_uint32(GPIO_STATUS_REGISTER);

            // Modify pin in the register
            if (value == HIGH) {
                reg_val = set_bit(reg_val, pin);
            } else if (value == LOW) {
                reg_val = clear_bit(reg_val, pin);
            } else {
                // raise exception
                throw std::runtime_error(fmt::format("GPIO:{},invalid value:{}", pin, value));
            }

            // write back modified value
            hololink_.write_uint32(GPIO_OUTPUT_REGISTER, reg_val);
            HOLOSCAN_LOG_INFO("GPIO:{},set to value:{}", pin, value);
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
    if (pin < GPIO_PIN_RANGE) {
        const uint32_t reg_val = hololink_.read_uint32(GPIO_STATUS_REGISTER);
        return read_bit(reg_val, pin);
    }
    // raise exception
    throw std::runtime_error(fmt::format("GPIO:{},invalid pin", pin));
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

} // namespace hololink