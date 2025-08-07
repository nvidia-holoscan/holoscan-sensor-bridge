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
#include <csignal>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "hololink/core/deserializer.hpp"
#include "hololink/core/hololink.hpp"
#include "hololink/core/serializer.hpp"

#include "data_plane.hpp"
#include "hsb_emulator.hpp"
#include "mem_register.hpp"
#include "net.hpp"
#include "utils.hpp"

// this is wrapping POSIX poll() flags. We read on any type of read and stop on all errors/hangups
#define POLL_RD_ANY (POLLIN | POLLPRI)
#define POLL_STOP (POLLHUP | POLLERR | POLLNVAL)

// This is part of workaround for polling on I2C transactions.
#define POLL_COUNT_TRIGGER 10
#define POLL_COUNT_EXIT 30

// Constants for control plane message
#define CONTROL_MESSAGE_READ_SIZE 10u
#define CONTROL_MESSAGE_WRITE_SIZE 14u
#define MIN_VALID_CONTROL_LENGTH CONTROL_MESSAGE_READ_SIZE
// reply is a ControlMessage with a uint32_t latched sequence
#define REPLY_MESSAGE_SIZE (CONTROL_MESSAGE_WRITE_SIZE + sizeof(uint32_t))

namespace hololink::emulation {

struct ControlMessage {
    uint8_t cmd_code;
    uint8_t flags;
    uint16_t sequence;
    uint8_t status;
    uint8_t reserved;
    uint32_t address;
    uint32_t value; // only received in cmd_code == WR_DWORD message, returned in all cases
};

// returns true if the message is valid, false otherwise.
// NOTE: if false, the ControlMessage object is in an indeterminate state.
bool deserialize_control_message(hololink::core::Deserializer& deserializer, ControlMessage& message)
{
    return deserializer.next_uint8(message.cmd_code)
        && deserializer.next_uint8(message.flags)
        && deserializer.next_uint16_be(message.sequence) // other code expects
        && deserializer.next_uint8(message.status)
        && deserializer.next_uint8(message.reserved)
        && (message.cmd_code == hololink::WR_DWORD ? deserializer.next_uint32_be(message.address)
                    && deserializer.next_uint32_be(message.value)
                                                   : deserializer.next_uint32_be(message.address));
}

// returns 0 on failure or the number of bytes written on success
// Note that on failure, serializer and buffer contents are in indeterminate state.
size_t serialize_reply_message(hololink::core::Serializer& serializer, ControlMessage& message)
{
    return serializer.append_uint8(message.cmd_code)
            && serializer.append_uint8(message.flags)
            && serializer.append_uint16_be(message.sequence)
            && serializer.append_uint8(message.status)
            && serializer.append_uint8(message.reserved)
            && serializer.append_uint32_be(message.address)
            && serializer.append_uint32_be(message.value)
            && serializer.append_uint32_be(0) /* latched_sequence */
        ? serializer.length()
        : 0;
}

HSBEmulator::HSBEmulator()
{
    registers_ = std::make_shared<MemRegister>(&configuration_);
}

HSBEmulator::~HSBEmulator()
{
    // stop the HSBEmulator. stop() is idempotent so this is safe.
    stop();
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
                uint8_t message_buffer[sizeof(ControlMessage)];
                const ssize_t message_length = recvfrom(control_socket, message_buffer, sizeof(message_buffer), 0, (struct sockaddr*)&host_addr, &host_addr_len);

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
                    fprintf(stderr, "deserialize_control_message failed\n");
                    continue;
                }

                switch (message.cmd_code) {
                case hololink::RD_DWORD: {
                    if (message_length < CONTROL_MESSAGE_READ_SIZE) {
                        fprintf(stderr, "incomplete message received: %zu - %u\n", message_length, CONTROL_MESSAGE_READ_SIZE);
                        continue;
                    }
                    handle_read_message(message, control_socket, &host_addr, host_addr_len);
                    break;
                }
                case hololink::WR_DWORD: {
                    if (message_length < CONTROL_MESSAGE_WRITE_SIZE) {
                        fprintf(stderr, "incomplete message received: %zu - %u\n", message_length, CONTROL_MESSAGE_WRITE_SIZE);
                        continue;
                    }
                    handle_write_message(message, control_socket, &host_addr, host_addr_len);
                    break;
                }
                default: {
                    handle_invalid_message(message, control_socket, &host_addr, host_addr_len);
                    break;
                }
                }

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

void HSBEmulator::handle_spi_control_write(uint32_t address, uint32_t value)
{
    switch (value) {
    case SPI_START:
        registers_->write(SPI_STATUS, SPI_DONE);
        break;
    case 0:
        registers_->write(SPI_STATUS, SPI_IDLE);
        break;
    default:
        break;
    }
}

void HSBEmulator::handle_i2c_control_write(uint32_t address, uint32_t value)
{
    switch (value) {
    case hololink::I2C_START: {
        registers_->write(I2C_STATUS, hololink::I2C_DONE);
        break;
    }
    default:
        registers_->write(I2C_STATUS, I2C_IDLE);
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
        registers_->write(SPI_STATUS, SPI_DONE);
        break;
    case hololink::I2C_CTRL:
        registers_->write(I2C_STATUS, hololink::I2C_DONE);
        break;
    default:
        registers_->write(last_read_address_, hololink::I2C_DONE);
        break;
    }
}

void HSBEmulator::handle_read_message(ControlMessage& message, int control_socket, struct sockaddr_in* host_addr, socklen_t host_addr_len)
{
    uint32_t address = message.address;

    if (detect_poll(address)) {
        handle_poll();
    }
    uint32_t value = registers_->read(address);

    uint8_t reply[REPLY_MESSAGE_SIZE];
    message.cmd_code = 0x80 | message.cmd_code;
    message.value = value;
    message.status = hololink::RESPONSE_SUCCESS;
    hololink::core::Serializer serializer(reply, sizeof(reply));
    size_t reply_length = serialize_reply_message(serializer, message);
    if (!reply_length) {
        fprintf(stderr, "serialize_reply_message failed\n");
        return;
    }

    if (sendto(control_socket, reply, reply_length, 0, (struct sockaddr*)host_addr, host_addr_len) < 0) {
        struct in_addr addr = ((struct sockaddr_in*)host_addr)->sin_addr;
        fprintf(stderr, "sendto in handle_read_message failed: %d - %s - host_addr: %s\n", errno, strerror(errno), inet_ntoa(addr));
    }
}
void HSBEmulator::handle_write_message(ControlMessage& message, int control_socket, struct sockaddr_in* host_addr, socklen_t host_addr_len)
{
    uint32_t address = message.address;
    uint32_t value = message.value;

    last_read_address_ = address; // TODO: fix this. workaround for polling such that intervening writes reset the poll read

    registers_->write(address, value);
    switch (address) {
    case hololink::SPI_CTRL:
        handle_spi_control_write(address, value);
        break;
    case hololink::I2C_CTRL:
        handle_i2c_control_write(address, value);
        break;
    default:
        break;
    }
    if (message.flags & hololink::REQUEST_FLAGS_ACK_REQUEST) {

        uint8_t reply[REPLY_MESSAGE_SIZE];
        message.cmd_code = 0x80 | message.cmd_code;
        message.status = hololink::RESPONSE_SUCCESS;
        hololink::core::Serializer serializer(reply, sizeof(reply));
        size_t reply_length = serialize_reply_message(serializer, message);
        if (!reply_length) {
            fprintf(stderr, "serialize_reply_message failed\n");
            return;
        }

        if (sendto(control_socket, reply, reply_length, 0, (struct sockaddr*)host_addr, host_addr_len) < 0) {
            struct in_addr addr = ((struct sockaddr_in*)host_addr)->sin_addr;
            fprintf(stderr, "sendto in handle_write_message failed: %d - %s - host_addr: %s\n", errno, strerror(errno), inet_ntoa(addr));
        }
    }
}
void HSBEmulator::handle_invalid_message(ControlMessage& message, int control_socket, struct sockaddr_in* host_addr, socklen_t host_addr_len)
{
    uint8_t reply[REPLY_MESSAGE_SIZE];
    message.status = hololink::RESPONSE_INVALID_CMD;
    hololink::core::Serializer serializer(reply, sizeof(reply));
    size_t reply_length = serialize_reply_message(serializer, message);
    if (!reply_length) {
        fprintf(stderr, "serialize_reply_message failed\n");
        return;
    }

    if (sendto(control_socket, reply, reply_length, 0, (struct sockaddr*)host_addr, host_addr_len) < 0) {
        struct in_addr addr = ((struct sockaddr_in*)host_addr)->sin_addr;
        fprintf(stderr, "sendto in handle_invalid_message failed: %d - %s - host_addr: %s\n", errno, strerror(errno), inet_ntoa(addr));
    }
}

}