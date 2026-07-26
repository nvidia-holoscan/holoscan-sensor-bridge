/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <queue>

#include <thread>
#include <vector>

#include <condition_variable>
#include <mutex>

#include <chrono>
#include <climits>

#include <atomic>
#include <cstring>
#include <memory>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

#include "../common/apb_events.hpp"
#include "address_map.hpp"
#include "data_plane.hpp"
#include "hsb_emulator.hpp"
#include "i2c.hpp"
#include "i2c_interface.hpp"
#include "net.hpp"
#include "utils.hpp"

// this is wrapping POSIX poll() flags. We read on any type of read and stop on all errors/hangups
#define POLL_RD_ANY (POLLIN | POLLPRI)
#define POLL_STOP (POLLHUP | POLLERR | POLLNVAL)

#define MAKE_OPAQUE_UNIQUE(var, type, ...) \
    var.reset(new type(__VA_ARGS__));      \
    var.get_deleter() = [](type* p) { delete p; }

int GPIO_init(void* ctxt)
{
    (void)ctxt;
    return 0;
}

int net_init(void* ctxt)
{
    (void)ctxt;
    return 0;
}

int tim_init(void* ctxt)
{
    (void)ctxt;
    return 0;
}

int spi_init(void* ctxt)
{
    (void)ctxt;
    return 0;
}

// msleep — target-portable millisecond sleep. Declared `extern "C"` in
// hsb_emulator.hpp so the STM32 build can implement it directly in tim.c.
// usleep takes microseconds and returns 0 / -1; we always return 0 to match
// the prototype's documented "0 on success" and ignore EINTR (the loops that
// use msleep will simply continue on the next iteration).
extern "C" int msleep(unsigned milliseconds)
{
    usleep((useconds_t)milliseconds * 1000u);
    return 0;
}

namespace hololink::emulation {

// Helper: downcast the common HSBEmulatorCtxt* held by HSBEmulator::ctxt_ to the Linux
// extension. Standard-layout guarantees `&ext->base == ext` so this cast is sound.
// Used by HSBEmulator methods (Linux-side) and free functions in this file to reach
// Linux-only fields (data_plane_list, control_thread_, running_mutex, dp_*_cache, ...).
static inline LinuxHSBEmulatorCtxt* linux_hsb_ctxt(HSBEmulatorCtxt* base)
{
    return reinterpret_cast<LinuxHSBEmulatorCtxt*>(base);
}

// Helper: downcast the common I2CControllerCtxt* held by I2CController::ctxt_ to the
// Linux extension. Standard-layout guarantees `&ext->base == ext`. Used by the free
// function i2c_transaction() and the I2CController methods to reach Linux-only fields
// (i2c_bus_map, i2c_mutex).
static inline LinuxI2CControllerCtxt* linux_i2c_ctxt(I2CControllerCtxt* base)
{
    return reinterpret_cast<LinuxI2CControllerCtxt*>(base);
}

I2CController::I2CController(HSBEmulator& hsb_emulator, uint32_t controller_address)
{
    (void)hsb_emulator; // no longer used by the linux I2CController (no register read/writes)
    LinuxI2CControllerCtxt* lctxt = new LinuxI2CControllerCtxt();
    ctxt_.reset(&lctxt->base);
    ctxt_.get_deleter() = [](I2CControllerCtxt* base) {
        delete reinterpret_cast<LinuxI2CControllerCtxt*>(base);
    };
    reset(controller_address);
}

void i2c_transaction(I2CControllerCtxt* i2c_ctxt, uint32_t value)
{
    uint16_t cmd = value & 0xFFFF;
    static const uint16_t general_call_address = 0x00; // using reserved i2c peripheral address for testing i2c transactions
    if (!cmd && i2c_ctxt->status == I2C_DONE) {
        i2c_ctxt->status = I2C_IDLE;
        return;
    }
    if (cmd != I2C_START) { // 10B address not yet supported
        return;
    }

    // find the peripheral. treat nullptr as a special case for a peripheral that consumes all data and returns 0s (user does not add any peripherals)
    // this will only work for stateless sensors
    uint32_t bus_en = i2c_ctxt->registers[I2C_REG_BUS_EN / REGISTER_SIZE];

    uint16_t peripheral_address = (value >> 16);
    i2c_ctxt->status = I2C_BUSY;

    uint16_t num_bytes_write = i2c_ctxt->registers[I2C_REG_NUM_BYTES / REGISTER_SIZE] & 0xFFFF;
    uint16_t num_bytes_read = (i2c_ctxt->registers[I2C_REG_NUM_BYTES / REGISTER_SIZE] >> 16) & 0xFFFF;

#ifdef I2C_TEST_MODE
    if (general_call_address == peripheral_address) {
        // write dummy data to the data buffer for receipt
        for (uint16_t i = 0; i < num_bytes_read; i++) {
            i2c_ctxt->data[i] = 0x01u << i;
        }
        // clear out remaining write buffer
        for (uint16_t i = num_bytes_read; i < num_bytes_write; i++) {
            i2c_ctxt->data[i] = 0;
        }
        i2c_ctxt->status = I2C_DONE;
        return;
    }
#else
    (void)general_call_address;
#endif

    I2CPeripheral* peripheral = nullptr;
    LinuxI2CControllerCtxt* lctxt = linux_i2c_ctxt(i2c_ctxt);
    // Lock i2c_mutex across the entire bus_map lookup + peripheral->i2c_transaction()
    // call. attach_i2c_peripheral() can mutate i2c_bus_map from the application thread.
    std::scoped_lock<std::mutex> lock(lctxt->i2c_mutex);

    if (peripheral_address != LII2CExpanderPeripheral::PERIPHERAL_ADDRESS) {
        auto& bus_map = lctxt->i2c_bus_map[bus_en];
        auto exp_it = bus_map.find(LII2CExpanderPeripheral::PERIPHERAL_ADDRESS);
        if (exp_it != bus_map.end()) {
            if (auto* expander = dynamic_cast<LII2CExpanderPeripheral*>(exp_it->second)) {
                bus_en += static_cast<uint32_t>(expander->output_state() >> 1);
            }
        }
    }

    std::unordered_map<uint16_t, I2CPeripheral*>& i2c_peripheral_map = lctxt->i2c_bus_map[bus_en];
    auto it = i2c_peripheral_map.find(peripheral_address);
    if (it != i2c_peripheral_map.end()) {
        // peripheral will only receive i2c_transactions if it is registered
        peripheral = it->second;
    }

    I2CStatus status = I2CStatus::I2C_STATUS_SUCCESS;
    // send transaction to the peripheral
    // to allow for stateless sensors/no i2c peripheral associated, nullptr is allowed and will no-op the i2c_transaction() call
    if (peripheral) {
        status = peripheral->i2c_transaction(peripheral_address, (uint8_t*)&i2c_ctxt->data[0], num_bytes_write, (uint8_t*)&i2c_ctxt->data[0], num_bytes_read);
    }

    if (status != I2CStatus::I2C_STATUS_SUCCESS) {
        fprintf(stderr, "I2CController: i2c_transaction reports non-success status. %d\n", status);
        i2c_ctxt->status = I2C_I2C_ERR;
        return; // wrote error to I2C. Don't read data back
    }

    i2c_ctxt->status = I2C_DONE;
}

void I2CController::start()
{
    LinuxI2CControllerCtxt* lctxt = linux_i2c_ctxt(ctxt_.get());
    std::scoped_lock<std::mutex> lock(lctxt->i2c_mutex);

    // start the peripherals
    for (auto& bus_map : lctxt->i2c_bus_map) {
        for (auto& peripheral : bus_map.second) {
            peripheral.second->start();
        }
    }

    ctxt_->running = true;
}

void I2CController::stop()
{
    if (!is_running()) {
        return;
    }
    LinuxI2CControllerCtxt* lctxt = linux_i2c_ctxt(ctxt_.get());
    std::scoped_lock<std::mutex> lock(lctxt->i2c_mutex);
    ctxt_->running = false;
    // stop the peripherals, which may be no-op. Do not monitor for completion.
    for (auto& bus_map : lctxt->i2c_bus_map) {
        for (auto& peripheral : bus_map.second) {
            peripheral.second->stop();
        }
    }
}

bool I2CController::is_running()
{
    return ctxt_->running;
}

void I2CController::attach_i2c_peripheral(uint32_t bus_address, uint16_t peripheral_address, I2CPeripheral* peripheral)
{
    if (!peripheral) {
        throw std::runtime_error("I2CController: peripheral cannot be nullptr");
    }
    LinuxI2CControllerCtxt* lctxt = linux_i2c_ctxt(ctxt_.get());
    std::scoped_lock<std::mutex> lock(lctxt->i2c_mutex);
    if (ctxt_->running) {
        throw std::runtime_error("I2CController: cannot attach peripherals while running");
    }

    // get peripheral map for the bus and default initialize if using bus address that is not in the map
    std::unordered_map<uint16_t, I2CPeripheral*>& i2c_peripheral_map = lctxt->i2c_bus_map[bus_address];

    auto it = i2c_peripheral_map.find(peripheral_address);
    if (it != i2c_peripheral_map.end()) {
        // Allow a user attach (e.g. Vb1940Emulator's VCL_EN_I2C_ADDRESS_1 = 0x70) to silently
        // override the auto-attached LII2CExpanderPeripheral. Other collisions are still errors.
        if (!dynamic_cast<LII2CExpanderPeripheral*>(it->second)) {
            throw std::runtime_error("I2CController: peripheral address " + std::to_string(peripheral_address) + " is already in use");
        }
    }
    i2c_peripheral_map[peripheral_address] = peripheral;
}

class ControlThread {
public:
    ControlThread(HSBEmulatorCtxt* ctxt)
        : ctxt_(ctxt)
    {
        int socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
        if (socket_fd < 0) {
            fprintf(stderr, "Failed to create control UDP socket: %d - %s\n", errno, strerror(errno));
            throw std::runtime_error("Failed to create control socket");
        }

        // Enable address reuse
        int reuse = 1;
        if (setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
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
        if (bind(socket_fd, (struct sockaddr*)&control_addr, sizeof(control_addr)) < 0) {
            fprintf(stderr, "Failed to bind control socket: %d - %s\n", errno, strerror(errno));
            close(socket_fd);
            throw std::runtime_error("Failed to bind control socket");
        }
        linux_hsb_ctxt(ctxt_)->control_socket_fd_ = socket_fd;
        running = true;
        thread_ = std::thread(&ControlThread::loop, this);
    }
    ~ControlThread()
    {
        running = false;
        if (thread_.joinable()) {
            thread_.join();
        }
        int& fd = linux_hsb_ctxt(ctxt_)->control_socket_fd_;
        if (fd >= 0) {
            close(fd);
        }
    }

    std::mutex* get_metadata_mutex() { return &metadata_mutex; }

private:
    void loop()
    {
        struct pollfd fds[1] = { { .fd = linux_hsb_ctxt(ctxt_)->control_socket_fd_, .events = POLL_RD_ANY, .revents = 0 } };
        while (true) {
            if (!running) {
                break;
            }
            // poll on control socket for any events
            switch (poll(fds, 1, CONTROL_INTERVAL_MSEC)) {
            case 0:
                break;
            case -1:
                fprintf(stderr, "poll() failed. disconnecting: %d - %s\n", errno, strerror(errno));
                break;
            default:
                if ((fds[0].revents & POLL_RD_ANY)) {
                    // received and deserialize the control message buffer
                    BSDControlMessage ctrl_msg;
                    ETH_BufferTypeDef* message_buffer = &ctrl_msg.message_buffer;
                    message_buffer->buffer = reinterpret_cast<uint8_t*>(&ctrl_msg.data[2]); // write 2 bytes into data so that the IP/UDP headers are dword aligned
                    message_buffer->len = 0;
                    message_buffer->next = nullptr;
                    udphdr* udp_hdr = NET_GET_UDP_HDR(&ctrl_msg.message_buffer);
                    char* payload_ptr = (char*)NET_GET_UDP_PAYLOAD(udp_hdr);
                    socklen_t host_addr_len = sizeof(ctrl_msg.host_addr);
                    const ssize_t message_length = recvfrom(linux_hsb_ctxt(ctxt_)->control_socket_fd_, payload_ptr, sizeof(ctrl_msg.data) - (payload_ptr - &ctrl_msg.data[0]), 0, (struct sockaddr*)&ctrl_msg.host_addr, &host_addr_len);

                    if (message_length < 0) {
                        fprintf(stderr, "recvfrom failed: %d - %s\n", errno, strerror(errno));
                        continue;
                    } else if (message_length < MIN_VALID_CONTROL_LENGTH) {
                        fprintf(stderr, "incomplete message received: %zu - %u\n", message_length, MIN_VALID_CONTROL_LENGTH);
                        continue;
                    }
                    message_buffer->len = message_length;

                    udp_hdr->len = htons(message_length + UDP_HDR_LEN);

                    std::scoped_lock<std::mutex> lock(metadata_mutex); // read or write must synchronization with consumers
                    handle_control_packet(ctxt_, &ctrl_msg.message_buffer);
                    // handle any errors or hangups
                } else if ((fds[0].revents & POLL_STOP)) {
                    if ((fds[0].revents & POLLHUP)) {
                        fprintf(stderr, "connection hangup\n");
                    } else if ((fds[0].revents & POLLERR)) {
                        fprintf(stderr, "poll error\n");
                    } else if ((fds[0].revents & POLLNVAL)) {
                        fprintf(stderr, "invalid control socket\n");
                    }
                }
                break;
            }
        }
    }
    HSBEmulatorCtxt* ctxt_;
    std::thread thread_;
    std::atomic<bool> running { false };
    uint16_t next_sequence_ { 0x100 };
    std::mutex metadata_mutex;
};

std::mutex* get_metadata_mutex(LinuxHSBEmulatorCtxt& ctxt)
{
    return ctxt.control_thread_->get_metadata_mutex();
}

HSBEmulator::HSBEmulator(const HSBConfiguration& config)
    : configuration_(config)
    , i2c_controller_(*this, hololink::I2C_CTRL)
{
    const char* config_error = validate_configuration(&configuration_);
    if (config_error) {
        throw std::runtime_error(config_error);
    }
    // Heap-allocate the Linux extension; ctxt_ stores a pointer to its embedded `base`
    // (the common HSBEmulatorCtxt). Standard-layout guarantees &lctxt->base == lctxt so
    // the deleter can downcast back to LinuxHSBEmulatorCtxt* and delete safely.
    LinuxHSBEmulatorCtxt* lctxt = new LinuxHSBEmulatorCtxt();
    ctxt_.reset(&lctxt->base);
    ctxt_.get_deleter() = [](HSBEmulatorCtxt* base) {
        delete reinterpret_cast<LinuxHSBEmulatorCtxt*>(base);
    };
    // ctxt_->hsb_emulator and ctxt_->register_memory's dispatch ctxt are wired by
    // reset() below, alongside the platform-invariant callback registrations.

    // attach peripherals after the i2c_controller is initialized
    lctxt->renesas_i2c = std::make_unique<RenesasI2CPeripheral>();
    lctxt->renesas_i2c->attach_to_i2c(this->get_i2c(hololink::I2C_CTRL), hololink::BL_I2C_BUS);
    lctxt->li_i2c_expander = std::make_unique<LII2CExpanderPeripheral>();
    lctxt->li_i2c_expander->attach_to_i2c(this->get_i2c(hololink::I2C_CTRL), hololink::CAM_I2C_BUS);

    lctxt->control_thread_ = new ControlThread(ctxt_.get());

    // Register the platform-invariant callbacks (HSB version, PTP, APB RAM, async events,
    // I2C register block). Linux has no GPIO/SPI extras to add on top.
    reset();
}

HSBEmulator::~HSBEmulator()
{
    delete linux_hsb_ctxt(ctxt_.get())->control_thread_;
}

bool detect_poll(uint32_t address);

void handle_poll();

I2CController& HSBEmulator::get_i2c(uint32_t controller_address)
{
    return i2c_controller_;
}

// returns true if any DataPlane is running or control plane thread, false otherwise
bool HSBEmulator::is_running()
{
    if (i2c_controller_.is_running()) {
        return true;
    }
    LinuxHSBEmulatorCtxt* lctxt = linux_hsb_ctxt(ctxt_.get());
    for (auto& data_plane : lctxt->data_plane_list) {
        if (data_plane->is_running()) {
            return true;
        }
    }
    // only control plane left — read base.running under the linux-side mutex.
    std::scoped_lock<std::mutex> lock(lctxt->running_mutex);
    return ctxt_->running;
}

int HSBEmulator::handle_msgs()
{
    // no-op for linux targets
    return 0;
}

void control_plane_reply(HSBEmulatorCtxt* ctxt, ETH_BufferTypeDef* buffer)
{
    // The buffer was prepared by handle_control_packet -> prepare_control_plane_reply:
    // headers swapped in place, buffer->len = full Ethernet frame size. The control
    // socket is a UDP socket; the kernel rebuilds L2/L3/L4 from socket state, so drop
    // all three header sizes from the prepared frame. host_addr is the sockaddr_in
    // captured at recvfrom time, sitting just before message_buffer in the BSD wrapper.
    sockaddr_in* host_addr = (struct sockaddr_in*)(BSDControlMessage*)((char*)buffer - offsetof(BSDControlMessage, message_buffer));
    int16_t status = eth_socket_send(linux_hsb_ctxt(ctxt)->control_socket_fd_,
        host_addr, sizeof(*host_addr),
        buffer, /*skip_head_bytes=*/ETHER_HDR_LEN + IP_HDR_LEN + UDP_HDR_LEN);
    if (status < 0) {
        Error_Handler("Failed to send control plane reply");
    }
}

void HSBEmulator::start()
{
    if (is_running()) {
        return;
    }

    i2c_controller_.start();

    LinuxHSBEmulatorCtxt* lctxt = linux_hsb_ctxt(ctxt_.get());
    for (auto& data_plane : lctxt->data_plane_list) {
        data_plane->start();
    }

    ctxt_->cp_write_map.build();
    ctxt_->cp_read_map.build();

    // start the control plane thread — set base.running under the linux mutex.
    {
        std::scoped_lock<std::mutex> lock(lctxt->running_mutex);
        ctxt_->running = true;
    }

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

    LinuxHSBEmulatorCtxt* lctxt = linux_hsb_ctxt(ctxt_.get());
    for (auto& data_plane : lctxt->data_plane_list) {
        data_plane->stop();
    }

    i2c_controller_.stop();
    // stop the control plane thread (set under the linux mutex) and wait for data planes to stop
    {
        std::scoped_lock<std::mutex> lock(lctxt->running_mutex);
        ctxt_->running = false;
    }
    while (is_running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    } // wait for data planes to stop
}

int HSBEmulator::add_data_plane(DataPlane& data_plane)
{
    std::vector<DataPlane*>& data_plane_list = linux_hsb_ctxt(ctxt_.get())->data_plane_list;
    if (data_plane_list.size() >= MAX_DATA_PLANES) {
        return 1;
    }
    data_plane_list.push_back(&data_plane);
    return 0;
}

DPRegisters* get_or_create_dp_registers(LinuxHSBEmulatorCtxt& ctxt, uint8_t data_plane_id)
{
    auto& cache = ctxt.dp_registers_cache;
    if (data_plane_id >= cache.size()) {
        cache.resize(static_cast<size_t>(data_plane_id) + 1);
    }
    auto& slot = cache[data_plane_id];
    if (!slot) {
        slot = std::make_shared<DPRegisters>();
        std::memset(slot.get(), 0, sizeof(DPRegisters));
    }
    return slot.get();
}

DPSensorRegisters* get_or_create_dp_sensor_registers(LinuxHSBEmulatorCtxt& ctxt, uint8_t sensor_id)
{
    auto& cache = ctxt.dp_sensor_registers_cache;
    if (sensor_id >= cache.size()) {
        cache.resize(static_cast<size_t>(sensor_id) + 1);
    }
    auto& slot = cache[sensor_id];
    if (!slot) {
        slot = std::make_shared<DPSensorRegisters>();
        std::memset(slot.get(), 0, sizeof(DPSensorRegisters));
    }
    return slot.get();
}

} // namespace hololink::emulation
