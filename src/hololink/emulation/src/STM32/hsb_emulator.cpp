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

#include <functional>

#include "../../hsb_config.hpp"
#include "../common/apb_events.hpp"
#include "STM32/gpio.hpp"
#include "STM32/hsb_emulator.hpp"
#include "STM32/i2c.hpp"
#include "STM32/spi.hpp"
#include "STM32/stm32_system.h"
#include "STM32/tim.h"
#include "data_plane.hpp"
#include <climits>
#include <cstring>

namespace hololink::emulation {

// File-scope STM32 extension instance (no heap on bare-metal). HSBEmulator's
// ctxt_ holds a pointer to its embedded `base` (the common HSBEmulatorCtxt).
static struct STM32HSBEmulatorCtxt HSBEMULATORCTXT;

// Helper: downcast the common HSBEmulatorCtxt* held by HSBEmulator::ctxt_ to the STM32
// extension. Standard-layout guarantees `&ext->base == ext`. Used by HSBEmulator methods
// (STM32-side) to reach STM32-only fields (eth_handle, spi_ctxt,
// data_plane_list, data_plane_count, up_time_msec, next_bootp_time_msec).
static inline STM32HSBEmulatorCtxt* stm32_hsb_ctxt(HSBEmulatorCtxt* base)
{
    return reinterpret_cast<STM32HSBEmulatorCtxt*>(base);
}

void HSBEmulator_deleter(struct HSBEmulatorCtxt* base)
{
    // Downcast to the STM32 extension and reset the storage so a re-construction
    // starts from zeroed state, matching the previous behavior.
    STM32HSBEmulatorCtxt* ext = stm32_hsb_ctxt(base);
    *ext = STM32HSBEmulatorCtxt();
}

int read_hsb_data(void* ctxt, struct AddressValuePair* addr_val, int max_count);
int ptp_readback_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count);
int ptp_configure_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count);
int reset_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count);

HSBEmulator::HSBEmulator(const HSBConfiguration& config)
    : ctxt_(&HSBEMULATORCTXT.base)
    , configuration_(config)
    , i2c_controller_(*this, hololink::I2C_CTRL)
{
    const char* config_error = validate_configuration(&configuration_);
    if (config_error) {
        Error_Handler(NULL);
    }

    // initialize the system, but leave individual modules uninitialized
    MPU_Config();
    HAL_Init();
    SystemClock_Config();
    // requires HAL_init. Here before net_init because the MAC address is used by DataPlane before the network is initialized.
    generate_mac_address();

    this->ctxt_.get_deleter() = HSBEmulator_deleter;
    // ctxt_->hsb_emulator and ctxt_->register_memory's dispatch ctxt are wired by
    // reset() below.
    STM32HSBEmulatorCtxt* sctxt = stm32_hsb_ctxt(ctxt_.get());
    sctxt->spi_ctxt = &SPI_CONTROLLER_CTXT;
    spi_constructor(sctxt->spi_ctxt, SPI_CTRL);
    ctxt_->register_memory.set_ctxt(ctxt_.get());

    // Register the platform-invariant callbacks (HSB version, PTP, APB RAM, async events,
    // I2C register block). STM32 keeps its GPIO/SPI extras below.
    reset();

    // register callbacks for gpio
    CHECK_STATUS(ctxt_->cp_write_map.set({ GPIO_OUTPUT_BASE_REGISTER, GPIO_OUTPUT_BASE_REGISTER + GPIO_REGISTER_RANGE }, { GPIO_set_value, nullptr }),
        "Failed to register write callback for gpio output");
    CHECK_STATUS(ctxt_->cp_write_map.set({ GPIO_DIRECTION_BASE_REGISTER, GPIO_DIRECTION_BASE_REGISTER + GPIO_REGISTER_RANGE }, { GPIO_set_direction, nullptr }),
        "Failed to register write callback for gpio direction");
    CHECK_STATUS(ctxt_->cp_read_map.set({ GPIO_STATUS_BASE_REGISTER, GPIO_STATUS_BASE_REGISTER + GPIO_REGISTER_RANGE }, { GPIO_get_value, nullptr }),
        "Failed to register read callback for gpio value");
    CHECK_STATUS(ctxt_->cp_read_map.set({ GPIO_DIRECTION_BASE_REGISTER, GPIO_DIRECTION_BASE_REGISTER + GPIO_REGISTER_RANGE }, { GPIO_get_direction, nullptr }),
        "Failed to register read callback for gpio direction");

    // register callbacks for spi
    auto spi_ctxt = sctxt->spi_ctxt;
    CHECK_STATUS(ctxt_->cp_read_map.set({ SPI_CTRL + SPI_REG_DATA_BUFFER, SPI_CTRL + SPI_REG_DATA_BUFFER + SPI_DATA_BUFFER_SIZE }, { spi_readback_cb, spi_ctxt }),
        "Failed to register read callback for spi data buffer");
    CHECK_STATUS(ctxt_->cp_read_map.set({ SPI_CTRL, SPI_CTRL + SPI_REG_NUM_CMD_BYTES + REGISTER_SIZE }, { spi_readback_cb, spi_ctxt }),
        "Failed to register read callback for spi cmd bytes");
    CHECK_STATUS(ctxt_->cp_read_map.set({ SPI_CTRL + SPI_REG_STATUS, SPI_CTRL + SPI_REG_STATUS + REGISTER_SIZE }, { spi_readback_cb, spi_ctxt }),
        "Failed to register read callback for spi status");
    CHECK_STATUS(ctxt_->cp_write_map.set({ SPI_CTRL + SPI_REG_DATA_BUFFER, SPI_CTRL + SPI_REG_DATA_BUFFER + SPI_DATA_BUFFER_SIZE }, { spi_configure_cb, spi_ctxt }),
        "Failed to register write callback for spi data buffer");
    CHECK_STATUS(ctxt_->cp_write_map.set({ SPI_CTRL, SPI_CTRL + SPI_REG_NUM_CMD_BYTES + REGISTER_SIZE }, { spi_configure_cb, spi_ctxt }),
        "Failed to register write callback for spi cmd bytes");
}

void control_plane_reply(HSBEmulatorCtxt* ctxt, ETH_BufferTypeDef* buffer)
{
    // The buffer was prepared by handle_control_packet -> prepare_control_plane_reply:
    // headers swapped in place, buffer->len updated to the outgoing frame size. The
    // shared eth_hal_send walks the (single-segment) chain, sums lengths, and calls
    // HAL_ETH_Transmit — identical setup to the COE/RoCEv2 transmit path.
    eth_hal_send(&stm32_hsb_ctxt(ctxt)->eth_handle, buffer);
}

int HSBEmulator::add_data_plane(DataPlane& data_plane)
{
    STM32HSBEmulatorCtxt* sctxt = stm32_hsb_ctxt(ctxt_.get());
    if (sctxt->data_plane_count >= MAX_DATA_PLANES) {
        return 1;
    }
    sctxt->data_plane_list[sctxt->data_plane_count++] = &data_plane;
    return 0;
}

int HSBEmulator::handle_msgs()
{
    ETH_BufferTypeDef* buffer;
    int status = eth_receive(&buffer);
    if (0 > status) {
        Error_Handler(NULL);
    }

    // RISK: potential infinite loop if under extreme load. TODO: need strategy to tradeoff packet loss against infinite loop.
    while (buffer) {
        handle_control_packet(ctxt_.get(), buffer);
        eth_release(buffer);
        status = eth_receive(&buffer);
        if (0 > status) {
            Error_Handler(NULL);
        }
    }

    // trigger events-based messages. Done this way so we don't have to consume a timer. Timing here is not critical so interrupts would be wasted.
    STM32HSBEmulatorCtxt* sctxt = stm32_hsb_ctxt(ctxt_.get());
    sctxt->up_time_msec = HAL_GetTick();
    if (sctxt->up_time_msec >= sctxt->next_bootp_time_msec) {
        for (unsigned short i = 0; i < sctxt->data_plane_count; i++) {
            DataPlane* data_plane = sctxt->data_plane_list[i];
            if (data_plane->is_running() && (data_plane->broadcast_bootp() != HAL_OK)) {
                Error_Handler("Failed to broadcast bootp");
            }
        }
        // schedule next bootp
        sctxt->next_bootp_time_msec = sctxt->up_time_msec + BOOTP_INTERVAL_SEC * 1000;
    }
    return 0;
}

HSBEmulator::~HSBEmulator()
{
}

bool HSBEmulator::is_running()
{
    return ctxt_->running;
}

/* this will block until all registered DataPlanes have stopped */
void HSBEmulator::stop()
{
    // if not running, do nothing...idempotent operation
    if (!is_running()) {
        return;
    }

    STM32HSBEmulatorCtxt* sctxt = stm32_hsb_ctxt(ctxt_.get());
    for (unsigned short i = 0; i < sctxt->data_plane_count; i++) {
        DataPlane* data_plane = sctxt->data_plane_list[i];
        if (data_plane->is_running()) {
            data_plane->stop();
        }
    }
    ctxt_->running = false;
}

// TODO: once Linux moves to the APB callbacks, move this to common
void HSBEmulator::start()
{
    if (is_running()) {
        return;
    }

    // GPIO should be first because it will initialize all the clocks
    if (GPIO_init(nullptr)) {
        return;
    }

    STM32HSBEmulatorCtxt* sctxt = stm32_hsb_ctxt(ctxt_.get());
    // initialize networking module. This is not in the DataPlane because it is global to the HSBEmulator instance since we only have one interface
    if (net_init(&sctxt->eth_handle)) {
        return;
    }

    // initialize timer module
    if (tim_init(nullptr)) {
        return;
    }

    // i2c_controller_ handles initializing the i2c module
    i2c_controller_.start();
    if (!i2c_controller_.is_running()) {
        return;
    }

    // do not have spi_controller yet, so initialize spi module here
    if (spi_init(&sctxt->spi_ctxt->hspi)) {
        return;
    }

    for (unsigned short i = 0; i < sctxt->data_plane_count; i++) {
        sctxt->data_plane_list[i]->start();
    }

    ctxt_->cp_write_map.build();
    ctxt_->cp_read_map.build();

    // start the control plane thread
    ctxt_->running = true;

    sctxt->up_time_msec = 0;
    sctxt->next_bootp_time_msec = 0;

    if (!is_running()) {
        Error_Handler(NULL);
    }
}

} // namespace hololink::emulation
