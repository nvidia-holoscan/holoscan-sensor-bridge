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

#include "STM32/i2c.hpp"
#include "../../hsb_config.hpp"
#include "STM32/hsb_emulator.hpp"
#include "STM32/stm32_system.h"
#include "board.h"
#include "utils.hpp"
#include <climits>
#include <cstring>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#define I2C_TIMEOUT 100

bool i2c_initialized = false;

static void (*Error_Handler)(const char* str) = hololink::emulation::Error_Handler;

int i2c_init(I2C_HandleTypeDef* hi2c)
{
    if (i2c_initialized) {
        return 0;
    }
    if (!hi2c) {
        i2c_initialized = false;
        return 1;
    }

    hi2c->Instance = STM32_CONF_I2C_INSTANCE;
    hi2c->Init.Timing = STM32_CONF_I2C_TIMING;
    hi2c->Init.OwnAddress1 = 0;
    hi2c->Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
    hi2c->Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
    hi2c->Init.OwnAddress2 = 0;
    hi2c->Init.OwnAddress2Masks = I2C_OA2_NOMASK;
    hi2c->Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
    hi2c->Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
    if (HAL_I2C_Init(hi2c) != HAL_OK) {
        Error_Handler("Failed to initialize I2C");
    }

    if (HAL_I2CEx_ConfigAnalogFilter(hi2c, I2C_ANALOGFILTER_ENABLE) != HAL_OK) {
        Error_Handler("Failed to configure I2C analog filter");
    }

    if (HAL_I2CEx_ConfigDigitalFilter(hi2c, 0) != HAL_OK) {
        Error_Handler("Failed to configure I2C digital filter");
    }

    i2c_initialized = true;
    return 0;
}

// HAL_I2C_MspInit / HAL_I2C_MspDeInit live in the board-specific source (board.c).

namespace hololink::emulation {

// File-scope STM32 extension (no heap). I2CController::ctxt_ holds a pointer to its
// embedded `base` (the common I2CControllerCtxt).
STM32I2CControllerCtxt I2C_CONTROLLER_CTXT = { .base = { .control_address = I2C_CTRL } };

// Helper: downcast the common I2CControllerCtxt* held by I2CController::ctxt_ to the
// STM32 extension. Standard-layout guarantees `&ext->base == ext`. Used to reach the
// STM32-only HAL handle (hi2c).
static inline STM32I2CControllerCtxt* stm32_i2c_ctxt(I2CControllerCtxt* base)
{
    return reinterpret_cast<STM32I2CControllerCtxt*>(base);
}

I2CController::I2CController(__attribute__((unused)) HSBEmulator& hsb_emulator, uint32_t controller_address)
{
    // The file-scope static pool's slot was initialized with `.control_address =
    // I2C_CTRL`. Reject mismatches early so a wrongly-wired HSBEmulator caller is
    // caught before reset() overwrites the field.
    if (I2C_CONTROLLER_CTXT.base.control_address != controller_address) {
        Error_Handler("I2C controller address mismatch");
    }
    ctxt_.reset(&I2C_CONTROLLER_CTXT.base);
    ctxt_.get_deleter() = [](I2CControllerCtxt*) { /* statically allocated */ };
    reset(controller_address);
}

void I2CController::start()
{
    if (i2c_init(&stm32_i2c_ctxt(ctxt_.get())->hi2c)) {
        return;
    }
    ctxt_->status = I2C_IDLE;
    memset(ctxt_->registers, 0, sizeof(ctxt_->registers));
    memset(ctxt_->data, 0, sizeof(ctxt_->data));
    ctxt_->running = true;
}

void I2CController::stop()
{
    ctxt_->running = false;
}

bool I2CController::is_running()
{
    return ctxt_->running;
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
    // shift right by 15 because the input to HAL_I2C_Master_* use 7:1 bits for the address and 0 bit for R/W internally
    uint16_t peripheral_address = (value >> 15) & 0xFFFF;
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
    if (HAL_I2C_IsDeviceReady(&stm32_i2c_ctxt(i2c_ctxt)->hi2c, peripheral_address, 2, I2C_TIMEOUT) != HAL_OK) {
        i2c_ctxt->status = I2C_I2C_NAK;
        return;
    }

    if (num_bytes_write) {
        HAL_StatusTypeDef status = HAL_I2C_Master_Transmit(&stm32_i2c_ctxt(i2c_ctxt)->hi2c, peripheral_address, (uint8_t*)&i2c_ctxt->data[0], num_bytes_write, I2C_TIMEOUT);
        if (status != HAL_OK) {
            i2c_ctxt->status = I2C_I2C_ERR;
            return;
        }
    }
    if (num_bytes_read) {
        HAL_StatusTypeDef status = HAL_I2C_Master_Receive(&stm32_i2c_ctxt(i2c_ctxt)->hi2c, peripheral_address, (uint8_t*)&i2c_ctxt->data[0], num_bytes_read, I2C_TIMEOUT);
        if (status != HAL_OK) {
            i2c_ctxt->status = I2C_I2C_ERR;
            return;
        }
    }

    i2c_ctxt->status = I2C_DONE;
}

}