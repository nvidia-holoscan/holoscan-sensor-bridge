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
#include "STM32/hsb_emulator.hpp"
#include "STM32/stm32_system.h"
#include <climits>
#include <cstring>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#define I2C_TIMEOUT 100

bool i2c_initialized = false;

int i2c_init(I2C_HandleTypeDef* hi2c)
{
    if (i2c_initialized) {
        return 0;
    }
    if (!hi2c) {
        i2c_initialized = false;
        return 1;
    }

    hi2c->Instance = I2C1;
    hi2c->Init.Timing = 0x20303E5D;
    hi2c->Init.OwnAddress1 = 0;
    hi2c->Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
    hi2c->Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
    hi2c->Init.OwnAddress2 = 0;
    hi2c->Init.OwnAddress2Masks = I2C_OA2_NOMASK;
    hi2c->Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
    hi2c->Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
    if (HAL_I2C_Init(hi2c) != HAL_OK) {
        Error_Handler();
    }

    if (HAL_I2CEx_ConfigAnalogFilter(hi2c, I2C_ANALOGFILTER_ENABLE) != HAL_OK) {
        Error_Handler();
    }

    if (HAL_I2CEx_ConfigDigitalFilter(hi2c, 0) != HAL_OK) {
        Error_Handler();
    }

    i2c_initialized = true;
    return 0;
}

void HAL_I2C_MspInit(I2C_HandleTypeDef* hi2c)
{
    GPIO_InitTypeDef GPIO_InitStruct = { 0 };
    RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = { 0 };
    if (hi2c->Instance == I2C1) {
        PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_I2C1;
        PeriphClkInitStruct.I2c1ClockSelection = RCC_I2C1CLKSOURCE_PCLK1;
        if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK) {
            Error_Handler();
        }

        __HAL_RCC_GPIOB_CLK_ENABLE();
        /**I2C1 GPIO Configuration
        PB6     ------> I2C1_SCL
        PB9     ------> I2C1_SDA
        */
        GPIO_InitStruct.Pin = GPIO_PIN_6 | GPIO_PIN_9;
        GPIO_InitStruct.Mode = GPIO_MODE_AF_OD;
        GPIO_InitStruct.Pull = GPIO_NOPULL;
        GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
        GPIO_InitStruct.Alternate = GPIO_AF4_I2C1;
        HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

        __HAL_RCC_I2C1_CLK_ENABLE();
    }
}

void HAL_I2C_MspDeInit(I2C_HandleTypeDef* i2cHandle)
{
    if (i2cHandle->Instance == I2C1) {
        __HAL_RCC_I2C1_CLK_DISABLE();
        HAL_GPIO_DeInit(GPIOB, GPIO_PIN_6);
        HAL_GPIO_DeInit(GPIOB, GPIO_PIN_9);
    }
}

namespace hololink::emulation {

struct I2CControllerCtxt {
    I2C_HandleTypeDef hi2c;
    I2CController* i2c_controller;
    uint32_t registers[(I2C_REG_CLK_CNT - I2C_REG_CONTROL) / REGISTER_SIZE + 1];
    uint32_t data[I2C_DATA_BUFFER_SIZE / REGISTER_SIZE];
    uint32_t control_address;
    uint32_t data_address;
    uint32_t status;
    bool running { false };
} I2C_CONTROLLER_CTXT = { .control_address = I2C_CTRL };
// if we need to add more controllers, we can add more instances of I2C_CONTROLLER_CTXT and use a map of their control addresses to differentiate

I2CController::I2CController(__attribute__((unused)) HSBEmulator& hsb_emulator, uint32_t controller_address)
{
    ctxt_.reset(&I2C_CONTROLLER_CTXT);
    ctxt_.get_deleter() = [](I2CControllerCtxt* p) { (void)p; };
    ctxt_->i2c_controller = this;
    // ctxt_->control_address should already be set in the static instance
    if (ctxt_->control_address != controller_address) {
        Error_Handler();
    }
    ctxt_->data_address = ctxt_->control_address + I2C_REG_DATA_BUFFER;
    ctxt_->status = I2C_IDLE;
    memset(ctxt_->registers, 0, sizeof(ctxt_->registers));
    memset(ctxt_->data, 0, sizeof(ctxt_->data));
    ctxt_->running = false;
}

void I2CController::start()
{
    if (i2c_init(&ctxt_->hi2c)) {
        return;
    }
    ctxt_->status = I2C_IDLE;
    memset(ctxt_->registers, 0, sizeof(ctxt_->registers));
    memset(ctxt_->data, 0, sizeof(ctxt_->data));
    ctxt_->running = true;
}

void I2CController::execute(uint32_t value)
{
    (void)value;
}

void I2CController::stop()
{
    ctxt_->running = false;
}

bool I2CController::is_running()
{
    return ctxt_->running;
}

void I2CController::i2c_execute()
{
    // not used in STM32 context
}

void I2CController::run()
{
    // not used in STM32 context
}

int i2c_readback_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    I2CControllerCtxt* i2c_ctxt = (I2CControllerCtxt*)ctxt;
    struct AddressValuePair* cur = addr_val;
    int n = 0;
    while (n < max_count) {
        uint32_t address = AVP_GET_ADDRESS(cur);
        if (address >= i2c_ctxt->data_address && address < i2c_ctxt->data_address + I2C_DATA_BUFFER_SIZE) {
            AVP_SET_VALUE(cur, i2c_ctxt->data[(address - i2c_ctxt->data_address) / REGISTER_SIZE]);
        } else if (i2c_ctxt->control_address + I2C_REG_STATUS == address) {
            AVP_SET_VALUE(cur, i2c_ctxt->status);
        } else if (address >= i2c_ctxt->control_address && address <= i2c_ctxt->control_address + I2C_REG_CLK_CNT) {
            AVP_SET_VALUE(cur, i2c_ctxt->registers[(address - i2c_ctxt->control_address) / REGISTER_SIZE]);
        } else { // not an I2C address
            return n;
        }
        cur++;
        n++;
    }
    return n;
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
    if (HAL_I2C_IsDeviceReady(&i2c_ctxt->hi2c, peripheral_address, 2, I2C_TIMEOUT) != HAL_OK) {
        i2c_ctxt->status = I2C_I2C_NAK;
        return;
    }

    if (num_bytes_write) {
        HAL_StatusTypeDef status = HAL_I2C_Master_Transmit(&i2c_ctxt->hi2c, peripheral_address, (uint8_t*)&i2c_ctxt->data[0], num_bytes_write, I2C_TIMEOUT);
        if (status != HAL_OK) {
            i2c_ctxt->status = I2C_I2C_ERR;
            return;
        }
    }
    if (num_bytes_read) {
        HAL_StatusTypeDef status = HAL_I2C_Master_Receive(&i2c_ctxt->hi2c, peripheral_address, (uint8_t*)&i2c_ctxt->data[0], num_bytes_read, I2C_TIMEOUT);
        if (status != HAL_OK) {
            i2c_ctxt->status = I2C_I2C_ERR;
            return;
        }
    }

    i2c_ctxt->status = I2C_DONE;
}

int i2c_configure_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    I2CControllerCtxt* i2c_ctxt = (I2CControllerCtxt*)ctxt;
    struct AddressValuePair* cur = addr_val;
    int n = 0;
    while (n < max_count) {
        uint32_t address = AVP_GET_ADDRESS(cur);
        if (address >= i2c_ctxt->control_address && address <= i2c_ctxt->control_address + I2C_REG_CLK_CNT) {
            i2c_ctxt->registers[(address - i2c_ctxt->control_address) / REGISTER_SIZE] = AVP_GET_VALUE(cur);
            // if on the control address, execute the transaction command
            if (address == i2c_ctxt->control_address) {
                i2c_transaction(i2c_ctxt, i2c_ctxt->registers[0]);
            }
        } else if (address >= i2c_ctxt->data_address && address < i2c_ctxt->data_address + I2C_DATA_BUFFER_SIZE) {
            i2c_ctxt->data[(address - i2c_ctxt->data_address) / REGISTER_SIZE] = AVP_GET_VALUE(cur);
        } else { // not an I2C address; cannot write to I2C_REG_STATUS
            return n;
        }
        cur++;
        n++;
    }
    return n;
}

}