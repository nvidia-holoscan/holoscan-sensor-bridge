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

#include "STM32/spi.hpp"
#include "STM32/hsb_emulator.hpp"
#include "STM32/stm32_system.h"
#include <climits>
#include <cstring>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#define SPI_TIMEOUT 100

// also maximum value for the prescaler
#define SPI_PRESCALER_MASK 0x0Eu
#define SPI_DATA_SIZE_MASK 0x03u
#define SPI_DATA_SIZE_SHIFT 8
#define SPI_DEFAULT_MODE (SPI_PRESCALER_MASK | SPI_CFG_CPOL | SPI_CFG_CPHA)

struct SpiControllerCtxt SPI_CONTROLLER_CTXT = { .control_address = SPI_CTRL };
// if we need to add more controllers, we can add more instances of SPI_CONTROLLER_CTXT and use a map of their control addresses to differentiate

/* SPI3 init function. this can be called multiple times on the same object
 to change initialization, but will only initialize the clocks and GPIOs once*/
int spi_init(SPI_HandleTypeDef* hspi)
{
    if (!hspi) {
        return 1;
    }

    if (HAL_SPI_Init(hspi) != HAL_OK) {
        return -1;
    }

    return 0;
}

void HAL_SPI_MspInit(SPI_HandleTypeDef* hspi)
{
    GPIO_InitTypeDef GPIO_InitStruct = { 0 };
    if (hspi->Instance == SPI3) {
        __HAL_RCC_SPI3_CLK_ENABLE();
        /**SPI3 GPIO Configuration
        PB2     ------> SPI3_MOSI
        PC10     ------> SPI3_SCK
        PC11     ------> SPI3_MISO
        */
        GPIO_InitStruct.Pin = GPIO_PIN_2;
        GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
        GPIO_InitStruct.Pull = GPIO_NOPULL;
        GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
        GPIO_InitStruct.Alternate = GPIO_AF7_SPI3;
        HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

        GPIO_InitStruct.Pin = GPIO_PIN_10 | GPIO_PIN_11;
        GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
        GPIO_InitStruct.Pull = GPIO_NOPULL;
        GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
        GPIO_InitStruct.Alternate = GPIO_AF6_SPI3;
        HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);
    }
    // deferred to spi_init function
}

void HAL_SPI_MspDeInit(SPI_HandleTypeDef* spiHandle)
{

    if (spiHandle->Instance == SPI3) {
        __HAL_RCC_SPI3_CLK_DISABLE();

        /**SPI3 GPIO Configuration
        PB2     ------> SPI3_MOSI
        PC10     ------> SPI3_SCK
        PC11     ------> SPI3_MISO
        */
        HAL_GPIO_DeInit(GPIOB, GPIO_PIN_2);

        HAL_GPIO_DeInit(GPIOC, GPIO_PIN_10 | GPIO_PIN_11);
    }
}

namespace hololink::emulation {
int spi_readback_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    SpiControllerCtxt* spi_ctxt = (SpiControllerCtxt*)ctxt;
    struct AddressValuePair* cur = addr_val;
    int n = 0;
    while (n < max_count) {
        uint32_t address = AVP_GET_ADDRESS(cur);
        if (address >= spi_ctxt->data_address && address < spi_ctxt->data_address + SPI_DATA_BUFFER_SIZE) {
            AVP_SET_VALUE(cur, spi_ctxt->data[(address - spi_ctxt->data_address) / REGISTER_SIZE]);
        } else if (spi_ctxt->control_address + SPI_REG_STATUS == address) {
            AVP_SET_VALUE(cur, spi_ctxt->status);
        } else if (address >= spi_ctxt->control_address && address <= spi_ctxt->control_address + SPI_REG_NUM_CMD_BYTES) {
            AVP_SET_VALUE(cur, spi_ctxt->registers[(address - spi_ctxt->control_address) / REGISTER_SIZE]);
        } else { // not an SPI address
            return n;
        }
        cur++;
        n++;
    }
    return n;
}

void spi_transaction(SpiControllerCtxt* spi_ctxt, uint32_t value)
{
    uint16_t cmd = value & 0xFFFF;
    if (!cmd && spi_ctxt->status == SPI_DONE) {
        spi_ctxt->status = SPI_IDLE;
    }
    if (cmd != SPI_START) { // 10B address not yet supported
        return;
    }
    spi_ctxt->status = SPI_BUSY;

    uint16_t num_bytes_write = spi_ctxt->registers[SPI_REG_NUM_BYTES / REGISTER_SIZE] & 0xFFFF;
    uint16_t num_bytes_read = (spi_ctxt->registers[SPI_REG_NUM_BYTES / REGISTER_SIZE] >> 16) & 0xFFFF;

#ifdef SPI_TEST_MODE
    // the first bit of the spi mode register (prescaler) is used to indicate a test since that baudrate does not exist on this target
    if (spi_ctxt->registers[SPI_REG_SPI_MODE / REGISTER_SIZE] & 1u) {
        // write dummy data to the data buffer for receipt
        for (uint16_t i = 0; i < num_bytes_read; i++) {
            spi_ctxt->data[i] = 0x01u << i;
        }
        // clear out remaining write buffer
        for (uint16_t i = num_bytes_read; i < num_bytes_write; i++) {
            spi_ctxt->data[i] = 0;
        }
        spi_ctxt->status = SPI_DONE;
        return;
    }
#endif

    if (num_bytes_write) {
        HAL_StatusTypeDef status = HAL_SPI_Transmit(&spi_ctxt->hspi, (uint8_t*)&spi_ctxt->data[0], num_bytes_write, SPI_TIMEOUT);
        if (status != HAL_OK) {
            spi_ctxt->status = SPI_SPI_ERR;
            return;
        }
    }
    if (num_bytes_read) {
        HAL_StatusTypeDef status = HAL_SPI_Receive(&spi_ctxt->hspi, (uint8_t*)&spi_ctxt->data[0], num_bytes_read, SPI_TIMEOUT);
        if (status != HAL_OK) {
            spi_ctxt->status = SPI_SPI_ERR;
            return;
        }
    }

    spi_ctxt->status = SPI_DONE;
}

int spi_set_mode(SpiControllerCtxt* spi_ctxt, uint32_t spi_mode)
{
    SPI_HandleTypeDef* hspi = &spi_ctxt->hspi;
    uint8_t data_size = (spi_mode >> SPI_DATA_SIZE_SHIFT) & SPI_DATA_SIZE_MASK;
    switch (data_size) {
    case 0:
        hspi->Init.DataSize = SPI_DATASIZE_4BIT;
        break;
    case 1:
        hspi->Init.DataSize = SPI_DATASIZE_8BIT;
        break;
    case 2:
        hspi->Init.DataSize = SPI_DATASIZE_16BIT;
        break;
    default:
        return -1;
    }
    hspi->Init.CLKPolarity = (spi_mode & SPI_CFG_CPOL) ? SPI_POLARITY_HIGH : SPI_POLARITY_LOW;
    hspi->Init.CLKPhase = (spi_mode & SPI_CFG_CPHA) ? SPI_PHASE_2EDGE : SPI_PHASE_1EDGE;
    hspi->Init.BaudRatePrescaler = (spi_mode & SPI_PRESCALER_MASK) << (SPI_CR1_BR_Pos - 1); //  see stm32f767xx.h which only has 3 bits and shifts 3 left
    return 0;
}

int reconfigure_spi(SpiControllerCtxt* spi_ctxt, uint32_t spi_mode)
{
    if (spi_mode == spi_ctxt->spi_mode) {
        return 0;
    }
    int status = spi_set_mode(spi_ctxt, spi_mode);
    if (status) {
        return status;
    }

    status = spi_init(&spi_ctxt->hspi);
    if (!status) {
        spi_ctxt->spi_mode = spi_mode;
    }
    return status;
}

// SPIController constructor placeholder
void spi_constructor(SpiControllerCtxt* spi_ctxt, uint32_t controller_address)
{
    SPI_HandleTypeDef* hspi = &spi_ctxt->hspi;
    hspi->Instance = SPI3;
    hspi->Init.Mode = SPI_MODE_MASTER;
    hspi->Init.Direction = SPI_DIRECTION_2LINES;
    hspi->Init.NSS = SPI_NSS_SOFT;
    hspi->Init.FirstBit = SPI_FIRSTBIT_MSB;
    hspi->Init.TIMode = SPI_TIMODE_DISABLE;
    hspi->Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
    hspi->Init.CRCPolynomial = 7;
    hspi->Init.CRCLength = SPI_CRC_LENGTH_DATASIZE;
    hspi->Init.NSSPMode = SPI_NSS_PULSE_ENABLE;
    // spi_ctxt->control_address should already be set in the static instance
    if (spi_ctxt->control_address != controller_address) {
        Error_Handler();
    }
    spi_ctxt->data_address = spi_ctxt->control_address + SPI_REG_DATA_BUFFER;
    spi_ctxt->status = SPI_IDLE;
    spi_ctxt->spi_mode = UINT32_MAX;
    memset(spi_ctxt->registers, 0, sizeof(spi_ctxt->registers));
    memset(spi_ctxt->data, 0, sizeof(spi_ctxt->data));
    spi_set_mode(spi_ctxt, SPI_DEFAULT_MODE);
}

int spi_configure_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count)
{
    SpiControllerCtxt* spi_ctxt = (SpiControllerCtxt*)ctxt;
    struct AddressValuePair* cur = addr_val;
    int n = 0;
    while (n < max_count) {
        uint32_t address = AVP_GET_ADDRESS(cur);
        if (address >= spi_ctxt->control_address && address <= spi_ctxt->control_address + SPI_REG_NUM_CMD_BYTES) {
            spi_ctxt->registers[(address - spi_ctxt->control_address) / REGISTER_SIZE] = AVP_GET_VALUE(cur);
            // if on the control address, execute the transaction command
            if (address == spi_ctxt->control_address) {
                spi_transaction(spi_ctxt, spi_ctxt->registers[0]);
            } else if (address == spi_ctxt->control_address + SPI_REG_SPI_MODE) {
                reconfigure_spi(spi_ctxt, spi_ctxt->registers[SPI_REG_SPI_MODE / REGISTER_SIZE]);
            }
        } else if (address >= spi_ctxt->data_address && address < spi_ctxt->data_address + SPI_DATA_BUFFER_SIZE) {
            spi_ctxt->data[(address - spi_ctxt->data_address) / REGISTER_SIZE] = AVP_GET_VALUE(cur);
        } else { // not an SPI address; cannot write to SPI_REG_STATUS
            return n;
        }
        cur++;
        n++;
    }
    return n;
}
}