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

#ifndef STM32_SPI_HPP
#define STM32_SPI_HPP

#include "../../hsb_emulator.hpp"
#include "STM32/stm32_system.h"

#define SPI_DATA_BUFFER_SIZE 0x100u

// this will be made an Opaque struct when SPIController class is implemented
struct SpiControllerCtxt {
    SPI_HandleTypeDef hspi;
    uint32_t registers[(SPI_REG_NUM_CMD_BYTES - SPI_REG_CONTROL) / REGISTER_SIZE + 1];
    uint32_t data[SPI_DATA_BUFFER_SIZE / REGISTER_SIZE];
    uint32_t control_address;
    uint32_t data_address;
    uint32_t status;
    uint32_t spi_mode;
};
extern struct SpiControllerCtxt SPI_CONTROLLER_CTXT;

/* SPI3 init function. this can be called multiple times on the same object
 to change initialization, but will only initialize the clocks and GPIOs once*/
int spi_init(SPI_HandleTypeDef* hspi);

namespace hololink::emulation {

// SPIController constructor placeholder
void spi_constructor(SpiControllerCtxt* spi_ctxt, uint32_t controller_address);

/**
 * @brief Callback function for SPI readback.
 * @param ctxt The context pointer.
 * @param addr_val The address-value pair. Addresses are those written to the SPI controller registers at SPI_CTRL.
 * @param max_count The maximum number of address-value pairs to read.
 * @return The number of address-value pairs read.
 */
int spi_readback_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count);

/**
 * @brief Callback function for SPI configuration.
 * @param ctxt The context pointer.
 * @param addr_val The address-value pair. Addresses are those written to the SPI controller registers at SPI_CTRL.
 * @param max_count The maximum number of address-value pairs to configure.
 * @return The number of address-value pairs configured.
 */
int spi_configure_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count);
}

#endif /* STM32_SPI_HPP */
