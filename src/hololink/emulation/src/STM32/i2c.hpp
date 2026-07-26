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

#ifndef STM32_I2C_HPP
#define STM32_I2C_HPP

#include "../../hsb_emulator.hpp"
#include "STM32/stm32_system.h"

/**
 * @brief Initialize I2C.
 *
 * @param hi2c The I2C handle.
 *
 * @return The result of the initialization.
 */
int i2c_init(I2C_HandleTypeDef* hi2c);

namespace hololink::emulation {

void i2c_transaction(I2CControllerCtxt* i2c_ctxt, uint32_t value);

/**
 * @brief STM32-specific I2CControllerCtxt extension. `base` (the common I2CControllerCtxt)
 * is the first member; STM32-only state is the HAL handle.
 */
struct STM32I2CControllerCtxt {
    I2CControllerCtxt base;
    I2C_HandleTypeDef hi2c;
};
// if we need to add more controllers, we can add more instances of I2C_CONTROLLER_CTXT and use a map of their control addresses to differentiate

/**
 * @brief Callback function for I2C readback.
 *
 * @param ctxt The context pointer.
 * @param addr_val The address-value pair. Addresses are those written to the i2c controller registers at 0x30000200.
 * @param max_count The maximum number of address-value pairs to read.
 *
 * @return The number of address-value pairs read.
 */
int i2c_readback_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count);

/**
 * @brief Callback function for I2C configuration.
 *
 * @param ctxt The context pointer.
 * @param addr_val The address-value pair. Addresses are those written to the i2c controller registers at 0x30000200.
 * @param max_count The maximum number of address-value pairs to configure.
 *
 * @return The number of address-value pairs configured.
 */
int i2c_configure_cb(void* ctxt, struct AddressValuePair* addr_val, int max_count);

}
#endif