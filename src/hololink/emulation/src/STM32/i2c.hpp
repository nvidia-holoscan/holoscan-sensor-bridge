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

#define I2C_DATA_BUFFER_SIZE 0x100u

/**
 * @brief Initialize I2C.
 *
 * @param hi2c The I2C handle.
 *
 * @return The result of the initialization.
 */
int i2c_init(I2C_HandleTypeDef* hi2c);

namespace hololink::emulation {

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