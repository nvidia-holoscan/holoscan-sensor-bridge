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

#ifndef STM32_GPIO_HPP
#define STM32_GPIO_HPP

#include "STM32/hsb_config.hpp"

/*
USEFUL GPIO ADDRESSES for NUCLEO-F767ZI development board:

DESCRIPTION - PIN NUMBER - VALUE_MASK
LED_RED          0x10         0x00010000
LED_GREEN        0x17         0x00800000
LED_BLUE         0x1E         0x40000000
*/
#define LED_RED 0x00010000u
#define LED_GREEN 0x00800000u
#define LED_BLUE 0x40000000u
#define LED_RED_PIN GPIO_PIN_0
#define LED_GREEN_PIN GPIO_PIN_7
#define LED_BLUE_PIN GPIO_PIN_14
#define LED_RED_PORT GPIOB
#define LED_GREEN_PORT GPIOB
#define LED_BLUE_PORT GPIOB
#define GPIO_ERROR_PORT LED_RED_PORT
#define GPIO_ERROR_PIN LED_RED_PIN

#define GPIO_OUTPUT_BASE_REGISTER 0x0000000Cu
#define GPIO_DIRECTION_BASE_REGISTER 0x0000002Cu
#define GPIO_STATUS_BASE_REGISTER 0x0000008Cu
#define GPIO_NUM_REGISTERS 8u
#define GPIO_REGISTER_RANGE (GPIO_NUM_REGISTERS * REGISTER_SIZE)

#define GPIO_IN 1
#define GPIO_OUT 0
#define GPIO_LOW 0
#define GPIO_HIGH 1

/**
 * @brief Initialize GPIO.
 *
 * @param ctxt The context pointer.
 *
 * @return 0 on success, non-zero on failure
 */
int GPIO_init(void* ctxt);

namespace hololink::emulation {

/**
 * @brief Get the value of the GPIO.
 *
 * @param ctxt The context pointer.
 * @param addr_vals The address-value pair. Each bit in values corresponds to high/low state of the GPIO pin at the address.
 *                   Each address supports 32 pins with the 0 pin at GPIO_STATUS_BASE_REGISTER + 0 and the 31 pin at GPIO_STATUS_BASE_REGISTER + 31.
 * @param max_count The maximum number of address-value pairs to get.
 *
 * @return The number of address-value pairs got.
 */
int GPIO_get_value(void* ctxt, struct AddressValuePair* addr_vals, int max_count);

/**
 * @brief Get the direction of the GPIO.
 *
 * @param ctxt The context pointer.
 * @param addr_vals The address-value pair. Each bit in values corresponds to direction of the GPIO pin at the address.
 *                   Each address supports 32 pins with the 0 pin at GPIO_DIRECTION_BASE_REGISTER + 0 and the 31 pin at GPIO_DIRECTION_BASE_REGISTER + 31.
 * @param max_count The maximum number of address-value pairs to get.
 *
 * @return The number of address-value pairs got.
 */
int GPIO_get_direction(void* ctxt, struct AddressValuePair* addr_vals, int max_count);

/**
 * @brief Set the value of the GPIO.
 *
 * @param ctxt The context pointer.
 * @param addr_vals The address-value pair. Each bit in values corresponds to high/low state of the GPIO pin at the address.
 *                   Each address supports 32 pins with the 0 pin at GPIO_OUTPUT_BASE_REGISTER + 0 and the 31 pin at GPIO_OUTPUT_BASE_REGISTER + 31.
 * @param max_count The maximum number of address-value pairs to set.
 *
 * @return The number of address-value pairs set.
 */
int GPIO_set_value(void* ctxt, struct AddressValuePair* addr_vals, int max_count);

/**
 * @brief Set the direction of the GPIO.
 *
 * @param ctxt The context pointer.
 * @param addr_vals The address-value pair. Each bit in values corresponds to direction of the GPIO pin at the address.
 *                   Each address supports 32 pins with the 0 pin at GPIO_DIRECTION_BASE_REGISTER + 0 and the 31 pin at GPIO_DIRECTION_BASE_REGISTER + 31.
 * @param max_count The maximum number of address-value pairs to set.
 *
 * @return The number of address-value pairs set.
 */
int GPIO_set_direction(void* ctxt, struct AddressValuePair* addr_vals, int max_count);
}
#endif /* STM32_GPIO_HPP */
