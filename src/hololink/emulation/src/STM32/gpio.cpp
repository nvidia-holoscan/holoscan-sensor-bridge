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

#include "STM32/gpio.hpp"
#include "../../hsb_config.hpp"
#include "STM32/stm32_system.h"
#include "board.h"
#include <stdint.h>

// 32-bit register contains 2 banks
#define GPIO_BANK_PER_REGISTER 2U
#define GPIO_DIRECTION_MASK 0x03UL

typedef enum {
    GPIO_DIR_INPUT = 0x00,
    GPIO_DIR_OUTPUT = 0x01,
    GPIO_DIR_AF = 0x02,
    GPIO_DIR_ANALOG = 0x03
} GPIO_Direction;

bool GPIO_initialized = false;

int GPIO_init(__attribute__((unused)) void* ctxt)
{
    if (GPIO_initialized) {
        return 0;
    }
    stm32_conf_gpio_init_board();
    GPIO_initialized = true;
    return 0;
}

const uint32_t GPIO_SUPPORTED_PIN_NUM = (STM32_CONF_GPIO_BANK_COUNT * STM32_CONF_GPIO_PIN_PER_BANK);

static inline GPIO_TypeDef* get_gpio_address(uint32_t address, uint32_t base_address)
{
    uint32_t nregister = (address - base_address) / REGISTER_SIZE;
    if (nregister * 32U >= GPIO_SUPPORTED_PIN_NUM) {
        return NULL;
    }
    return (GPIO_TypeDef*)(STM32_CONF_GPIO_BANK_BASE + STM32_CONF_GPIO_BANK_STRIDE * nregister * 2); // * 2 because each 32-bit register contains 2 banks
}

static inline GPIO_TypeDef* get_next_bank(GPIO_TypeDef* gpio)
{
    uint32_t bank = (uint32_t)gpio + STM32_CONF_GPIO_BANK_STRIDE;
    if (bank >= STM32_CONF_GPIO_BANK_BASE + STM32_CONF_GPIO_BANK_STRIDE * STM32_CONF_GPIO_BANK_COUNT) {
        return NULL;
    }
    return (GPIO_TypeDef*)bank;
}

// pin_num is the sequential number of the pin in the GPIO bank...not the mask
static inline bool gpio_is_output(GPIO_TypeDef* gpio, uint16_t pin_num)
{
    // There is no HAL Driver API to directly check if a pin is output, so we have to use the MODER register
    return ((gpio->MODER >> (pin_num * 2U)) & GPIO_DIRECTION_MASK) == GPIO_DIR_OUTPUT;
}

// pin_num is the sequential number of the pin in the GPIO bank...not the mask
static inline bool gpio_is_input(GPIO_TypeDef* gpio, uint16_t pin_num)
{
    // There is no HAL Driver API to directly check if a pin is output, so we have to use the MODER register
    return ((gpio->MODER >> (pin_num * 2U)) & GPIO_DIRECTION_MASK) == GPIO_DIR_INPUT;
}

namespace hololink::emulation {

static inline uint32_t gpio_dir_changed(uint32_t address, uint32_t requested)
{
    // get the current status of the pins. This will be 1 if the pin is output, else 0.
    struct AddressValuePair addr_vals[] = {
        { address, 0 },
        { SENTINEL_ADDRESS, SENTINEL_VALUE }
    };
    GPIO_get_direction(NULL, addr_vals, 1);
    return AVP_GET_VALUE(addr_vals) ^ requested;
}

int GPIO_get_value(__attribute__((unused)) void* ctxt, AddressValuePair* addr_vals, int max_count)
{
    int i = 0;
    GPIO_TypeDef* gpio = get_gpio_address(AVP_GET_ADDRESS(addr_vals), GPIO_STATUS_BASE_REGISTER);
    while (i < max_count && gpio) {
        // bypass the HAL driver API here to directly read the pin status to avoid an unnecessary for loop
        // can directly use the value since GPIO_PIN_SET in HAL library is same as hololink GPIO_HIGH
        AVP_SET_VALUE(addr_vals, (uint32_t)(gpio->IDR & GPIO_PIN_All));
        gpio = get_next_bank(gpio);
        if (gpio) {
            AVP_SET_VALUE(addr_vals, (AVP_GET_VALUE(addr_vals) | (((uint32_t)(gpio->IDR & GPIO_PIN_All)) << 16U)));
        }
        i++;
        addr_vals++;
        if (i < max_count) {
            gpio = get_gpio_address(AVP_GET_ADDRESS(addr_vals), GPIO_STATUS_BASE_REGISTER);
        }
    }
    return i;
}

int GPIO_set_value(__attribute__((unused)) void* ctxt, AddressValuePair* addr_vals, int max_count)
{
    int i = 0;
    GPIO_TypeDef* gpio = get_gpio_address(AVP_GET_ADDRESS(addr_vals), GPIO_OUTPUT_BASE_REGISTER);
    while (i < max_count && gpio) {
        uint32_t pin_mask = 1U;
        uint32_t pin_values = AVP_GET_VALUE(addr_vals);
        for (uint16_t pin = 0; pin < STM32_CONF_GPIO_PIN_PER_BANK; pin++) {
            if (gpio_is_output(gpio, pin)) {
                HAL_GPIO_WritePin(gpio, pin_mask, (pin_values & pin_mask) ? GPIO_PIN_SET : GPIO_PIN_RESET);
            }
            pin_mask <<= 1U;
        }
        gpio = get_next_bank(gpio);
        if (gpio) {
            pin_mask = 1U;
            pin_values >>= 16U;
            for (uint16_t pin = 0; pin < STM32_CONF_GPIO_PIN_PER_BANK; pin++) {
                if (gpio_is_output(gpio, pin)) {
                    HAL_GPIO_WritePin(gpio, pin_mask, (pin_values & pin_mask) ? GPIO_PIN_SET : GPIO_PIN_RESET);
                }
                pin_mask <<= 1U;
            }
        }
        addr_vals++;
        i++;
        if (i < max_count) {
            gpio = get_gpio_address(AVP_GET_ADDRESS(addr_vals), GPIO_OUTPUT_BASE_REGISTER);
        }
    }
    return i;
}

int GPIO_get_direction(__attribute__((unused)) void* ctxt, AddressValuePair* addr_vals, int max_count)
{
    int i = 0;
    GPIO_TypeDef* gpio = get_gpio_address(AVP_GET_ADDRESS(addr_vals), GPIO_DIRECTION_BASE_REGISTER);
    while (i < max_count && gpio) {
        // treat everything not output as input so it cannot be toggled. TODO: WARNING: This is semantically different from the existing hololink API/IP
        uint32_t pin_mask = 1U;
        for (uint16_t pin = 0; pin < STM32_CONF_GPIO_PIN_PER_BANK; pin++) {
            if (!gpio_is_output(gpio, pin)) {
                AVP_SET_VALUE(addr_vals, (AVP_GET_VALUE(addr_vals) | (pin_mask << pin)));
            }
        }
        gpio = get_next_bank(gpio);
        if (gpio) {
            pin_mask = 1U;
            for (uint16_t pin = 0; pin < STM32_CONF_GPIO_PIN_PER_BANK; pin++) {
                if (!gpio_is_output(gpio, pin)) {
                    AVP_SET_VALUE(addr_vals, (AVP_GET_VALUE(addr_vals) | (pin_mask << (pin + 16U))));
                }
            }
        }
        addr_vals++;
        i++;
        if (i < max_count) {
            gpio = get_gpio_address(AVP_GET_ADDRESS(addr_vals), GPIO_DIRECTION_BASE_REGISTER);
        }
    }
    return i;
}

int GPIO_set_direction(__attribute__((unused)) void* ctxt, AddressValuePair* addr_vals, int max_count)
{
    int i = 0;
    GPIO_TypeDef* gpio = get_gpio_address(AVP_GET_ADDRESS(addr_vals), GPIO_DIRECTION_BASE_REGISTER);
    while (i < max_count && gpio) {
        uint32_t changed = gpio_dir_changed(AVP_GET_ADDRESS(addr_vals), AVP_GET_VALUE(addr_vals));
        {
            GPIO_InitTypeDef GPIO_InitStruct = { 0 };
            GPIO_InitStruct.Pin = changed & AVP_GET_VALUE(addr_vals) & GPIO_PIN_All; // the pins that are changed and set to input
            GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
            GPIO_InitStruct.Pull = GPIO_NOPULL;
            HAL_GPIO_Init(gpio, &GPIO_InitStruct);
        }

        {
            GPIO_InitTypeDef GPIO_InitStruct = { 0 };
            GPIO_InitStruct.Pin = changed & (~(AVP_GET_VALUE(addr_vals))) & GPIO_PIN_All; // the pins that are changed and set to output
            GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
            GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
            HAL_GPIO_Init(gpio, &GPIO_InitStruct);
        }

        gpio = get_next_bank(gpio);
        if (gpio) {
            {
                GPIO_InitTypeDef GPIO_InitStruct = { 0 };
                GPIO_InitStruct.Pin = ((changed & AVP_GET_VALUE(addr_vals)) >> 16) & GPIO_PIN_All; // the pins that are changed and set to input
                GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
                GPIO_InitStruct.Pull = GPIO_NOPULL;
                HAL_GPIO_Init(gpio, &GPIO_InitStruct);
            }

            {
                GPIO_InitTypeDef GPIO_InitStruct = { 0 };
                GPIO_InitStruct.Pin = ((changed & (~(AVP_GET_VALUE(addr_vals)))) >> 16) & GPIO_PIN_All; // the pins that are changed and set to output
                GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
                GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
                HAL_GPIO_Init(gpio, &GPIO_InitStruct);
            }
        }
        i++;
        addr_vals++;
        if (i < max_count) {
            gpio = get_gpio_address(AVP_GET_ADDRESS(addr_vals), GPIO_DIRECTION_BASE_REGISTER);
        }
    }
    return i;
}

}
