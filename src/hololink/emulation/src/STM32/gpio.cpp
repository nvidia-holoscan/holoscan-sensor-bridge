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
#include "STM32/hsb_config.hpp"
#include "STM32/stm32_system.h"
#include <stdint.h>

// WARNING: AHB1PERIPH_BASE and several constants are stm32f767 specific
#define GPIO_BANK_COUNT 11U
#define GPIO_PIN_PER_BANK 16U
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

    GPIO_InitTypeDef gpio_cfg = { 0 };

    /* GPIO Ports Clock Enable */
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_GPIOD_CLK_ENABLE();
    __HAL_RCC_GPIOE_CLK_ENABLE();
    __HAL_RCC_GPIOF_CLK_ENABLE();
    __HAL_RCC_GPIOG_CLK_ENABLE();
    //__HAL_RCC_GPIOH_CLK_ENABLE();

    /* reset LED pins */
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0 | GPIO_PIN_7 | GPIO_PIN_14, GPIO_PIN_RESET);

    /* reset USB power switch on pin */
    HAL_GPIO_WritePin(GPIOG, GPIO_PIN_6, GPIO_PIN_RESET);

    // Configure GPIO port A
    // analog input pins
    gpio_cfg.Pin = GPIO_PIN_0 | GPIO_PIN_3
        | GPIO_PIN_5 | GPIO_PIN_6
        | GPIO_PIN_15;
    gpio_cfg.Mode = GPIO_MODE_ANALOG;
    gpio_cfg.Pull = GPIO_NOPULL;
    gpio_cfg.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOA, &gpio_cfg);

    // Configure GPIO port B
    // analog input pins
    gpio_cfg.Pin = GPIO_PIN_1
        | GPIO_PIN_4 | GPIO_PIN_5
        | GPIO_PIN_8 | GPIO_PIN_10 | GPIO_PIN_11
        | GPIO_PIN_12 | GPIO_PIN_15;
    gpio_cfg.Mode = GPIO_MODE_ANALOG;
    gpio_cfg.Pull = GPIO_NOPULL;
    gpio_cfg.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOB, &gpio_cfg);
    // output (LED) pins
    gpio_cfg.Pin = GPIO_PIN_0
        | GPIO_PIN_7
        | GPIO_PIN_14;
    gpio_cfg.Mode = GPIO_MODE_OUTPUT_PP;
    gpio_cfg.Pull = GPIO_NOPULL;
    gpio_cfg.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOB, &gpio_cfg);

    // Configure GPIO port C
    gpio_cfg.Pin = GPIO_PIN_0 | GPIO_PIN_2 | GPIO_PIN_3
        | GPIO_PIN_6 | GPIO_PIN_7
        | GPIO_PIN_8 | GPIO_PIN_9
        | GPIO_PIN_12;
    gpio_cfg.Mode = GPIO_MODE_ANALOG;
    gpio_cfg.Pull = GPIO_NOPULL;
    gpio_cfg.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOC, &gpio_cfg);
    // User Button Pin (EXTI13)
    gpio_cfg.Pin = GPIO_PIN_13;
    gpio_cfg.Mode = GPIO_MODE_IT_RISING;
    gpio_cfg.Pull = GPIO_NOPULL;
    gpio_cfg.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOC, &gpio_cfg);

    HAL_NVIC_SetPriority(EXTI15_10_IRQn, 6, 0);
    HAL_NVIC_EnableIRQ(EXTI15_10_IRQn);

    // Configure GPIO port D
    // analog input pins
    gpio_cfg.Pin = GPIO_PIN_0 | GPIO_PIN_1 | GPIO_PIN_2 | GPIO_PIN_3
        | GPIO_PIN_4 | GPIO_PIN_5 | GPIO_PIN_6 | GPIO_PIN_7
        | GPIO_PIN_10 | GPIO_PIN_11
        | GPIO_PIN_12 | GPIO_PIN_13 | GPIO_PIN_14 | GPIO_PIN_15;
    gpio_cfg.Mode = GPIO_MODE_ANALOG;
    gpio_cfg.Pull = GPIO_NOPULL;
    gpio_cfg.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOD, &gpio_cfg);

    // Configure GPIO port E
    gpio_cfg.Pin = GPIO_PIN_0 | GPIO_PIN_1 | GPIO_PIN_2 | GPIO_PIN_3
        | GPIO_PIN_4 | GPIO_PIN_5 | GPIO_PIN_6 | GPIO_PIN_7
        | GPIO_PIN_8 | GPIO_PIN_9 | GPIO_PIN_10 | GPIO_PIN_11
        | GPIO_PIN_12 | GPIO_PIN_13 | GPIO_PIN_15;
    gpio_cfg.Mode = GPIO_MODE_ANALOG;
    gpio_cfg.Pull = GPIO_NOPULL;
    gpio_cfg.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOE, &gpio_cfg);
    // generic input, 0 when disconnected
    gpio_cfg.Pin = GPIO_PIN_14;
    gpio_cfg.Mode = GPIO_MODE_INPUT;
    gpio_cfg.Pull = GPIO_PULLDOWN;
    HAL_GPIO_Init(GPIOE, &gpio_cfg);

    // Configure GPIO port F
    gpio_cfg.Pin = GPIO_PIN_0 | GPIO_PIN_1 | GPIO_PIN_2 | GPIO_PIN_3
        | GPIO_PIN_4 | GPIO_PIN_5 | GPIO_PIN_6 | GPIO_PIN_7
        | GPIO_PIN_8 | GPIO_PIN_9 | GPIO_PIN_10 | GPIO_PIN_11
        | GPIO_PIN_12;
    gpio_cfg.Mode = GPIO_MODE_ANALOG;
    gpio_cfg.Pull = GPIO_NOPULL;
    gpio_cfg.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOF, &gpio_cfg);
    // it rising
    gpio_cfg.Pin = GPIO_PIN_15;
    gpio_cfg.Mode = GPIO_MODE_IT_RISING;
    gpio_cfg.Pull = GPIO_PULLDOWN;
    HAL_GPIO_Init(GPIOF, &gpio_cfg);
    // it falling
    gpio_cfg.Pin = GPIO_PIN_14;
    gpio_cfg.Mode = GPIO_MODE_IT_FALLING;
    gpio_cfg.Pull = GPIO_PULLDOWN;
    HAL_GPIO_Init(GPIOF, &gpio_cfg);
    // it rising & falling
    gpio_cfg.Pin = GPIO_PIN_12;
    gpio_cfg.Mode = GPIO_MODE_IT_RISING_FALLING;
    gpio_cfg.Pull = GPIO_PULLDOWN;
    HAL_GPIO_Init(GPIOF, &gpio_cfg);
    // output medium speed
    gpio_cfg.Pin = GPIO_PIN_13;
    gpio_cfg.Mode = GPIO_MODE_OUTPUT_PP;
    gpio_cfg.Pull = GPIO_PULLDOWN;
    gpio_cfg.Speed = GPIO_SPEED_FREQ_MEDIUM;
    HAL_GPIO_Init(GPIOF, &gpio_cfg);

    // configure GPIO bank G
    gpio_cfg.Pin = GPIO_PIN_0 | GPIO_PIN_1
        | GPIO_PIN_4 | GPIO_PIN_5
        | GPIO_PIN_8 | GPIO_PIN_9 | GPIO_PIN_10
        | GPIO_PIN_12 | GPIO_PIN_14 | GPIO_PIN_15;
    gpio_cfg.Mode = GPIO_MODE_ANALOG;
    gpio_cfg.Pull = GPIO_NOPULL;
    gpio_cfg.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOG, &gpio_cfg);
    // output (USB power switch on pin)
    gpio_cfg.Pin = GPIO_PIN_6;
    gpio_cfg.Mode = GPIO_MODE_OUTPUT_PP;
    gpio_cfg.Pull = GPIO_NOPULL;
    gpio_cfg.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOG, &gpio_cfg);
    // output high speed
    gpio_cfg.Pin = GPIO_PIN_2;
    gpio_cfg.Mode = GPIO_MODE_OUTPUT_PP;
    gpio_cfg.Pull = GPIO_PULLDOWN;
    gpio_cfg.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOG, &gpio_cfg);
    // output very high speed
    gpio_cfg.Pin = GPIO_PIN_3;
    gpio_cfg.Mode = GPIO_MODE_OUTPUT_PP;
    gpio_cfg.Pull = GPIO_PULLDOWN;
    gpio_cfg.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    HAL_GPIO_Init(GPIOG, &gpio_cfg);

    GPIO_initialized = true;
    return 0;
}

extern "C" void EXTI15_10_IRQHandler(void)
{
    HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_12);
    HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_13);
    HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_14);
    HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_15);
}

const uint32_t GPIO_SUPPORTED_PIN_NUM = (GPIO_BANK_COUNT * GPIO_PIN_PER_BANK);

static inline GPIO_TypeDef* get_gpio_address(uint32_t address, uint32_t base_address)
{
    uint32_t nregister = (address - base_address) / REGISTER_SIZE;
    if (nregister * 32U >= GPIO_SUPPORTED_PIN_NUM) {
        return NULL;
    }
    // WARNING: AHB1PERIPH_BASE is stm32f767 specific
    return (GPIO_TypeDef*)(AHB1PERIPH_BASE + 0x0400UL * nregister * 2); // * 2 because each 32-bit register contains 2 banks
}

static inline GPIO_TypeDef* get_next_bank(GPIO_TypeDef* gpio)
{
    uint32_t bank = (uint32_t)gpio + 0x400UL;
    if (bank >= AHB1PERIPH_BASE + 0x400UL * GPIO_BANK_COUNT) {
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
        for (uint16_t pin = 0; pin < GPIO_PIN_PER_BANK; pin++) {
            if (gpio_is_output(gpio, pin)) {
                HAL_GPIO_WritePin(gpio, pin_mask, (pin_values & pin_mask) ? GPIO_PIN_SET : GPIO_PIN_RESET);
            }
            pin_mask <<= 1U;
        }
        gpio = get_next_bank(gpio);
        if (gpio) {
            pin_mask = 1U;
            pin_values >>= 16U;
            for (uint16_t pin = 0; pin < GPIO_PIN_PER_BANK; pin++) {
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
        for (uint16_t pin = 0; pin < GPIO_PIN_PER_BANK; pin++) {
            if (!gpio_is_output(gpio, pin)) {
                AVP_SET_VALUE(addr_vals, (AVP_GET_VALUE(addr_vals) | (pin_mask << pin)));
            }
        }
        gpio = get_next_bank(gpio);
        if (gpio) {
            pin_mask = 1U;
            for (uint16_t pin = 0; pin < GPIO_PIN_PER_BANK; pin++) {
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
