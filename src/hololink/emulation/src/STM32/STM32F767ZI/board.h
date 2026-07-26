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

#ifndef BOARD_H
#define BOARD_H

// Board / MCU specifics for NUCLEO-F767ZI (STM32F767ZI). Generic source files under
// src/STM32/ include this header to pick up peripheral instances, pin counts, timing
// values, and prototypes of the board-init hooks defined in board.c. To add another
// STM32 line, copy this folder, edit the macros below to match the new board, and
// select it with cmake -DHSB_EMULATOR_TARGET=<NEW_LINE>.

#include "stm32f7xx_hal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Peripheral instances (used by src/STM32/{i2c,spi,net,dac}.cpp)
 * ============================================================ */
#define STM32_CONF_I2C_INSTANCE        I2C1
#define STM32_CONF_I2C_TIMING          0x20303E5D
#define STM32_CONF_I2C_PERIPHCLK       RCC_PERIPHCLK_I2C1
#define STM32_CONF_I2C_CLKSOURCE       RCC_I2C1CLKSOURCE_PCLK1

#define STM32_CONF_SPI_INSTANCE        SPI3

#define STM32_CONF_ETH_INSTANCE        ETH

#define STM32_CONF_DAC_INSTANCE        DAC

/* ============================================================
 * GPIO topology (used by src/STM32/gpio.cpp)
 * ============================================================ */
#define STM32_CONF_GPIO_BANK_COUNT     11U
#define STM32_CONF_GPIO_PIN_PER_BANK   16U
#define STM32_CONF_GPIO_BANK_BASE      AHB1PERIPH_BASE
#define STM32_CONF_GPIO_BANK_STRIDE    0x400UL

/* ============================================================
 * NUCLEO-F767ZI on-board LED pins (legacy convenience macros)
 * ============================================================ */
#define LED_RED                        0x00010000u
#define LED_GREEN                      0x00800000u
#define LED_BLUE                       0x40000000u
#define LED_RED_PIN                    GPIO_PIN_0
#define LED_GREEN_PIN                  GPIO_PIN_7
#define LED_BLUE_PIN                   GPIO_PIN_14
#define LED_RED_PORT                   GPIOB
#define LED_GREEN_PORT                 GPIOB
#define LED_BLUE_PORT                  GPIOB
#define GPIO_ERROR_PORT                LED_RED_PORT
#define GPIO_ERROR_PIN                 LED_RED_PIN

/* ============================================================
 * Timer instances and IRQs (used by src/STM32/tim.c)
 * Defined in board.c; declared here so generic code can iterate.
 * ============================================================ */
#define STM32_CONF_TIMER_COUNT         6
extern TIM_TypeDef* const stm32_conf_timer_map[STM32_CONF_TIMER_COUNT];
extern const IRQn_Type   stm32_conf_timer_irq[STM32_CONF_TIMER_COUNT];
void stm32_conf_timer_clock_enable(int index);
void stm32_conf_timer_clock_disable(int index);

/* ============================================================
 * Board init hooks (defined in board.c).
 * Called by the corresponding generic source files in src/STM32/.
 * ============================================================ */
// Pin enumeration + per-pin HAL_GPIO_Init for the entire NUCLEO-F767ZI board.
// Called from GPIO_init() in gpio.cpp once at startup.
void stm32_conf_gpio_init_board(void);

#ifdef __cplusplus
}
#endif

#endif /* BOARD_H */
