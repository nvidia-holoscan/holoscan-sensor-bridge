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

// All board / MCU specifics for NUCLEO-F767ZI live here. To bring up another STM32 line,
// copy this directory, edit board.h, and re-implement this file's bodies for the
// new pinout and clock tree.

#include "STM32/stm32_system.h"
#include "board.h"
#include <stdint.h>

// Forward declaration so we can reference ETH_HANDLE without pulling in net.hpp.
extern ETH_HandleTypeDef* ETH_HANDLE;

void Error_Handler(const char * str)
{
    (void)str;
    __disable_irq(); // should disable interrupts except for systick
    while (1)
    {
        
    }
}

/* ============================================================
 * I2C MSP overrides (HAL weak symbols)
 * ============================================================ */
void HAL_I2C_MspInit(I2C_HandleTypeDef* hi2c)
{
    GPIO_InitTypeDef GPIO_InitStruct = { 0 };
    RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = { 0 };
    if (hi2c->Instance == STM32_CONF_I2C_INSTANCE) {
        PeriphClkInitStruct.PeriphClockSelection = STM32_CONF_I2C_PERIPHCLK;
        PeriphClkInitStruct.I2c1ClockSelection = STM32_CONF_I2C_CLKSOURCE;
        if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK) {
            Error_Handler(NULL);
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
    if (i2cHandle->Instance == STM32_CONF_I2C_INSTANCE) {
        __HAL_RCC_I2C1_CLK_DISABLE();
        HAL_GPIO_DeInit(GPIOB, GPIO_PIN_6);
        HAL_GPIO_DeInit(GPIOB, GPIO_PIN_9);
    }
}

/* ============================================================
 * SPI MSP overrides
 * ============================================================ */
void HAL_SPI_MspInit(SPI_HandleTypeDef* hspi)
{
    GPIO_InitTypeDef GPIO_InitStruct = { 0 };
    if (hspi->Instance == STM32_CONF_SPI_INSTANCE) {
        __HAL_RCC_SPI3_CLK_ENABLE();
        /**SPI3 GPIO Configuration
        PA15    ------> SPI3_NSS
        PB2     ------> SPI3_MOSI
        PC10    ------> SPI3_SCK
        PC11    ------> SPI3_MISO
        */

        GPIO_InitStruct.Pin = GPIO_PIN_15;
        GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
        GPIO_InitStruct.Pull = GPIO_NOPULL;
        GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
        GPIO_InitStruct.Alternate = GPIO_AF6_SPI3;
        HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

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
}

void HAL_SPI_MspDeInit(SPI_HandleTypeDef* spiHandle)
{
    if (spiHandle->Instance == STM32_CONF_SPI_INSTANCE) {
        __HAL_RCC_SPI3_CLK_DISABLE();

        HAL_GPIO_DeInit(GPIOA, GPIO_PIN_15);
        HAL_GPIO_DeInit(GPIOB, GPIO_PIN_2);
        HAL_GPIO_DeInit(GPIOC, GPIO_PIN_10 | GPIO_PIN_11);
    }
}

/* ============================================================
 * Ethernet MSP overrides and ISR
 * ============================================================ */
void HAL_ETH_MspInit(ETH_HandleTypeDef* ethHandle)
{
    GPIO_InitTypeDef gpio_cfg = { 0 };
    if (ethHandle->Instance == STM32_CONF_ETH_INSTANCE) {
        // enable ETH clock
        __HAL_RCC_ETH_CLK_ENABLE();

        // RMII MDC, RX Data 0, RX Data 1 pins
        gpio_cfg.Pin = GPIO_PIN_1 | GPIO_PIN_4 | GPIO_PIN_5;
        gpio_cfg.Mode = GPIO_MODE_AF_PP;
        gpio_cfg.Pull = GPIO_NOPULL;
        gpio_cfg.Speed = GPIO_SPEED_FREQ_HIGH;
        gpio_cfg.Alternate = GPIO_AF11_ETH;
        HAL_GPIO_Init(GPIOC, &gpio_cfg);

        // RMII REF CLK, MDIO, CRS DV pins
        gpio_cfg.Pin = GPIO_PIN_1 | GPIO_PIN_2 | GPIO_PIN_7;
        gpio_cfg.Mode = GPIO_MODE_AF_PP;
        gpio_cfg.Pull = GPIO_NOPULL;
        gpio_cfg.Speed = GPIO_SPEED_FREQ_HIGH;
        gpio_cfg.Alternate = GPIO_AF11_ETH;
        HAL_GPIO_Init(GPIOA, &gpio_cfg);

        // RMII TX Data 1 pin
        gpio_cfg.Pin = GPIO_PIN_13;
        gpio_cfg.Mode = GPIO_MODE_AF_PP;
        gpio_cfg.Pull = GPIO_NOPULL;
        gpio_cfg.Speed = GPIO_SPEED_FREQ_HIGH;
        gpio_cfg.Alternate = GPIO_AF11_ETH;
        HAL_GPIO_Init(GPIOB, &gpio_cfg);

        // RMII TX Enable, TX Data 0 pins
        gpio_cfg.Pin = GPIO_PIN_11 | GPIO_PIN_13;
        gpio_cfg.Mode = GPIO_MODE_AF_PP;
        gpio_cfg.Pull = GPIO_NOPULL;
        gpio_cfg.Speed = GPIO_SPEED_FREQ_HIGH;
        gpio_cfg.Alternate = GPIO_AF11_ETH;
        HAL_GPIO_Init(GPIOG, &gpio_cfg);
    }
}

void HAL_ETH_MspDeInit(ETH_HandleTypeDef* ethHandle)
{
    if (ethHandle->Instance == STM32_CONF_ETH_INSTANCE) {
        ETH_HANDLE = NULL;

        // disable ETH clock
        __HAL_RCC_ETH_CLK_DISABLE();

        // de-init RMII pins
        HAL_GPIO_DeInit(GPIOC, GPIO_PIN_1 | GPIO_PIN_4 | GPIO_PIN_5);
        HAL_GPIO_DeInit(GPIOA, GPIO_PIN_1 | GPIO_PIN_2 | GPIO_PIN_7);
        HAL_GPIO_DeInit(GPIOB, GPIO_PIN_13);
        HAL_GPIO_DeInit(GPIOG, GPIO_PIN_11 | GPIO_PIN_13);
    }
}

void ETH_IRQHandler(void)
{
    HAL_ETH_IRQHandler(ETH_HANDLE);
}

/* ============================================================
 * EXTI 10..15 ISR (pin list is NUCLEO-F767ZI specific)
 * ============================================================ */
void EXTI15_10_IRQHandler(void)
{
    HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_12);
    HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_13);
    HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_14);
    HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_15);
}

/* ============================================================
 * Timer instance / IRQ tables and clock enable/disable
 * ============================================================ */
TIM_TypeDef* const stm32_conf_timer_map[STM32_CONF_TIMER_COUNT] = {
    TIM2, TIM3, TIM4, TIM5, TIM6, TIM7
};

const IRQn_Type stm32_conf_timer_irq[STM32_CONF_TIMER_COUNT] = {
    TIM2_IRQn, TIM3_IRQn, TIM4_IRQn, TIM5_IRQn, TIM6_DAC_IRQn, TIM7_IRQn
};

void stm32_conf_timer_clock_enable(int index)
{
    switch (index) {
    case 0: __HAL_RCC_TIM2_CLK_ENABLE(); break;
    case 1: __HAL_RCC_TIM3_CLK_ENABLE(); break;
    case 2: __HAL_RCC_TIM4_CLK_ENABLE(); break;
    case 3: __HAL_RCC_TIM5_CLK_ENABLE(); break;
    case 4: __HAL_RCC_TIM6_CLK_ENABLE(); break;
    case 5: __HAL_RCC_TIM7_CLK_ENABLE(); break;
    default: return;
    }
}

void stm32_conf_timer_clock_disable(int index)
{
    switch (index) {
    case 0: __HAL_RCC_TIM2_CLK_DISABLE(); break;
    case 1: __HAL_RCC_TIM3_CLK_DISABLE(); break;
    case 2: __HAL_RCC_TIM4_CLK_DISABLE(); break;
    case 3: __HAL_RCC_TIM5_CLK_DISABLE(); break;
    case 4: __HAL_RCC_TIM6_CLK_DISABLE(); break;
    case 5: __HAL_RCC_TIM7_CLK_DISABLE(); break;
    default: return;
    }
}

/* ============================================================
 * GPIO board init - enumerates every pin and sets its mode for NUCLEO-F767ZI.
 * ============================================================ */
void stm32_conf_gpio_init_board(void)
{
    GPIO_InitTypeDef gpio_cfg = { 0 };

    /* GPIO Ports Clock Enable */
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_GPIOD_CLK_ENABLE();
    __HAL_RCC_GPIOE_CLK_ENABLE();
    __HAL_RCC_GPIOF_CLK_ENABLE();
    __HAL_RCC_GPIOG_CLK_ENABLE();
    //__HAL_RCC_GPIOH_CLK_ENABLE(); // currently unused

    /* reset LED pins */
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0 | GPIO_PIN_7 | GPIO_PIN_14, GPIO_PIN_RESET);

    /* reset USB power switch on pin */
    HAL_GPIO_WritePin(GPIOG, GPIO_PIN_6, GPIO_PIN_RESET);

    // Configure GPIO port A
    // analog input pins
    gpio_cfg.Pin = GPIO_PIN_0 | GPIO_PIN_3
        | GPIO_PIN_6;
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
    gpio_cfg.Pin = GPIO_PIN_1
        | GPIO_PIN_4 | GPIO_PIN_5 | GPIO_PIN_6
        | GPIO_PIN_8 | GPIO_PIN_9 | GPIO_PIN_10
        | GPIO_PIN_12 | GPIO_PIN_14 | GPIO_PIN_15;
    gpio_cfg.Mode = GPIO_MODE_ANALOG;
    gpio_cfg.Pull = GPIO_NOPULL;
    gpio_cfg.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOG, &gpio_cfg);
    // output (USB power switch on pin)
    gpio_cfg.Pin = GPIO_PIN_0;
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
}

/* ============================================================
 * System clock / MPU configuration (called by name from startup)
 * ============================================================ */
void SystemClock_Config(void)
{
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

    // configure the power bkup
    HAL_PWR_EnableBkUpAccess();

    // configure internal regulator
    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

    // initialize the RCC oscillators
    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
    RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS;
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
    RCC_OscInitStruct.PLL.PLLM = 4;
    RCC_OscInitStruct.PLL.PLLN = 96;
    RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
    RCC_OscInitStruct.PLL.PLLQ = 4;
    RCC_OscInitStruct.PLL.PLLR = 2;
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
    {
        Error_Handler(NULL);
    }

    // activate the over-drive mode
    if (HAL_PWREx_EnableOverDrive() != HAL_OK)
    {
        Error_Handler(NULL);
    }

    // initialize the CPU, AHB and APB buses clocks
    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                                |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3) != HAL_OK)
    {
        Error_Handler(NULL);
    }
}

void MPU_Config(void)
{
    MPU_Region_InitTypeDef MPU_InitStruct = {0};

    HAL_MPU_Disable();

    MPU_InitStruct.Enable = MPU_REGION_ENABLE;
    MPU_InitStruct.Number = MPU_REGION_NUMBER0;
    MPU_InitStruct.BaseAddress = 0x0;
    MPU_InitStruct.Size = MPU_REGION_SIZE_4GB;
    MPU_InitStruct.SubRegionDisable = 0x87;
    MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL0;
    MPU_InitStruct.AccessPermission = MPU_REGION_NO_ACCESS;
    MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
    MPU_InitStruct.IsShareable = MPU_ACCESS_SHAREABLE;
    MPU_InitStruct.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
    MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;
    HAL_MPU_ConfigRegion(&MPU_InitStruct);

    HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);
}

void HAL_MspInit(void)
{
    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_RCC_SYSCFG_CLK_ENABLE();
}
