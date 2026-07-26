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
 
#include "STM32/stm32_system.h"
#include "STM32/tim.h"
#include "board.h"
#include <stdint.h>

void Error_Handler(const char * str);

int clock_gettime(__attribute__((unused)) clockid_t clock_id, struct timespec * tp) {
    if (!tp || clock_id != CLOCK_REALTIME) {
        return -1;
    }
    static uint32_t last_tick = 0;
    static uint32_t offset_ctr = 0;
    uint32_t tick = HAL_GetTick();
    if (tick < last_tick) {
        offset_ctr++;
    }
    last_tick = tick;
    uint64_t total_ms = tick + offset_ctr * ((uint64_t) UINT32_MAX + 1);
    tp->tv_sec = (time_t)(total_ms / 1000);
    tp->tv_nsec = (long)((total_ms % 1000) * 1000000L);
    return 0;
}

_Bool tim_initialized = 0;

// initialize the module and timer
int tim_init(__attribute__((unused)) void * ctxt)
{
    if (tim_initialized) {
        return 0;
    }
    tim_initialized = 1;
    return 0;
}

_Bool timer_initialized[STM32_CONF_TIMER_COUNT] = {0};
_Bool timer_running[STM32_CONF_TIMER_COUNT] = {0};

int get_timer_index(TIM_TypeDef* instance) {
    for (int i = 0; i < STM32_CONF_TIMER_COUNT; i++) {
        if (stm32_conf_timer_map[i] == instance) {
            return i;
        }
    }
    return -1;
}

// returns -2 on invalid timer instance, -1 if timer already initialized, 0 on success
int timer_init(TIM_HandleTypeDef* htim, uint32_t PreemptPriority, uint32_t SubPriority) {
    int index = get_timer_index(htim->Instance);
    if (index == -1) {
        return -2;
    }
    if (timer_initialized[index]) {
        return -1;
    }
    
    HAL_TIM_Base_MspInit(htim);
    if (HAL_TIM_Base_Init(htim) != HAL_OK)
    {
        Error_Handler(NULL);
    }

    HAL_NVIC_SetPriority(stm32_conf_timer_irq[index], PreemptPriority, SubPriority);
    HAL_NVIC_EnableIRQ(stm32_conf_timer_irq[index]);

    timer_initialized[index] = 1;
    return 0;
}

int timer_start(TIM_HandleTypeDef* htim) {
    int index = get_timer_index(htim->Instance);
    if (index == -1) {
        return -1;
    }
    if (!timer_initialized[index]) {
        return -2;
    }
    if (HAL_TIM_Base_Start_IT(htim) != HAL_OK) {
        return -3;
    }
    timer_running[index] = 1;
    return 0;
}

int timer_stop(TIM_HandleTypeDef* htim) {
    int index = get_timer_index(htim->Instance);
    if (index == -1) {
        return -1;
    }
    if (!timer_running[index]) {
        return 0;
    }
    if (HAL_TIM_Base_Stop_IT(htim) != HAL_OK) {
        return -3;
    }
    timer_running[index] = 0;
    return 0;
}

int timer_deinit(TIM_HandleTypeDef* htim) {
    int index = get_timer_index(htim->Instance);
    if (index == -1) {
        return -1;
    }
    if (!timer_initialized[index]) {
        return -2;
    }

    if (timer_stop(htim)) {
        return -3;
    }
    HAL_NVIC_DisableIRQ(stm32_conf_timer_irq[index]);
    HAL_TIM_Base_DeInit(htim);
    HAL_TIM_Base_MspDeInit(htim);
    timer_initialized[index] = 0;
    return 0;
}

// initialize the timer MSP
void HAL_TIM_Base_MspInit(TIM_HandleTypeDef* tim_baseHandle) {
    int index = get_timer_index(tim_baseHandle->Instance);
    if (index == -1) {
        return;
    }
    stm32_conf_timer_clock_enable(index);
}

// de-initialize the timer MSP
void HAL_TIM_Base_MspDeInit(TIM_HandleTypeDef* tim_baseHandle) {
    int index = get_timer_index(tim_baseHandle->Instance);
    if (index == -1) {
        return;
    }
    stm32_conf_timer_clock_disable(index);
}

// Block the calling thread for `milliseconds` ms via the STM32 HAL SysTick-based
// delay. Declared in hsb_emulator.hpp (extern "C") so C++ callers see it; also
// in tim.h so peer C files in this directory can call it without pulling the
// C++ header in.
int msleep(unsigned milliseconds) {
    HAL_Delay(milliseconds);
    return 0;
}

