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

#ifndef STM32_TIM_H
#define STM32_TIM_H

#include "STM32/stm32_system.h"
#include <time.h>

// constants for time conversions
#define MSEC_PER_SEC 1000U
#define USEC_PER_SEC 1000000U
#define NSEC_PER_SEC 1000000000U
#define USEC_PER_MSEC MSEC_PER_SEC
#define NSEC_PER_MSEC USEC_PER_SEC
#define NSEC_PER_USEC MSEC_PER_SEC

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the timer.
 * @param ctxt The context to pass to the callback.
 * @return 0 on success, else error code.
 */
int tim_init(void * ctxt);

/**
 * @brief Initialize the timer.
 * @param htim The timer handle.
 * @param PreemptPriority The preempt priority.
 * @param SubPriority The subpriority.
 * @return -2 on invalid timer instance, -1 if timer already initialized, 0 on success
 */
int timer_init(TIM_HandleTypeDef* htim, uint32_t PreemptPriority, uint32_t SubPriority);

/**
 * @brief Start the timer.
 * @param htim The timer handle.
 * @return 0 on success, else error code.
 */
int timer_start(TIM_HandleTypeDef* htim);

/**
 * @brief Stop the timer.
 * @param htim The timer handle.
 * @return 0 on success, else error code.
 */
int timer_stop(TIM_HandleTypeDef* htim);

/**
 * @brief Deinitialize the timer.
 * @param htim The timer handle.
 * @return 0 on success, else error code.
 */
int timer_deinit(TIM_HandleTypeDef* htim);

/**
 * @brief Initialize the timer MSP.
 * @param tim_baseHandle The timer handle.
 */
void HAL_TIM_Base_MspInit(TIM_HandleTypeDef* tim_baseHandle);
void HAL_TIM_Base_MspDeInit(TIM_HandleTypeDef* tim_baseHandle);

/**
 * @brief Get the current time. Signature meant to match the corresponding posix function, but uses SysTick by default which means resolution is 1ms.
 * @param clock_id The clock ID. Note, only CLOCK_REALTIME is supported
 * @param tp The timespec to write the time to.
 * @return 0 on success, < 0 on error.
 */
int clock_gettime(clockid_t clock_id, struct timespec * tp);

// helper function to get the delta in milliseconds between two timespecs and writing to BootpPacket::secs (delta time must fit in 16 bits)
static inline uint32_t get_delta_msecs(const struct timespec* start_time, const struct timespec* end_time )
{
    time_t delta_sec = end_time->tv_sec - start_time->tv_sec;
    uint32_t delta_nsec = 0;
    if (start_time->tv_nsec > end_time->tv_nsec) {
        delta_sec--;
        delta_nsec = NSEC_PER_SEC + end_time->tv_nsec - start_time->tv_nsec;
    } else {
        delta_nsec = end_time->tv_nsec - start_time->tv_nsec;
    }
    return delta_sec * MSEC_PER_SEC + delta_nsec/NSEC_PER_MSEC;
}

#ifdef __cplusplus
}
#endif

#endif /* TIM_H */

