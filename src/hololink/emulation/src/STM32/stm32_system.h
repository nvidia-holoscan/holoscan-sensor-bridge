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
 
#ifndef STM32_SYSTEM_H
#define STM32_SYSTEM_H

// The selected board's board.h pulls in the correct HAL header for its STM32 series.
#include "board.h"

#ifdef __cplusplus
extern "C" {
#endif

// system utilities

// returns a 96-bit UID as a non-null-terminated string of 12 characters from static storage
char * get_uid(void);

// interrupt handler declarations
void NMI_Handler(void);
void HardFault_Handler(void);
void MemManage_Handler(void);
void BusFault_Handler(void);
void UsageFault_Handler(void);
void SVC_Handler(void);
void DebugMon_Handler(void);
void PendSV_Handler(void);
void SysTick_Handler(void);

// system configuration function declarations
void SystemClock_Config(void);
void MPU_Config(void);
void HAL_MspInit(void);

#ifdef __cplusplus
}
#endif

#endif /* STM32_SYSTEM_H */