/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 */

#ifndef COMPUTE_CRC_HPP
#define COMPUTE_CRC_HPP

#include <cstdint>

#define IS_CRC_MIRROR (1 << 0)

typedef enum {
    CRC_8bit = 8,
    CRC_16bit = 16,
    CRC_32bit = 32,
} CRC_TYPE;

typedef union {
    uint8_t crc_8bit;
    uint16_t crc_16bit;
    uint32_t crc_32bit;
} crc_output_t;

/* Structure to control the parameters of the algorithm */
typedef struct {
    CRC_TYPE type;
    union {
        uint8_t polynomial_crc8_bit;
        uint16_t polynomial_crc16_bit;
        uint32_t polynomial_crc32_bit;
    } polynomial;

    crc_output_t initial_crc;
    uint8_t crc_compute_flags;

} crc_parameters_t;

extern uint32_t const crc32_table[256];

crc_output_t compute_crc(crc_parameters_t *crc_parameters, uint8_t *data,
                         uint32_t data_len);

#endif // COMPUTE_CRC_HPP
