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

#include <cassert>
#include <cstdint>
#include <cstring>

#include "compute_crc.hpp"

//extern uint32_t const crc32_table;

/**
 * Returns the pointer to CRC table
 */
void *get_pointer_to_crctable(crc_parameters_t *crc_parameters) {
    /* Only CRC32 supported for now */
    assert(crc_parameters->type == CRC_32bit);

    if (crc_parameters->type == CRC_32bit)
        return (void *)crc32_table;

    return NULL;
}

unsigned char generate_mirror(unsigned char value) {
    unsigned char loopCount = (sizeof(unsigned char) * 8) - 1;
    unsigned char mirrorValue = value;

    value >>= 1;
    while (value) {
        mirrorValue <<= 1;
        mirrorValue |= value & 1;
        value >>= 1;
        loopCount--;
    }
    mirrorValue <<= loopCount;

    return mirrorValue;
}

crc_output_t compute_crc(crc_parameters_t *crc_parameters, uint8_t *data,
                         uint32_t data_len) {
    crc_output_t temp_value;
    uint32_t loop_count;
    void *tmp_ptr;

    memcpy(&temp_value, &crc_parameters->initial_crc, sizeof(temp_value));

    if (crc_parameters->type == CRC_32bit) {
        tmp_ptr = ((uint32_t *)get_pointer_to_crctable(crc_parameters));
        for (loop_count = 0; loop_count < data_len; loop_count++) {
            if (crc_parameters->crc_compute_flags & IS_CRC_MIRROR)
                temp_value.crc_32bit =
                    ((uint32_t *)
                         tmp_ptr)[generate_mirror(data[loop_count]) ^
                                  ((temp_value.crc_32bit >> 0x18) & 0xFF)] ^
                    (temp_value.crc_32bit << 8);
        }
    }

    return temp_value;
}
