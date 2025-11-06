/**
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
 *
 * See README.md for detailed information.
 */

#include <cstdint>

#define MAX_8BIT_COLOR 255
#define MAX_10BIT_COLOR 1023
#define MAX_12BIT_COLOR 4095

// statistical round and clamp to range
__device__ uint16_t bankers_round(float value, uint16_t max_value, uint16_t min_value) {
    if (value > max_value) {
        return max_value;
    }
    if (value < min_value) {
        return min_value;
    }
    uint16_t trim = (uint16_t)value;
    float diff = value - trim;
    if (diff > 0.5) {
        return trim + 1;
    } else if (diff < 0.5) {
        return trim;
    }
    return trim + (trim & 1);
}

// CUDA kernel
// convert 8-bit bayer pattern to 10-bit bayer pattern with T_X2Rc10Rb10Ra10 encoding (3 pixels per 4 bytes)
extern "C" __global__ void bayer8p_to_T_X2Rc10Rb10Ra10_kernel(uint8_t * dest, uint16_t line_bytes, uint8_t * src, uint16_t pixel_height, uint16_t pixel_width) {
    int32_t ix = 3 * (blockIdx.x * blockDim.x + threadIdx.x);
    int32_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= pixel_width || iy >= pixel_height) {
        return;
    }

    int32_t src_offset = iy * pixel_width + ix;
    int32_t dest_offset = iy * line_bytes + ix / 3 * 4;
    const float factor = MAX_10BIT_COLOR * 1.0f / MAX_8BIT_COLOR; // adjust for weighting based on byte size of each pixel

    uint16_t color = bankers_round(src[src_offset + 0] * factor, MAX_10BIT_COLOR, 0);
    dest[dest_offset] = color & 0xFF;
    dest[dest_offset + 1] = (color >> 8) & 0x03;
    if (ix + 1 < pixel_width) {
        color = bankers_round(src[src_offset + 1] * factor, MAX_10BIT_COLOR, 0);
        dest[dest_offset + 1] |= (color & 0x3F) << 2;
        dest[dest_offset + 2] = (color >> 6) & 0x0F;
    }
    if (ix + 2 < pixel_width) {
        color = bankers_round(src[src_offset + 2] * factor, MAX_10BIT_COLOR, 0);
        dest[dest_offset + 2] |= (color & 0xF) << 4;
        dest[dest_offset + 3] = (color >> 4) & 0x3F;
    }
}

// CUDA kernel
// convert 8-bit bayer pattern to 10-bit packed bayer pattern (4 pixels per 5 bytes)
extern "C" __global__ void bayer8p_to_10p_kernel(uint8_t * dest, uint16_t line_bytes, uint8_t * src, uint16_t pixel_height, uint16_t pixel_width) {
    int32_t ix = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    int32_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= pixel_width || iy >= pixel_height) {
        return;
    }

    int32_t src_offset = iy * pixel_width + ix;
    int32_t dest_offset = iy * line_bytes + ix / 4 * 5;
    const float factor = MAX_10BIT_COLOR * 1.0f / MAX_8BIT_COLOR; // adjust for weighting based on byte size of each pixel

    memset(&dest[dest_offset], 0, 5);
    
    uint16_t color = bankers_round(src[src_offset + 0] * factor, MAX_10BIT_COLOR, 0);
    dest[dest_offset] = (color >> 2) & 0xFF;
    dest[dest_offset + 4] = (color & 0x3) << 0;
    if (ix + 1 < pixel_width) {
        color = bankers_round(src[src_offset + 1] * factor, MAX_10BIT_COLOR, 0);
        dest[dest_offset + 1] = (color >> 2) & 0xFF;
        dest[dest_offset + 4] |= (color & 0x3) << 2;
    }
    if (ix + 2 < pixel_width) {
        color = bankers_round(src[src_offset + 2] * factor, MAX_10BIT_COLOR, 0);
        dest[dest_offset + 2] = (color >> 2) & 0xFF;
        dest[dest_offset + 4] |= (color & 0x3) << 4;
    }
    if (ix + 3 < pixel_width) {
        color = bankers_round(src[src_offset + 3] * factor, MAX_10BIT_COLOR, 0);
        dest[dest_offset + 3] = (color >> 2) & 0xFF;
        dest[dest_offset + 4] |= (color & 0x3) << 6;
    }
}

// CUDA kernel
// convert 8-bit bayer pattern to 12-bit bayer pattern with T_R12_PK_ISP encoding (2 pixels per 3 bytes)
extern "C" __global__ void bayer8p_to_T_R12_PK_ISP_kernel(uint8_t * dest, uint16_t line_bytes, uint8_t * src, uint16_t pixel_height, uint16_t pixel_width) {
    int32_t ix = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    int32_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= pixel_width || iy >= pixel_height) {
        return;
    }

    int32_t src_offset = iy * pixel_width + ix;
    int32_t dest_offset = iy * line_bytes + ix / 2 * 3;
    const float factor = MAX_12BIT_COLOR * 1.0f / MAX_8BIT_COLOR; // adjust for weighting based on byte size of each pixel

    uint16_t color = bankers_round(src[src_offset + 0] * factor, MAX_12BIT_COLOR, 0);
    dest[dest_offset] = color & 0xFF;
    dest[dest_offset + 1] = (color >> 8) & 0x0F;
    if (ix + 1 < pixel_width) {
        color = bankers_round(src[src_offset + 1] * factor, MAX_12BIT_COLOR, 0);
        dest[dest_offset + 1] |= (color & 0x0F) << 4;
        dest[dest_offset + 2] = (color >> 4) & 0xFF;
    }
}

// CUDA kernel
// generate 8-bit bayer pattern (GBRG)
extern "C" __global__ void generate_bayerGB8p_kernel(uint8_t * data, uint16_t pixel_height, uint16_t pixel_width)
{
    int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= pixel_height || col >= pixel_width) {
        return;
    }

    // assign the green value as the row
    // assign the red value as the column
    // assign the blue value as the reverse column
    
    if (row & 0x01u) { // RG of GBRG pattern
        if (col & 0x01u) { // green
            data[row * pixel_width + col] = bankers_round((row - 1) * 1.0f / pixel_height * MAX_8BIT_COLOR, MAX_8BIT_COLOR, 0);
        } else { // red
            data[row * pixel_width + col] = bankers_round(col * 1.0f / pixel_width * MAX_8BIT_COLOR, MAX_8BIT_COLOR, 0);
        }
    } else { // GB of GBRG pattern
        if (col & 0x01u) { // blue
            data[row * pixel_width + col] = bankers_round(1.0f *(pixel_width - col - 1) / pixel_width * MAX_8BIT_COLOR, MAX_8BIT_COLOR, 0);
        } else { // green
            data[row * pixel_width + col] = bankers_round(row * 1.0f / pixel_height * MAX_8BIT_COLOR, MAX_8BIT_COLOR, 0);
        }
    }
}
