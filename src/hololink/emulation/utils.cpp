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

#include <climits>
#include <cstdlib>
#include <cstring>

#include "utils.hpp"

// this is the effective number of elements covered by the underlying memory buffer, not necessarily the number of valid elements.
int64_t DL_n_elements(int32_t ndim, const int64_t* shape, const int64_t* strides)
{
    int64_t n_el = 1;
    if (strides) {
        for (int32_t i = 0; i < ndim; i++) {
            if (shape[i] > 1) {
                n_el += strides[i] * (shape[i] - 1);
            }
        }
    } else {
        for (int32_t i = 0; i < ndim; i++) {
            n_el *= shape[i];
        }
    }
    return n_el;
}
// calculate number of bytes required in the underlying memory buffer for a given type and dimensions of requested Tensor
int64_t DL_n_bytes(const DLDataType& dtype, int32_t ndim, const int64_t* shape, const int64_t* strides)
{
    if (!ndim) {
        return 0;
    }
    return DL_n_elements(ndim, shape, strides) * ((dtype.bits * (int64_t)dtype.lanes + (CHAR_BIT - 1)) / CHAR_BIT);
}

int64_t DLTensor_n_bytes(const DLTensor& tensor)
{
    return DL_n_bytes(tensor.dtype, tensor.ndim, tensor.shape, tensor.strides);
}
