/*
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
 */

#ifndef SRC_HOLOLINK_OPERATORS_IQ_DEC_IQ_DEC_KERNELS_HPP
#define SRC_HOLOLINK_OPERATORS_IQ_DEC_IQ_DEC_KERNELS_HPP

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

namespace hololink::operators {

// Decode count IQ samples from the int16 buffer layout into interleaved floats, on stream.
void cuda_iq_decode(float* iq_component, const int16_t* encoded_input, size_t count, float scale,
    cudaStream_t stream);

} // namespace hololink::operators

#endif /* SRC_HOLOLINK_OPERATORS_IQ_DEC_IQ_DEC_KERNELS_HPP */
