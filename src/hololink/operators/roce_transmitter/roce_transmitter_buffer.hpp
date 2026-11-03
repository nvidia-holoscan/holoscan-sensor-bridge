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

#ifndef SRC_HOLOLINK_OPERATORS_ROCE_TRANSMITTER_ROCE_TRANSMITTER_BUFFER_HPP
#define SRC_HOLOLINK_OPERATORS_ROCE_TRANSMITTER_ROCE_TRANSMITTER_BUFFER_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace hololink::operators::detail {

struct DmaBufLayout {
    size_t registered_size;
    size_t allocation_size;
};

inline DmaBufLayout dma_buf_layout(size_t payload_size, size_t page_size)
{
    if (payload_size == 0) {
        throw std::invalid_argument("RoCE transmitter buffer size must be greater than zero");
    }
    if (page_size == 0) {
        throw std::invalid_argument("Host page size must be greater than zero");
    }
    if (payload_size > std::numeric_limits<size_t>::max() - (page_size - 1)) {
        throw std::overflow_error("RoCE transmitter buffer size is too large for page-aligned allocation");
    }

    const size_t registered_size = ((payload_size + page_size - 1) / page_size) * page_size;
    // Leave one additional page for aligning the returned CUDA pointer.
    if (registered_size > std::numeric_limits<size_t>::max() - page_size) {
        throw std::overflow_error("RoCE transmitter buffer size is too large for page-aligned allocation");
    }
    return { registered_size, registered_size + page_size };
}

inline char* page_aligned_pointer(char* ptr, size_t page_size)
{
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(ptr);
    const size_t remainder = address % page_size;
    if (remainder == 0) {
        return ptr;
    }
    return reinterpret_cast<char*>(address + page_size - remainder);
}

} // namespace hololink::operators::detail

#endif /* SRC_HOLOLINK_OPERATORS_ROCE_TRANSMITTER_ROCE_TRANSMITTER_BUFFER_HPP */
