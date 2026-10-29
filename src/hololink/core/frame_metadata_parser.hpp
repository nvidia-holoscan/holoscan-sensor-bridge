/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#ifndef SRC_HOLOLINK_CORE_FRAME_METADATA_PARSER_HPP
#define SRC_HOLOLINK_CORE_FRAME_METADATA_PARSER_HPP

#include <stdexcept>

#include <hololink/core/deserializer.hpp>
#include <hololink/core/hololink.hpp>

namespace hololink::core {

// Header-only so a data-plane plugin can decode the trailer without compiling
// hololink.cpp. static gives this function internal linkage. The ignored
// 16-bit word is the padding the wire format inserts before frame_number.
static inline hololink::Hololink::FrameMetadata parse_frame_metadata(
    const uint8_t* metadata_buffer, unsigned metadata_buffer_size)
{
    Deserializer deserializer(metadata_buffer, metadata_buffer_size);
    hololink::Hololink::FrameMetadata metadata {};
    uint16_t ignored = 0;
    if (!(deserializer.next_uint32_be(metadata.flags)
            && deserializer.next_uint32_be(metadata.psn)
            && deserializer.next_uint32_be(metadata.crc)
            && deserializer.next_uint64_be(metadata.timestamp_s)
            && deserializer.next_uint32_be(metadata.timestamp_ns)
            && deserializer.next_uint64_be(metadata.bytes_written)
            && deserializer.next_uint16_be(ignored)
            && deserializer.next_uint16_be(metadata.frame_number)
            && deserializer.next_uint64_be(metadata.metadata_s)
            && deserializer.next_uint32_be(metadata.metadata_ns))) {
        throw std::runtime_error("Buffer underflow in metadata");
    }
    return metadata;
}

} // namespace hololink::core

#endif /* SRC_HOLOLINK_CORE_FRAME_METADATA_PARSER_HPP */
