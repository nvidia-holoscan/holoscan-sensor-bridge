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

#include "base_transmitter.hpp"

namespace hololink::emulation {

// returns 0 on failure or the number of bytes written on success.
// Note that on failure, serializer and buffer contents are in indeterminate state.
size_t serialize_frame_metadata(hololink::core::Serializer& serializer, FrameMetadata& frame_metadata)
{
    size_t start = serializer.length();
    return serializer.append_uint32_be(frame_metadata.flags)
            && serializer.append_uint32_be(frame_metadata.psn)
            && serializer.append_uint32_be(frame_metadata.crc)
            && serializer.append_uint64_be(frame_metadata.timestamp_s)
            && serializer.append_uint32_be(frame_metadata.timestamp_ns)
            && serializer.append_uint64_be(frame_metadata.bytes_written)
            && serializer.append_uint32_be(frame_metadata.frame_number)
            && serializer.append_uint64_be(frame_metadata.metadata_s)
            && serializer.append_uint32_be(frame_metadata.metadata_ns)
            && serializer.append_buffer(frame_metadata.reserved, sizeof(frame_metadata.reserved))
        ? serializer.length() - start
        : 0;
}

} // namespace hololink::emulation
