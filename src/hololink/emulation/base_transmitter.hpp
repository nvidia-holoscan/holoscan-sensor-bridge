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

#ifndef BASE_TRANSMITTER_HPP
#define BASE_TRANSMITTER_HPP

#include <cstdint>

#include "../core/serializer.hpp"
#include "dlpack/dlpack.h"

// Note that if monotonic times are critical downstream (accuracy, repeatability or conflict with other system process), this clock must be configurable
#define FRAME_METADATA_CLOCK CLOCK_REALTIME
#define FRAME_METADATA_SIZE 128u

namespace hololink::emulation {

// data should all be in host byte order
struct FrameMetadata {
    uint32_t flags;
    uint32_t psn;
    uint32_t crc;
    // Time when the first sample data for the frame was received
    uint32_t timestamp_s_high;
    uint32_t timestamp_s_low;
    uint32_t timestamp_ns;
    uint32_t bytes_written_high;
    uint32_t bytes_written_low;
    uint32_t frame_number;
    // Time at which the metadata packet was sent
    uint32_t metadata_s_high;
    uint32_t metadata_s_low;
    uint32_t metadata_ns;
    uint8_t reserved[80];
};

static_assert(sizeof(FrameMetadata) == 128, "FrameMetadata size is not 128 bytes/there are padded misalignments");

extern struct FrameMetadata* DEFAULT_FRAME_METADATA;

// Forward declaration. The transmitter receives a pointer to the common DataPlaneCtxt
// (held by DataPlane::data_plane_ctxt_); each concrete transmitter downcasts to its
// transport-specific layout (COECtxt / RoCEv2Ctxt / UDPCtxt) since those structs
// embed the platform DataPlaneCtxt extension as their first member.
struct DataPlaneCtxt;

// returns 0 on failure or the number of bytes written on success.
// Note that on failure, serializer and buffer contents are in indeterminate state.
size_t serialize_frame_metadata(hololink::core::Serializer& serializer, FrameMetadata& frame_metadata);

/**
 * @brief Abstract base class for all transmitters
 *
 * Concrete transmitters (COETransmitter, RoCEv2Transmitter, UDPTransmitter) receive a
 * `DataPlaneCtxt*` and downcast to their transport-specific ctxt (COECtxt, RoCEv2Ctxt,
 * UDPCtxt) — those structs embed the platform DataPlaneCtxt extension as their first
 * member, so the pointer cast is sound.
 */
class BaseTransmitter {
public:
    /**
     * @brief Construct a new BaseTransmitter object
     */
    BaseTransmitter() = default;
    /**
     * @brief Destroy the BaseTransmitter object
     */
    virtual ~BaseTransmitter() { }

    /**
     * @brief Send a host buffer to the destination.
     * @param ctxt DataPlaneCtxt for the owning DataPlane. Concrete transmitter downcasts.
     * @param buffer Host-memory buffer.
     * @param buffer_size Bytes to send.
     * @param frame_metadata Optional frame-flush trigger; see DataPlane::send for semantics.
     * @return Bytes sent, or < 0 on error.
     */
    virtual int64_t send(DataPlaneCtxt* ctxt, const uint8_t* buffer, size_t buffer_size, FrameMetadata* frame_metadata = nullptr) = 0;
};

} // namespace hololink::emulation

#endif // BASE_TRANSMITTER_HPP