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

#include "dlpack/dlpack.h"
#include "hololink/core/serializer.hpp"

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
    uint64_t timestamp_s;
    uint32_t timestamp_ns;
    uint64_t bytes_written;
    uint32_t frame_number;
    // Time at which the metadata packet was sent
    uint64_t metadata_s;
    uint32_t metadata_ns;
    uint8_t reserved[80];
};

/**
 * This is metadata that is associated with all Transmitters that implement the abstract BaseTransmitter class.
 *
 * payload_size: Size of the payload in bytes. Semantics may be up to the type of transmitter.
 * e.g., pre-PAGE_SIZE calculation for LinuxTransmitter
 */
struct TransmissionMetadata {
    uint32_t dest_mac_low;
    uint32_t dest_mac_high;
    uint32_t dest_ip_address;
    uint32_t frame_size;
    uint32_t payload_size;
    uint16_t dest_port;
    uint16_t src_port;
};

typedef void* (*memcpy_func_t)(void* dst, const void* src, size_t n);

// returns 0 on failure or the number of bytes written on success.
// Note that on failure, serializer and buffer contents are in indeterminate state.
size_t serialize_frame_metadata(hololink::core::Serializer& serializer, FrameMetadata& frame_metadata);

/**
 * @brief Abstract base class for all transmitters
 *
 * This class is used to send DLPack tensors to the destination to interfacing with a variety of array memory models.
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
     * @brief Send a tensor to the destination
     *
     * @param metadata The metadata for the transmission. This is always aliased from the appropriate type of metadata for the Transmitter instance.
     * @param tensor The tensor to send. See dlpack.h for its contents and semantics.
     * @return The number of bytes sent or < 0 on error
     *
     * @note The tensor is not owned by the transmitter and must not be
     * propagated to other objects to satisfy the DLPack Python API specification.
     */
    virtual int64_t send(const TransmissionMetadata* metadata, const DLTensor& tensor) = 0;
};

} // namespace hololink::emulation

#endif // BASE_TRANSMITTER_HPP