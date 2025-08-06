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

namespace hololink::emulation {

/**
 * This is metadata that is associated with all Transmitters that implement the abstract BaseTransmitter class.
 *
 * payload_size: Size of the payload in bytes. Semantics may be up to the type of transmitter.
 * e.g., pre-PAGE_SIZE calculation for LinuxTransmitter
 */
struct TransmissionMetadata {
    uint16_t payload_size;
};

/**
 * @brief Abstract base class for all transmitters
 *
 * This class is used to send DLPack tensors to the destination to interfacing with a variety of array memory models.
 */
class BaseTransmitter {
public:
    BaseTransmitter() = default;
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