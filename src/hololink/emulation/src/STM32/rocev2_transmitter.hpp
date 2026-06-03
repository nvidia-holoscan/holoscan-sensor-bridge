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

#ifndef EMULATION_STM32_ROCEV2_TRANSMITTER_HPP
#define EMULATION_STM32_ROCEV2_TRANSMITTER_HPP

#include <cstdint>
#include <cstring>

#include "dlpack/dlpack.h"

#include "STM32/hsb_config.hpp"
#include "STM32/net.hpp"
#include "base_transmitter.hpp"

#ifndef MAX_TRANSMITTERS
#define MAX_TRANSMITTERS MAX_DATA_PLANES
#endif

// offset from the start of the packet buffer to the start of the data buffer to ensure 64-bit alignment of headers
#define PACKET_OFFSET_RESET 2u

namespace hololink::emulation {

/**
 * Metadata for a transmission that is specific to the RoCEv2Transmitter.
 *
 * This contains the data for RoCEv2 that is liable to change every frame of data that is sent
 */
struct RoCEv2TransmissionMetadata {
    union {
        uint32_t vp_data[DP_HOST_UDP_PORT / REGISTER_SIZE + 1];
        struct TransmissionMetadata transmission_metadata;
    } base;
    uint32_t hif_data[DP_VP_MASK / REGISTER_SIZE + 1];
    // this is 64-bit aligned
    uint64_t address;
    ETH_HandleTypeDef* eth_handle;
    uint32_t metadata_offset;
    uint32_t payload_size;
    uint32_t page;
    // frame_metadata also has these values, but they are in network byte order. Use these for actual tracking
    uint32_t frame_number;
    uint32_t psn;
    uint32_t vp_address;
    uint32_t hif_address;

    // frame_metadata, packet, and packet_offset are possibly not in all implementations
    // FrameMetadata should be 64-bit aligned
    FrameMetadata frame_metadata;
    // should already be 64-bit aligned
    alignas(uint64_t) uint8_t packet[TX_BUFFER_SIZE + PACKET_OFFSET_RESET];
    uint16_t packet_offset;

    // final due to alignment requirements
    bool in_use;
};

#define ROCEV2_HDR_LEN (ETHER_HDR_LEN + IP_HDR_LEN + UDP_HDR_LEN + BT_HDR_LEN + RET_HDR_LEN)
#define RoCEv2TransmissionMetadata_get_buffer_base(metadatap) (&((metadatap)->packet[PACKET_OFFSET_RESET]))
#define RoCEv2TransmissionMetadata_get_buffer(metadatap) (&((metadatap)->packet[(metadatap)->packet_offset]))
#define RoCEv2TransmissionMetadata_get_buffer_size(metadatap) ((metadatap)->packet_offset - PACKET_OFFSET_RESET)
#define RoCEv2TransmissionMetadata_buffer_clear(metadatap) ((metadatap)->packet_offset = PACKET_OFFSET_RESET + ROCEV2_HDR_LEN)
#define RoCEv2TransmissionMetadata_get_max_size(metadatap) (sizeof(metadatap->packet) - PACKET_OFFSET_RESET)

/**
 * @brief The RoCEv2Transmitter implements the BaseTransmitter interface and encapsulates the transport over RoCEv2
 */
class RoCEv2Transmitter : public BaseTransmitter {
public:
    /**
     * @brief Send a tensor to the destination using the TransmissionMetadata provided. Implementation of BaseTransmitter::send interface method.
     * @param metadata The metadata for the transmission. This is always aliased from the appropriate type of metadata for the Transmitter instance.
     * @param tensor The tensor to send. See dlpack.h for its contents and semantics.
     * @return The number of bytes sent or < 0 on error
     */
    int64_t send(TransmissionMetadata* metadata, const DLTensor& tensor, FrameMetadata* frame_metadata = nullptr) override;
    /**
     * @brief Send a buffer to the destination using the TransmissionMetadata provided. Implementation of BaseTransmitter::send interface method.
     * @param metadata The metadata for the transmission. This is always aliased from the appropriate type of metadata for the Transmitter instance.
     * @param buffer The buffer to send.
     * @param buffer_size The size of the buffer.
     * @param frame_metadata The frame metadata to send. Acts as a buffer flush. If nullptr, the data is by default buffered until the next send command that fills an MTU or a non-nullptr frame_metadata is provided.
     * @return The number of bytes sent or < 0 if error occurred.
     */
    int64_t send(TransmissionMetadata* metadata, const uint8_t* buffer, size_t buffer_size, FrameMetadata* frame_metadata = nullptr) override;

    void init_metadata(RoCEv2TransmissionMetadata* metadata, const IPAddress& source_ip);

private:
    // double buffering is for GPU inputs currently
};

} // namespace hololink::emulation

#endif
