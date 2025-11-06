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

#ifndef EMULATION_TRANSMITTER_HPP
#define EMULATION_TRANSMITTER_HPP

#include <cstdint>
#include <cstring>

#include "dlpack/dlpack.h"

#include "base_transmitter.hpp"
#include "net.hpp"

namespace hololink::emulation {

// from Infiniband specification
struct BTHeader {
    uint8_t opcode; // opcode {= 0x2A for write, 0x2B for write immediate}
    uint8_t flags; // flags {= 0}
    uint16_t p_key; // partition {= 0xFFFF}
    uint32_t qp; // qp {= 0xFF << 24 | qp}
    uint32_t psn; // psn only lower 24 bits are used
};

// from Infiniband specification
struct RETHeader {
    uint64_t vaddress; // virtual address where data is stored on receiver side
    uint32_t rkey; // rkey
    uint32_t content_size; // size of data payload not including any headers or CRC
};

// structure for storing header information
// there is a global instance that has defaults for the general LinuxTransmitter implementation
// and an instance copy that is used for instance-local values filled in on construction but not
// shared between instances - mostly the source port and ip address
// all data is assumed to be in host byte order in this structure. Serialization will handle conversion to network byte order
// see DEFAULT_LINUX_HEADERS in linux_transmitter.cpp for default values and comments on when elements are expected to be updated
struct LinuxHeaders {
    iphdr ip_h;
    udphdr udp_h;
    BTHeader bt_h;
    RETHeader ret_h;
};
// reference values and magic numbers for headers.
// explanations of which fields are overwritten and when
extern const LinuxHeaders DEFAULT_LINUX_HEADERS;

/**
 * Metadata for a transmission that is specific to the LinuxTransmitter.
 *
 * This contains the data for RoCEv2 that is liable to change every frame of data that is sent
 */
struct LinuxTransmissionMetadata {
    struct TransmissionMetadata transmission_metadata;
    uint64_t address;
    uint32_t qp;
    uint32_t rkey;
    uint32_t page;
    uint32_t metadata_offset; // host side sends this to HSB to indicate where the metadata is in
                              // the frame, but it's unclear this is really necessary. payload_size
                              // and page size have enough information to define this, I think.
    uint32_t page_mask;
};

/**
 * @brief The LinuxTransmitter implements the BaseTransmitter interface and encapsulates the transport over RoCEv2
 */
class LinuxTransmitter : public BaseTransmitter {
public:
    /**
     * @brief Construct a new LinuxTransmitter object
     * @param source_ip The IP address to be used as the source address of the transmitter.
     * @note The MAC address is derived from the interface name using the mac_from_if function in net.hpp.
     */
    LinuxTransmitter(const IPAddress& source_ip);
    /**
     * @brief Construct a new LinuxTransmitter object
     * @param headers The fully configurable headers to use. See source code for details
     */
    LinuxTransmitter(const LinuxHeaders& headers);
    ~LinuxTransmitter();

    /**
     * @brief Send a tensor to the destination using the TransmissionMetadata provided. Implementation of BaseTransmitter::send interface method.
     * @param metadata The metadata for the transmission. This is always aliased from the appropriate type of metadata for the Transmitter instance.
     * @param tensor The tensor to send. See dlpack.h for its contents and semantics.
     * @return The number of bytes sent or < 0 on error
     */
    int64_t send(const TransmissionMetadata* metadata, const DLTensor& tensor) override;

private:
    void init_socket();
    int data_socket_fd_ { -1 };

    // state data that is not thread safe without the DataPlane::send level synchronization
    LinuxHeaders linux_headers_;
    uint32_t psn_ { 0 };
    uint32_t frame_number_ { 0 };
    // double buffering is for GPU inputs currently
    void* double_buffer_ { nullptr };
    int64_t double_buffer_size_ { 0 };
};

} // namespace hololink::emulation

#endif
