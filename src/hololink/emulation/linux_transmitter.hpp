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

// IP header from RFC 791
struct IPHeader {
    uint8_t version_and_header_length; // 4 bits version, 4 bits header length
    uint8_t type_of_service;
    uint16_t length;
    uint16_t identification;
    uint16_t flags_and_fragment_offset;
    uint8_t time_to_live;
    uint8_t protocol;
    uint16_t checksum;
    uint32_t source_ip_address;
    uint32_t destination_ip_address;
    // uint32_t options; // only in upper 24 bits not present in datagrams
};

// UDP header from RFC 768
struct UDPHeader {
    uint16_t source_port;
    uint16_t destination_port;
    uint16_t length; // length in bytes of header + payload
    uint16_t checksum;
};

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
    IPHeader ip_h;
    UDPHeader udp_h;
    BTHeader bt_h;
    RETHeader ret_h;
};

// reference values and magic numbers for headers.
// explanations of which fields are overwritten and when
const struct LinuxHeaders DEFAULT_LINUX_HEADERS {
    .ip_h = {
        .version_and_header_length = 0x45, // IPv4
        .type_of_service = 0,
        .length = 0, // filled in on send based on payload size
        .identification = 0, // not used without fragmentation
        .flags_and_fragment_offset = 0x4000, // not used without fragmentation
        .time_to_live = DEFAULT_TTL, // not used without fragmentation
        .protocol = 17, // UDP
        .checksum = 0,
        .source_ip_address = 0, // optionally written at construction
        .destination_ip_address = 0, // filled in on send
    },
    .udp_h = {
        .source_port = 0, // optionally written at construction
        .destination_port = 0, // filled in on send from LinuxTransmitterMetadata
        .length = 0, // filled in on send based on payload size
        .checksum = 0,
    },
    .bt_h = {
        .opcode = 0, // filled in on send
        .flags = 0,
        .p_key = 0xFFFF,
        .qp = 0, // filled in on send from LinuxTransmitterMetadata
        .psn = 0, // filled in on send based on payload size
    },
    .ret_h = {
        .vaddress = 0, // filled in on send from LinuxTransmitterMetadata
        .rkey = 0, // filled in on send from LinuxTransmitterMetadata
        .content_size = 0, // filled in on send based on payload size
    },
};

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
    uint32_t dest_ip_address;
    uint16_t dest_port;
};

// RoCEv2 transmitter without using ib verbs apis
class LinuxTransmitter : public BaseTransmitter {
public:
    LinuxTransmitter(const std::string& source_ip, uint16_t source_port);
    LinuxTransmitter(const LinuxHeaders* headers = &DEFAULT_LINUX_HEADERS);
    ~LinuxTransmitter();

    /**
     * @brief Send a tensor to the destination. Implementation of BaseTransmitter::send interface method.
     */
    int64_t send(const TransmissionMetadata* metadata, const DLTensor& tensor) override;

private:
    void init_socket();
    int data_socket_fd_ { -1 };

    // state data that is not thread safe without the DataPlane::send level synchronization
    LinuxHeaders linux_headers_;
    uint32_t psn_ { 0 };
    uint32_t frame_number_ { 0 };
};

} // namespace hololink::emulation

#endif
