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

#ifndef COE_TRANSMITTER_HPP
#define COE_TRANSMITTER_HPP

#include "dlpack/dlpack.h"
#include <array>
#include <cstdint>

#include "hololink/emulation/base_transmitter.hpp"
#include "net.hpp"
#include <linux/if_packet.h>

#define ETHERTYPE_AVTP 0x22F0u /* IEEE 1722 */
#define SUBTYPE_AVTP_NTSCF 0x82u
#define COE_ACF_MSG_TYPE 0x06u /* ACF_SERIAL??? need to check with Patrick/Sanjeev Jain (according to Corey) */
#define ACF_MSG_TYPE_SHIFT 9u
#define STREAM_ID                                      \
    {                                                  \
        0x00, 0x00, 0x00, 0x00, 0xAA, 0xAA, 0xAA, 0xAA \
    }
#define PACKET_SEQUENCE_NUMBER_MASK 0x7F
#define COE_VERSION_NTSCF_LEN_HIGH 0x80 /* SV = 0b1, Version = 0b000, R = 0b0, length_high_3bits = 0b000 */

namespace hololink::emulation {

struct MACHeader {
    uint8_t mac_dest[6];
    uint8_t mac_src[6];
    uint16_t ethertype; // 0x22F0
};

// this is a constant in HSB 1722 packets
struct AVTPDUCommonHeader {
    uint8_t subtype; // 0x82
    // header specific and version fields subsumed by that header
};

// these are constants in HSB 1722 packets
struct NTSCFHeader {
    AVTPDUCommonHeader avtpdu_header;
    uint8_t version_ntscf_len_high; // lowest 3 bits for NTSCF data
    uint8_t ntscf_len_low; // data length occupies 11 lsbs. See S 9.2 Figure 50 of IEEE 1722.1-2016. This is the size of the ACF_Payload (everything after) NTSCFHeader
    uint8_t sequence_num; // this is NOT packet sequence number
    uint8_t stream_id[8]; // defined in IEEE 802.1Q-2014
};

// these are constants in HSB 1722 packets
struct ACFCommonHeader {
    uint16_t acf_metadata; // msg_type occupies 7 msbs, msg_length occupies 9 lsbs. See S 9.4 Figure 52 of IEEE 1722.1-2106
                           // msg_length is the number of quadlets (4 bytes/quadlet) in the ACF_Payload (including this header)
                           // NOTE: the way the msg_length is read, it is not 9 bits, but 8.
};

struct ACFUser0CHeader {
    ACFCommonHeader acf_header;
    uint8_t reserved;
    uint8_t sensor_info; // SIF port index
    uint32_t timestamp_sec;
    uint32_t timestamp_nsec;
    uint8_t psn; // packet sequence number. only 7 bits are used, so max is 127
    uint8_t flags;
    uint8_t channel; // only 6 bits are used, so max is 63
    uint8_t frame_flags;
    uint32_t address; // first 4 bits are 0b00 and 2 bits of frame number. Then 28-bit of virtual address offset
};

struct COEHeaders {
    MACHeader mac_header;
    NTSCFHeader ntscf_header; // adds padding. ntscf_header and the acf header are 32-bit aligned, mac_header is 16-bit aligned
    ACFUser0CHeader acf_user0c_header;
};

extern const COEHeaders DEFAULT_COE_HEADERS;

struct COETransmissionMetadata {
    struct TransmissionMetadata transmission_metadata;
    // TODO: common fields to be added to TransmissionMetadata
    uint8_t mac_dest[6];
    // ACF fields
    uint8_t sensor_info;
    uint8_t channel; // only 6 bits are used, so max is 63
    uint8_t line_threshold_log2_enable_1722b; // upper 7 bits for line_threshold_log2. 0th bit is enable_1722B, which should always be 1
    bool enable_1722b; // 1 bit for enable_1722B, which should always be 1
};

/**
 * @brief The COETransmitter implements the BaseTransmitter interface and encapsulates the transport over IEEE 1722B.
 */
class COETransmitter : public BaseTransmitter {
public:
    COETransmitter() = delete;
    /**
     * @brief Construct a new COETransmitter object
     * @param if_name The name of the network interface to use.
     * @note The MAC address is derived from the interface name using the mac_from_if function in net.hpp.
     */
    COETransmitter(const std::string& if_name);
    /**
     * @brief Construct a new COETransmitter object
     * @param if_name The name of the network interface to use.
     * @param mac_src The MAC address to use.
     */
    COETransmitter(const std::string& if_name, const std::array<uint8_t, 6>& mac_src);
    /**
     * @brief Construct a new COETransmitter object
     * @param if_name The name of the network interface to use.
     * @param headers The fully configurable headers to use. See source code for details
     */
    COETransmitter(const std::string& if_name, const COEHeaders& headers);
    ~COETransmitter();

    /**
     * @brief Send a tensor to the destination using the TransmissionMetadata provided. Implementation of BaseTransmitter::send interface method.
     * @param metadata The metadata for the transmission. This is always aliased from the appropriate type of metadata for the Transmitter instance.
     * @param tensor The tensor to send. See dlpack.h for its contents and semantics.
     * @return The number of bytes sent or < 0 on error
     */
    int64_t send(const TransmissionMetadata* metadata, const DLTensor& tensor) override;

private:
    void init_socket(const std::string& if_name);
    int data_socket_fd_ { -1 };

    COEHeaders headers_;
    uint32_t frame_number_ { 0 }; // only 2 bits used
    // double buffering is for GPU inputs currently
    void* double_buffer_ { nullptr };
    int64_t double_buffer_size_ { 0 };
};

} // namespace hololink::emulation

#endif // COE_TRANSMITTER_HPP