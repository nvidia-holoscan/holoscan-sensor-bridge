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

#include <cerrno>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "coe_transmitter.hpp"
#include "hololink/core/serializer.hpp"
#include "utils.hpp"

#define MAC_HEADER_SIZE 14
#define AVTPDU_HEADER_SIZE 1
#define NTSCF_HEADER_SIZE (11 + AVTPDU_HEADER_SIZE)
#define NTSCF_DATA_LENGTH_MASK 0x7FF
#define ACF_COMMON_HEADER_SIZE 2
// this is also the theoretical max length in quadlets. actual is determined by the MTU
#define ACF_MSG_LENGTH_MASK 0x1FFu
#define ACF_USER0C_HEADER_SIZE (18 + ACF_COMMON_HEADER_SIZE)

#define COE_HEADER_SIZE (MAC_HEADER_SIZE + NTSCF_HEADER_SIZE + ACF_USER0C_HEADER_SIZE)
#define MAX_MESSAGE_SIZE (COE_HEADER_SIZE + ((ACF_MSG_LENGTH_MASK) << 2))
#define COE_PAYLOAD_OFFSET (COE_HEADER_SIZE)
#define COE_PSN_MASK 0xFF

#define COE_FLAG_FRAME_START 0b00000001
#define COE_FLAG_FRAME_END 0b00000010
#define COE_FLAG_LINE_END 0b00010000

namespace hololink::emulation {

static void* memcpy_gpu_to_host(void* dst, const void* src, size_t n)
{
    const auto error = cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost);
    if (cudaSuccess != error) {
        throw std::runtime_error(std::string("cudaMemcpy Device->Host failed.") + std::string(cudaGetErrorString(error)));
    }
    return dst;
}

const COEHeaders DEFAULT_COE_HEADERS = {
    .mac_header = {
        .ethertype = ETHERTYPE_AVTP,
    },
    .ntscf_header = {
        .avtpdu_header = {
            .subtype = SUBTYPE_AVTP_NTSCF,
        },
        .version_ntscf_len_high = COE_VERSION_NTSCF_LEN_HIGH,
        .sequence_num = 0,
        .stream_id = STREAM_ID,
    },
    .acf_user0c_header = {
        .acf_header = {
            .acf_metadata = COE_ACF_MSG_TYPE << ACF_MSG_TYPE_SHIFT,
        },
        .reserved = 0x20, // 0x00 in the spec but Leopard Eagle sets this to 0x20
    },
    // all other fields initialized to 0
};

// returns 0 on failure and the number of bytes written on success. if failure, the underlying buffer is in an indeterminate state.
static inline size_t serialize_mac_header(hololink::core::Serializer& serializer, MACHeader& mac_header)
{
    size_t start = serializer.length();
    return serializer.append_buffer(mac_header.mac_dest, sizeof(mac_header.mac_dest))
            && serializer.append_buffer(mac_header.mac_src, sizeof(mac_header.mac_src))
            && serializer.append_uint16_be(mac_header.ethertype)
        ? serializer.length() - start
        : 0;
}

// returns 0 on failure and the number of bytes written on success. if failure, the underlying buffer is in an indeterminate state.
static inline size_t serialize_avtpdu_header(hololink::core::Serializer& serializer, AVTPDUCommonHeader& avtpdu_header)
{
    return serializer.append_uint8(avtpdu_header.subtype)
        ? 1 // sizeof(uint8_t) is by definition 1
        : 0;
}

// returns 0 on failure and the number of bytes written on success. if failure, the underlying buffer is in an indeterminate state.
static inline size_t serialize_ntscf_header(hololink::core::Serializer& serializer, NTSCFHeader& ntscf_header)
{
    size_t start = serializer.length();
    return (serialize_avtpdu_header(serializer, ntscf_header.avtpdu_header) == AVTPDU_HEADER_SIZE)
            && serializer.append_uint8(ntscf_header.version_ntscf_len_high)
            && serializer.append_uint8(ntscf_header.ntscf_len_low)
            && serializer.append_uint8(ntscf_header.sequence_num)
            && serializer.append_buffer(ntscf_header.stream_id, sizeof(ntscf_header.stream_id))
        ? serializer.length() - start
        : 0;
}

// returns 0 on failure and the number of bytes written on success. if failure, the underlying buffer is in an indeterminate state.
static inline size_t serialize_acf_common_header(hololink::core::Serializer& serializer, ACFCommonHeader& common_header)
{
    return serializer.append_uint16_be(common_header.acf_metadata)
        ? sizeof(uint16_t)
        : 0;
}

// returns 0 on failure and the number of bytes written on success. if failure, the underlying buffer is in an indeterminate state.
static inline size_t serialize_acf_user0c_header(hololink::core::Serializer& serializer, ACFUser0CHeader& acf_header)
{
    size_t start = serializer.length();
    return (serialize_acf_common_header(serializer, acf_header.acf_header)
               && serializer.append_uint8(acf_header.reserved)
               && serializer.append_uint8(acf_header.sensor_info)
               && serializer.append_uint32_be(acf_header.timestamp_sec)
               && serializer.append_uint32_be(acf_header.timestamp_nsec)
               && serializer.append_uint8(acf_header.psn)
               && serializer.append_uint8(acf_header.flags)
               && serializer.append_uint8(acf_header.channel)
               && serializer.append_uint8(acf_header.frame_flags)
               && serializer.append_uint32_be(acf_header.address))
        ? serializer.length() - start
        : 0;
}

// Assumes all fields of headers have been properly sanitized for range checks
// returns 0 on failure and the number of bytes written on success. if failure, the buffer is in an indeterminate state.
static size_t serialize_packet(COEHeaders& headers, uint8_t* __restrict__ buffer, size_t buffer_size, const uint8_t* __restrict__ content, size_t content_size, memcpy_func_t copy_func)
{
    // The way we send Ethernet frames, there is only ever one packet so headersntscf_header.version_ntscf_length should always match headers.acf_header.acf_metadata & 0x1FF
    // should we add ACF payloads, ntscf_length is the source of truth
    static constexpr size_t data_loc = MAC_HEADER_SIZE + NTSCF_HEADER_SIZE + ACF_USER0C_HEADER_SIZE;
    if (content_size > buffer_size || data_loc > buffer_size - content_size) {
        fprintf(stderr, "buffer size is too small to hold the whole packet. found %zu expected %zu\n", buffer_size, data_loc + content_size);
        return 0;
    }
    hololink::core::Serializer serializer(buffer, buffer_size);

    if (MAC_HEADER_SIZE != serialize_mac_header(serializer, headers.mac_header)) {
        fprintf(stderr, "failure in serializing MACHeader\n");
        return 0;
    }

    if (NTSCF_HEADER_SIZE != serialize_ntscf_header(serializer, headers.ntscf_header)) {
        fprintf(stderr, "failure in serializing NTSCFHeader\n");
        return 0;
    }

    if (ACF_USER0C_HEADER_SIZE != serialize_acf_user0c_header(serializer, headers.acf_user0c_header)) {
        fprintf(stderr, "failure in serializing ACFUser0CHeader\n");
        return 0;
    }

    // if content is directly provided, copy it into the buffer
    if (content && copy_func && content_size) {
        copy_func(buffer + data_loc, content, content_size);
    }
    return data_loc + content_size; // return the size of the packet
}

// returns 0 on failure or the number of bytes written to buffer on success
static size_t write_frame_metadata(uint8_t* __restrict__ buffer, size_t buffer_size, const struct timespec& frame_start_timestamp, const struct timespec& metadata_timestamp, size_t n_bytes_sent, uint32_t frame_number, uint32_t psn)
{
    FrameMetadata frame_metadata = {
        .flags = 0,
        .psn = psn,
        .crc = 0,
        // Time when the first sample data for the frame was received
        .timestamp_s = (uint64_t)frame_start_timestamp.tv_sec,
        .timestamp_ns = (uint32_t)frame_start_timestamp.tv_nsec,
        .bytes_written = n_bytes_sent,
        .frame_number = frame_number & 0xFFFF, // only use lower 16 bits of frame number. 0-padded to 32 bits
        // Time at which the metadata packet was sent
        .metadata_s = (uint64_t)metadata_timestamp.tv_sec,
        .metadata_ns = (uint32_t)metadata_timestamp.tv_nsec,
    };

    hololink::core::Serializer serializer = hololink::core::Serializer(buffer, buffer_size);
    return serialize_frame_metadata(serializer, frame_metadata);
}

void COETransmitter::init_socket(const std::string& if_name)
{
    // TODO: wrap this in a function in net.hpp/cpp
    data_socket_fd_ = socket(AF_PACKET, SOCK_RAW, htons(headers_.mac_header.ethertype));
    if (data_socket_fd_ < 0) {
        fprintf(stderr, "Failed to create socket: %d - %s\n", errno, strerror(errno));
        throw std::runtime_error("Failed to create socket");
    }

    // minimal sockaddr_ll initialization for binding to receive on interface
    struct sockaddr_ll sockaddr;
    sockaddr.sll_family = AF_PACKET;
    sockaddr.sll_protocol = htons(headers_.mac_header.ethertype);
    sockaddr.sll_ifindex = if_nametoindex(if_name.c_str());

    if (bind(data_socket_fd_, (struct sockaddr*)&sockaddr, sizeof(sockaddr)) == -1) {
        perror("bind");
        throw std::runtime_error("Failed to bind socket to interface " + if_name);
    }
}

int64_t COETransmitter::send(const TransmissionMetadata* metadata, const DLTensor& tensor)
{
    uint8_t mesg[MAX_MESSAGE_SIZE];
    COETransmissionMetadata* coe_metadata = (COETransmissionMetadata*)metadata;
    if (!coe_metadata->enable_1722b) {
        return 0;
    }

    // initialize locals for hot-loop
    uint8_t* content = (uint8_t*)tensor.data;
    int64_t offset = 0;
    size_t line_offset = 0;
    int64_t n_bytes_sent = 0;
    const int64_t n_bytes = DLTensor_n_bytes(tensor);
    memcpy_func_t copy_func = memcpy;
    uint8_t psn = 0;

    const int64_t payload_size = metadata->payload_size;
    if (!payload_size) {
        throw std::runtime_error("payload_size is 0");
    }
    struct timespec frame_start_timestamp;
    clock_gettime(FRAME_METADATA_CLOCK, &frame_start_timestamp);
    size_t line_threshold = 1u << (coe_metadata->line_threshold_log2_enable_1722b);

    // set header values, if any
    memcpy(headers_.mac_header.mac_dest, coe_metadata->mac_dest, sizeof(headers_.mac_header.mac_dest));
    headers_.acf_user0c_header.sensor_info = coe_metadata->sensor_info;
    headers_.acf_user0c_header.channel = coe_metadata->channel;

    // set the copy function to be used based on content's device location
    switch (tensor.device.device_type) {
    case kDLCPU:
    case kDLCUDAHost:
        break;
    // For managed memory, assume it is on GPU and use double buffering.
    // TODO: add support for using cudaMemcpyDefault determined at initialization time for managed
    //   memory as possible improvement (doesn't appear to guarantee performance improvement).
    //   Need to ensure Unified Addressing is enabled on device.
    case kDLCUDAManaged:
    case kDLCUDA: {
        if (double_buffer_size_ < n_bytes) {
            void* buffer = realloc(double_buffer_, n_bytes);
            if (!buffer) {
                fprintf(stderr, "failed to reallocate double buffer from %lld to %lld\n", (long long)double_buffer_size_, (long long)n_bytes);
                throw std::runtime_error("failed to reallocate double buffer");
            }
            double_buffer_ = buffer;
            double_buffer_size_ = n_bytes;
        }
        cudaError_t error = cudaMemcpy(double_buffer_, tensor.data, (size_t)n_bytes, cudaMemcpyDeviceToHost); // should change to asynchronous copy and sync on the stream
        if (cudaSuccess != error) {
            fprintf(stderr, "cudaMemcpy Device->Host failed: %s\n", cudaGetErrorString(error));
            throw std::runtime_error("cudaMemcpy Device->Host failed");
        }
        content = (uint8_t*)double_buffer_;
        break;
    }
    default:
        fprintf(stderr, "Unsupported device memory type: %d\n", (int)tensor.device.device_type);
        return -1;
    }

    const uint32_t packet_frame_number = (frame_number_ & 0x3) << 28; // pre-shift for | operation

    // send out the packet
    while (offset < n_bytes) {
        int64_t n_bytes_to_send = n_bytes - offset;
        if (n_bytes_to_send > payload_size) {
            n_bytes_to_send = payload_size;
        }
        // line offset gets updated up front so that the flag gets set appropriately
        line_offset += n_bytes_to_send;

        struct timespec packet_timestamp;
        clock_gettime(FRAME_METADATA_CLOCK, &packet_timestamp);
        // prepare updated headers for packet
        headers_.acf_user0c_header.address = packet_frame_number | (offset & 0xFFFFFFF);
        headers_.acf_user0c_header.timestamp_sec = (uint32_t)packet_timestamp.tv_sec;
        headers_.acf_user0c_header.timestamp_nsec = (uint32_t)packet_timestamp.tv_nsec;
        headers_.acf_user0c_header.psn = psn;
        uint8_t flags = !offset ? COE_FLAG_FRAME_START : 0;
        if (line_offset >= line_threshold) {
            flags |= COE_FLAG_LINE_END;
            line_offset = 0;
        }
        headers_.acf_user0c_header.frame_flags = flags;

        size_t message_size = serialize_packet(headers_, mesg, sizeof(mesg), content + offset, n_bytes_to_send, copy_func);
        if (message_size != COE_HEADER_SIZE + (size_t)n_bytes_to_send) {
            fprintf(stderr, "error in writing packet. found %zu bytes sent expected %zu\n", message_size, COE_HEADER_SIZE + n_bytes_to_send);
        } else if (::send(data_socket_fd_, &mesg, message_size, 0) <= 0) {
            fprintf(stderr, "packet not sent: %d - %s\n", errno, strerror(errno));
        } else {
            n_bytes_sent += n_bytes_to_send;
        }

        offset += n_bytes_to_send;
        psn = (psn + 1) & COE_PSN_MASK;
    }

    // write metadata
    struct timespec packet_timestamp;
    clock_gettime(FRAME_METADATA_CLOCK, &packet_timestamp);
    size_t n_bytes_to_send = write_frame_metadata(&mesg[COE_PAYLOAD_OFFSET], MAX_MESSAGE_SIZE - COE_PAYLOAD_OFFSET, frame_start_timestamp, packet_timestamp, n_bytes_sent, frame_number_, psn);
    if (n_bytes_to_send != FRAME_METADATA_SIZE) {
        fprintf(stderr, "could not write frame metadata. found %zu expected %u\n", n_bytes_to_send, FRAME_METADATA_SIZE);
    }

    headers_.acf_user0c_header.address = packet_frame_number | (offset & 0xFFFFFFF);
    headers_.acf_user0c_header.timestamp_sec = (uint32_t)packet_timestamp.tv_sec;
    headers_.acf_user0c_header.timestamp_nsec = (uint32_t)packet_timestamp.tv_nsec;
    headers_.acf_user0c_header.psn = psn;
    headers_.acf_user0c_header.frame_flags = COE_FLAG_FRAME_END | (line_threshold >= n_bytes_to_send ? COE_FLAG_LINE_END : 0);

    size_t message_size = serialize_packet(headers_, mesg, sizeof(mesg), nullptr, n_bytes_to_send, nullptr);
    if (message_size != COE_PAYLOAD_OFFSET + n_bytes_to_send) {
        fprintf(stderr, "error in serialize frame metadata packet. found %zu expected %zu\n", message_size, COE_PAYLOAD_OFFSET + n_bytes_to_send);
    } else if (::send(data_socket_fd_, &mesg, message_size, 0) <= 0) {
        // TODO: need more sophisticated error handling here. For now, just show error and move on
        fprintf(stderr, "frame metadata packet not sent\n");
    } else {
        n_bytes_sent += n_bytes_to_send;
    }

    // update LinuxTransmitter state and return number of bytes successfully sent
    psn = (psn + 1) & COE_PSN_MASK;
    frame_number_ = (frame_number_ + 1); // wrap-around OK for 32 bits

    return n_bytes_sent;
}

COETransmitter::COETransmitter(const std::string& if_name)
    : headers_(DEFAULT_COE_HEADERS)
{
    IPAddress iface;
    mac_from_if(iface, if_name);
    memcpy(headers_.mac_header.mac_src, iface.mac.data(), sizeof(headers_.mac_header.mac_src));
    init_socket(if_name);
}

COETransmitter::COETransmitter(const std::string& if_name, const std::array<uint8_t, 6>& mac_src)
    : headers_(DEFAULT_COE_HEADERS)
{
    memcpy(headers_.mac_header.mac_src, mac_src.data(), sizeof(headers_.mac_header.mac_src));
    init_socket(if_name);
}

COETransmitter::COETransmitter(const std::string& if_name, const COEHeaders& headers)
    : headers_(headers)
{
    init_socket(if_name);
}

COETransmitter::~COETransmitter()
{
    if (data_socket_fd_ >= 0) {
        close(data_socket_fd_);
    }
    free(double_buffer_);
    double_buffer_ = nullptr;
    double_buffer_size_ = 0;
}

} // namespace hololink::emulation
