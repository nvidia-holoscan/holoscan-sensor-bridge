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

#include <cassert>
#include <cstdint>
#include <cstring>

#include <zlib.h>

#include "cuda_runtime.h"
#include "dlpack/dlpack.h"

#include "net.hpp"

#include "hololink/core/serializer.hpp"
#include "linux_transmitter.hpp"
#include "utils.hpp"

// payload is 1500 bytes (MTU), minus the IP header, UDP header, and CRC (4 bytes)
#define DEFAULT_PAYLOAD_SIZE (((1500 - sizeof(IPHeader) - sizeof(UDPHeader) - sizeof(uint32_t)) / PAGE_SIZE) * PAGE_SIZE)
#define DEFAULT_ROCE_PAYLOAD_SIZE (((1500 - sizeof(IPHeader) - sizeof(UDPHeader) - sizeof(BTHeader) - sizeof(RETHeader) - sizeof(uint32_t)) / PAGE_SIZE) * PAGE_SIZE)

#define IB_OPCODE_WRITE 0x2A
#define IB_OPCODE_WRITE_IMMEDIATE 0x2B

/* 20 bytes for IP header, 8 bytes for UDP header */
#define IP_HEADER_SIZE 20
#define UDP_HEADER_SIZE 8
#define UDP_PAYLOAD_OFFSET (UDP_HEADER_SIZE + IP_HEADER_SIZE)
#define BT_HEADER_SIZE 12
#define RET_HEADER_SIZE 16
#define IB_HEADER_SIZE (BT_HEADER_SIZE + RET_HEADER_SIZE)
// this is specifically the payload offset for our opcode. Note that unreliable write/write immediate and other IB opcodes do not necessary have this offset
#define IB_PAYLOAD_OFFSET (UDP_PAYLOAD_OFFSET + IB_HEADER_SIZE)
#define FRAME_METADATA_SIZE 48

// Note that if monotonic times are critical downstream (accuracy, repeatability or conflict with other system process), this clock must be configurable
#define FRAME_METADATA_CLOCK CLOCK_REALTIME

namespace hololink::emulation {

typedef void* (*memcpy_func_t)(void* dst, const void* src, size_t n) noexcept;

// all of FrameMetadata is assumed to be in little endian
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
};

// returns 0 on failure or the number of bytes written on success.
// Note that on failure, serializer and buffer contents are in indeterminate state.
static inline size_t serialize_frame_metadata(hololink::core::Serializer& serializer, FrameMetadata& frame_metadata)
{
    size_t start = serializer.length();
    return serializer.append_uint32_be(frame_metadata.flags)
            && serializer.append_uint32_be(frame_metadata.psn)
            && serializer.append_uint32_be(frame_metadata.crc)
            && serializer.append_uint64_be(frame_metadata.timestamp_s)
            && serializer.append_uint32_be(frame_metadata.timestamp_ns)
            && serializer.append_uint64_be(frame_metadata.bytes_written)
            && serializer.append_uint16_be(0)
            && serializer.append_uint16_be(frame_metadata.frame_number & 0xFFFF)
            && serializer.append_uint64_be(frame_metadata.metadata_s)
            && serializer.append_uint32_be(frame_metadata.metadata_ns)
        ? serializer.length() - start
        : 0;
}

// returns 0 on failure or the number of bytes written on success
// Note that on failure, serializer and buffer contents are in indeterminate state.
static inline size_t serialize_ip_header(hololink::core::Serializer& serializer, IPHeader& header_ip)
{
    size_t start = serializer.length();
    return serializer.append_uint8(header_ip.version_and_header_length)
            && serializer.append_uint8(header_ip.type_of_service)
            && serializer.append_uint16_be(header_ip.length)
            && serializer.append_uint16_be(header_ip.identification)
            && serializer.append_uint16_be(header_ip.flags_and_fragment_offset)
            && serializer.append_uint8(header_ip.time_to_live)
            && serializer.append_uint8(header_ip.protocol)
            && serializer.append_uint16_be(header_ip.checksum)
            && serializer.append_uint32_be(header_ip.source_ip_address)
            && serializer.append_uint32_be(header_ip.destination_ip_address)
        ? serializer.length() - start
        : 0;
}

// returns 0 on failure or the number of bytes written on success
// Note that on failure, serializer and buffer contents are in indeterminate state.
static inline size_t serialize_udp_header(hololink::core::Serializer& serializer, UDPHeader& header_udp)
{
    size_t start = serializer.length();
    return serializer.append_uint16_be(header_udp.source_port)
            && serializer.append_uint16_be(header_udp.destination_port)
            && serializer.append_uint16_be(header_udp.length)
            && serializer.append_uint16_be(header_udp.checksum)
        ? serializer.length() - start
        : 0;
}

// returns 0 on failure or the number of bytes written on success
// Note that on failure, serializer and buffer contents are in indeterminate state.
static inline size_t serialize_bt_header(hololink::core::Serializer& serializer, BTHeader& bt_header)
{
    size_t start = serializer.length();
    return serializer.append_uint8(bt_header.opcode)
            && serializer.append_uint8(bt_header.flags)
            && serializer.append_uint16_be(bt_header.p_key)
            && serializer.append_uint32_be(bt_header.qp)
            && serializer.append_uint32_be(bt_header.psn & 0xFFFFFF)
        ? serializer.length() - start
        : 0;
}

// returns 0 on failure or the number of bytes written on success
// Note that on failure, serializer and buffer contents are in indeterminate state.
static inline size_t serialize_ret_header(hololink::core::Serializer& serializer, RETHeader& ret_header)
{
    size_t start = serializer.length();
    return serializer.append_uint64_be(ret_header.vaddress)
            && serializer.append_uint32_be(ret_header.rkey)
            && serializer.append_uint32_be(ret_header.content_size)
        ? serializer.length() - start
        : 0;
}

// writes the crc for the buffer of size buffer_size with content of length content_size.
// essentially the crc is written into buffer at location content_size
// Per IB spec, content_size must include all headers unless we go outside of subnet
bool update_crc(uint8_t* buffer, size_t buffer_size, size_t content_size)
{
    if (buffer_size < content_size + sizeof(uint32_t)) {
        fprintf(stderr, "buffer size is too small to update crc\n");
        return false;
    }
    static uint8_t crc_init_buf[8] = { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };
    uint32_t crc = crc32(crc32(0L, crc_init_buf, sizeof(crc_init_buf)), buffer, content_size);
    // per zlib docs, crc is already in network byte order and application should not modify. direct copy.
    memcpy(buffer + content_size, &crc, sizeof(uint32_t));
    return true;
}

// assumes that the buffer size is large enough to hold the whole packet, else UB.
// if any of content, headers.ret_h.content_size, or copy_func are 0/nullptr, the payload will not be copied and it is up to the caller to ensure payload is written to buffer
size_t serialize_packet(LinuxHeaders& headers, uint8_t* __restrict__ buffer, size_t buffer_size, const uint8_t* __restrict__ content, memcpy_func_t copy_func)
{
    // local copy of content_size.
    // NOTE: headers.ret_h.content_size is the single source of truth for the content_size
    uint32_t content_size = headers.ret_h.content_size;
    // calculate leangths for headers
    size_t crc_offset = IP_HEADER_SIZE + UDP_HEADER_SIZE + IB_HEADER_SIZE + content_size;
    size_t message_size = crc_offset + sizeof(uint32_t);
    if (message_size > buffer_size) {
        fprintf(stderr, "buffer size is too small to hold the whole packet. found %zu expected %zu\n", buffer_size, message_size);
        return 0;
    }

    // ensure the header lengths are consistent
    headers.ip_h.length = message_size;
    headers.udp_h.length = message_size - IP_HEADER_SIZE;

    hololink::core::Serializer serializer = hololink::core::Serializer(buffer, buffer_size);

    // copy and use the magic values for CRC calculation
    IPHeader ip_header = headers.ip_h;
    ip_header.type_of_service = 0xFF;
    ip_header.time_to_live = 0xFF;
    ip_header.checksum = 0xFFFF;
    UDPHeader udp_header = headers.udp_h;
    udp_header.checksum = 0xFFFF;

    // each header write is checked individually because otherwise the serializer will write to the wrong portion of the packet. Exit early.
    size_t ip_header_offset = serialize_ip_header(serializer, ip_header);
    if (ip_header_offset != IP_HEADER_SIZE) {
        fprintf(stderr, "failure in serializing IPHeader\n");
        return 0;
    }
    size_t udp_header_offset = serialize_udp_header(serializer, udp_header);
    if (udp_header_offset != UDP_HEADER_SIZE) {
        fprintf(stderr, "failure in serializing UDPHeader\n");
        return 0;
    }
    // write the ib headers
    if (BT_HEADER_SIZE != serialize_bt_header(serializer, headers.bt_h)) {
        fprintf(stderr, "failure in serializing BTHeader\n");
        return 0;
    }
    if (RET_HEADER_SIZE != serialize_ret_header(serializer, headers.ret_h)) {
        fprintf(stderr, "failure in serializing RETHeader\n");
        return 0;
    }

    if (content_size && buffer && copy_func) { // guard against 0 byte copy, memcpy UB, and no copy function
        // Serializer object does not have the ability to navigate the buffer or handle non-cpu memory copies, so work outside the serializer here
        copy_func(buffer + IB_PAYLOAD_OFFSET, content, content_size);
    }

    update_crc(buffer, buffer_size, crc_offset);

    // reset the serializer to the beginning of the buffer to write the actual IPHeader & UDPHeader
    serializer = hololink::core::Serializer(buffer, buffer_size);
    ip_header_offset = serialize_ip_header(serializer, headers.ip_h);
    if (ip_header_offset != IP_HEADER_SIZE) {
        fprintf(stderr, "failure in serializing IPHeader\n");
        return 0;
    }
    udp_header_offset = serialize_udp_header(serializer, headers.udp_h);
    if (udp_header_offset != UDP_HEADER_SIZE) {
        fprintf(stderr, "failure in serializing UDPHeader\n");
        return 0;
    }

    return message_size;
}

// returns 0 on failure or the number of bytes written to buffer on success
size_t write_frame_metadata(uint8_t* __restrict__ buffer, size_t buffer_size, const struct timespec& frame_start_timestamp, size_t n_bytes_sent, uint32_t frame_number, uint32_t psn, uint8_t page)
{
    int32_t immediate_value = (page) | ((psn & 0xFFFFFF) << 8);
    struct timespec meta_timestamp;
    clock_gettime(FRAME_METADATA_CLOCK, &meta_timestamp);
    FrameMetadata frame_metadata = {
        .flags = 0,
        .psn = psn,
        .crc = 0,
        // Time when the first sample data for the frame was received
        .timestamp_s = (uint64_t)frame_start_timestamp.tv_sec,
        .timestamp_ns = (uint32_t)frame_start_timestamp.tv_nsec,
        .bytes_written = n_bytes_sent,
        .frame_number = frame_number,
        // Time at which the metadata packet was sent
        .metadata_s = (uint64_t)meta_timestamp.tv_sec,
        .metadata_ns = (uint32_t)meta_timestamp.tv_nsec,
    };

    hololink::core::Serializer serializer = hololink::core::Serializer(buffer, buffer_size);
    serializer.append_uint32_be(immediate_value);
    size_t frame_metadata_size = serialize_frame_metadata(serializer, frame_metadata);
    if (frame_metadata_size != FRAME_METADATA_SIZE) {
        fprintf(stderr, "failure in serializing frame metadata\n");
        return 0;
    }
    return sizeof(uint32_t) + frame_metadata_size;
}

static inline void* memcpy_gpu(void* dst, const void* src, size_t n) noexcept
{
    if (cudaSuccess != cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost)) {
        return nullptr;
    }
    return dst;
}

void LinuxTransmitter::init_socket()
{
    data_socket_fd_ = socket(AF_INET, SOCK_RAW, IPPROTO_RAW);
    if (data_socket_fd_ < 0) {
        fprintf(stderr, "Failed to create socket: %d - %s\n", errno, strerror(errno));
        throw std::runtime_error("Failed to create socket");
    }
    int on = 1;
    if (setsockopt(data_socket_fd_, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)) < 0) {
        fprintf(stderr, "Failed to set socket reused address: %d - %s\n", errno, strerror(errno));
        throw std::runtime_error("Failed to set socket reuse");
    }
    // bind to the selected port
    struct sockaddr_in bind_addr {
        .sin_family = AF_INET,
        .sin_port = htons(linux_headers_.udp_h.source_port),
        .sin_addr = {
            /* binds to any IP address, but will send out as if from linux_headers_.ip_h.source_ip_address */
            // TODO: test multiple HSBEmulators on same adapter. Might have to use source_ip here instead of INADDR_ANY
            .s_addr = INADDR_ANY,
        },
    };
    if (bind(data_socket_fd_, (struct sockaddr*)&bind_addr, sizeof(bind_addr)) < 0) {
        fprintf(stderr, "Failed to bind socket: %d - %s\n", errno, strerror(errno));
        throw std::runtime_error("Failed to bind socket");
    }
}

LinuxTransmitter::LinuxTransmitter(const LinuxHeaders* headers)
    : linux_headers_(*headers)
{
    init_socket();
}

// Infiniband transmitter without using ib verbs apis
LinuxTransmitter::LinuxTransmitter(const std::string& source_ip, uint16_t source_port)
    : linux_headers_(DEFAULT_LINUX_HEADERS)
{
    // assign constant source ip and port to headers. Need to undo network byte order of inet_addr
    linux_headers_.ip_h.source_ip_address = ntohl(inet_addr(source_ip.c_str()));
    linux_headers_.udp_h.source_port = source_port;

    init_socket();
}

LinuxTransmitter::~LinuxTransmitter()
{
    close(data_socket_fd_);
}

// returns -1 on failure or the number of bytes successfully sent, which may be 0.
int64_t LinuxTransmitter::send(const TransmissionMetadata* metadata, const DLTensor& tensor)
{
    // buffer for packet data
    uint8_t mesg[hololink::core::UDP_PACKET_SIZE];
    // buffer for immediate value and frame metadata
    LinuxTransmissionMetadata* linux_metadata = (LinuxTransmissionMetadata*)metadata;
    // if no destination port or address, short circuit and return 0
    if (!linux_metadata->dest_port) {
        return 0;
    }
    if (!linux_metadata->dest_ip_address) {
        return 0;
    }

    // copy metadata to linux_headers_ and local data
    const uint16_t udp_payload_size = metadata->payload_size;
    linux_headers_.ip_h.destination_ip_address = linux_metadata->dest_ip_address;
    linux_headers_.udp_h.destination_port = linux_metadata->dest_port;
    BTHeader* bt_header = (BTHeader*)&linux_headers_.bt_h;
    bt_header->opcode = IB_OPCODE_WRITE; // overwrite in immediate packet
    bt_header->qp = (0xFF << 24) | (linux_metadata->qp & 0xFFFFFF);
    bt_header->psn = psn_; // overwrite in loop
    RETHeader* ret_header = (RETHeader*)&linux_headers_.ret_h;
    ret_header->vaddress = linux_metadata->address;
    ret_header->rkey = linux_metadata->rkey;
    ret_header->content_size = udp_payload_size; // overwrite in loop

    int64_t n_bytes = DLTensor_n_bytes(tensor);
    int64_t offset = 0;
    int64_t n_bytes_sent = 0;
    uint8_t* content = (uint8_t*)tensor.data;
    memcpy_func_t copy_func = nullptr;

    struct timespec frame_start_timestamp;
    clock_gettime(FRAME_METADATA_CLOCK, &frame_start_timestamp);

    // set up the data socket destination address
    struct sockaddr_in dest_addr {
        .sin_family = AF_INET,
        .sin_port = htons(linux_metadata->dest_port),
        .sin_addr = {
            .s_addr = htonl(linux_metadata->dest_ip_address),
        },
    };

    // set the copy function to be used based on content's device location
    switch (tensor.device.device_type) {
    case kDLCPU:
    case kDLCUDAHost:
        copy_func = memcpy;
        break;
    // For managed memory, assume it is on GPU.
    // TODO: add support for using cudaMemcpyDefault determined at initialization time for managed
    //   memory as possible improvement (doesn't appear to guarantee performance improvement).
    //   Need to ensure Unified Addressing is enabled on device.
    case kDLCUDAManaged:
    case kDLCUDA:
        copy_func = memcpy_gpu;
        break;
    default:
        fprintf(stderr, "Unsupported device memory type: %d\n", (int)tensor.device.device_type);
        return -1;
    }

    while (offset < n_bytes) {
        int64_t n_bytes_to_send = n_bytes - offset;
        if (n_bytes_to_send > udp_payload_size) {
            n_bytes_to_send = udp_payload_size;
        }
        // prepare headers for packet
        bt_header->psn = psn_;
        ret_header->content_size = n_bytes_to_send;
        ret_header->vaddress = linux_metadata->address + offset;

        size_t message_size = serialize_packet(linux_headers_, mesg, sizeof(mesg), content + offset, copy_func);
        // sizeof(uin32_t) to account for crc
        if (message_size != IB_PAYLOAD_OFFSET + n_bytes_to_send + sizeof(uint32_t)) {
            fprintf(stderr, "error in writing packet psn %d. found %zu expected %zu\n", psn_, message_size, IB_PAYLOAD_OFFSET + n_bytes_to_send + sizeof(uint32_t));
        } else if (sendto(data_socket_fd_, &mesg, message_size, 0, (struct sockaddr*)&dest_addr, sizeof(dest_addr)) <= 0) {
            // TODO: need more sophisticated error handling here. For now, just show error and move on
            fprintf(stderr, "packet psn %d not sent\n", psn_);
        } else {
            n_bytes_sent += n_bytes_to_send;
        }

        offset += n_bytes_to_send;
        psn_ = (psn_ + 1) & 0xFFFFFF;
    }

    // write directly into the mesg packet buffer
    size_t n_bytes_to_send = write_frame_metadata(&mesg[IB_PAYLOAD_OFFSET], sizeof(uint32_t) + FRAME_METADATA_SIZE, frame_start_timestamp, n_bytes_sent, frame_number_, psn_, linux_metadata->page);

    // prep headers for immediate packet
    bt_header->opcode = IB_OPCODE_WRITE_IMMEDIATE;
    bt_header->psn = psn_;
    ret_header->content_size = n_bytes_to_send;
    ret_header->vaddress = linux_metadata->address + linux_metadata->metadata_offset;

    if (!n_bytes_to_send) {
        fprintf(stderr, "could not write frame metadata\n");
    }

    // pass nullptr content and copy function since metadata is already written to the mesg packet buffer
    size_t message_size = serialize_packet(linux_headers_, mesg, sizeof(mesg), nullptr, nullptr);
    // 2 * sizeof(uin32_t) to account for immediate value and crc
    if (message_size != IB_PAYLOAD_OFFSET + n_bytes_to_send + sizeof(uint32_t)) {
        fprintf(stderr, "error in serialize immediate packet. found %zu expected %zu\n", message_size, IB_PAYLOAD_OFFSET + n_bytes_to_send + sizeof(uint32_t));
    } else if (sendto(data_socket_fd_, &mesg, message_size, 0, (struct sockaddr*)&dest_addr, sizeof(dest_addr)) <= 0) {
        // TODO: need more sophisticated error handling here. For now, just show error and move on
        fprintf(stderr, "immediate packet not sent\n");
    } else {
        n_bytes_sent += n_bytes_to_send;
    }

    // update LinuxTransmitter state and return number of bytes successfully sent
    psn_ = (psn_ + 1) & 0xFFFFFF;
    frame_number_++;
    return n_bytes_sent;
}

} // namespace hololink::emulation
