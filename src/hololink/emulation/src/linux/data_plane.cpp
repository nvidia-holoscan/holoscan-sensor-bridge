/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <atomic>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <time.h>
#include <unistd.h>

#include "data_plane.hpp"
#include "linux/net.hpp"

namespace hololink::emulation {

struct DataPlaneCtxt {
    // metadata protection
    std::mutex metadata_mutex_;
    // bootp thread protection
    std::atomic<bool> running_ { false };
    std::thread bootp_thread_;
};

// HSBEmulator must remain alive for as long as the longest-lived DataPlane object it was used to construct
// IPAddress is both the source IP address and subnet mask to be able to set the appropriate broadcast address (!cannot use INADDR_BROADCAST!)
DataPlane::DataPlane(HSBEmulator& hsb_emulator, const IPAddress& ip_address, uint8_t data_plane_id, uint8_t sensor_id)
    : registers_(hsb_emulator.get_memory())
    , ip_address_(ip_address)
    , configuration_(hsb_emulator.get_config())
    , sensor_id_(sensor_id)
    , data_plane_id_(data_plane_id)
    , hif_address_(0x02000300 + 0x10000 * data_plane_id)
    , sif_address_(0x01000000 + 0x10000 * sensor_id)
{

    // DataPlaneConfiguration
    configuration_.data_plane = data_plane_id;
    // SensorConfiguration. See DataChannel::use_sensor() for more details
    uint8_t sif0_index = sensor_id_ * configuration_.sifs_per_sensor;
    vp_mask_ = 1 << sif0_index;
    switch (sif0_index) {
    case 0:
        frame_end_event_ = EVENT_SIF_0_FRAME_END;
        break;
    case 1:
        frame_end_event_ = EVENT_SIF_1_FRAME_END;
        break;
    default:
        // TODO: what's a reasonable default value since enumerator.cpp does not assign anything to the metadata here
        break;
    }
    vp_address_ = 0x1000 + 0x40 * sif0_index;

    if (data_plane_id_ >= MAX_DATA_PLANES) {
        throw std::runtime_error("Data plane index exceeds maximum number of data planes");
    }
    data_plane_ctxt_ = std::make_unique<DataPlaneCtxt>();

    // register the DataPlane with the HSBEmulator so that it can be started/stopped
    hsb_emulator.add_data_plane(*this);
}

void DataPlane::start()
{
    if (init() != 0 || !data_plane_ctxt_) {
        throw std::runtime_error("Failed to initialize DataPlane");
    }
    if (data_plane_ctxt_->running_) {
        return;
    }
    data_plane_ctxt_->running_ = true;
    data_plane_ctxt_->bootp_thread_ = std::thread(&DataPlane::broadcast_bootp, this);
}

void DataPlane::stop()
{
    if (!is_running()) {
        return;
    }
    data_plane_ctxt_->running_ = false;
    if (data_plane_ctxt_->bootp_thread_.joinable()) {
        data_plane_ctxt_->bootp_thread_.join();
    }
}

bool DataPlane::is_running()
{
    if (!data_plane_ctxt_) {
        return false;
    }
    return data_plane_ctxt_->running_;
}

void DataPlane::update_metadata()
{
}

int64_t DataPlane::send(const DLTensor& tensor, FrameMetadata* frame_metadata)
{
    if (!transmitter_) {
        fprintf(stderr, "DataPlane::send() no transmitter\n");
        return -1;
    }

    if (frame_metadata == nullptr) {
        frame_metadata = DEFAULT_FRAME_METADATA;
    }

    std::scoped_lock<std::mutex> lock(data_plane_ctxt_->metadata_mutex_); // locks metadata for the duration of the send
    update_metadata(); // call the abstract method under the protection of the lock. Transmitter is assured to have synchronized access to the metadata
    int64_t n_bytes = transmitter_->send(metadata_, tensor, frame_metadata);
    if (n_bytes < 0) {
        fprintf(stderr, "DataPlane::send() error sending tensor\n");
    }
    return n_bytes;
}

int64_t DataPlane::send(const uint8_t* buffer, size_t buffer_size, FrameMetadata* frame_metadata)
{
    if (!transmitter_) {
        fprintf(stderr, "DataPlane::send() no transmitter\n");
        return -1;
    }
    std::scoped_lock<std::mutex> lock(data_plane_ctxt_->metadata_mutex_); // locks metadata for the duration of the send
    update_metadata(); // call the abstract method under the protection of the lock. Transmitter is assured to have synchronized access to the metadata
    int64_t n_bytes = transmitter_->send(metadata_, buffer, buffer_size, frame_metadata);
    if (n_bytes < 0) {
        fprintf(stderr, "DataPlane::send() error sending buffer\n");
    }
    return n_bytes;
}
// returns 0 on failure or the number of bytes written on success
// Note that on failure, serializer and buffer contents are in indeterminate state.
static inline size_t serialize_vendor_info(hololink::core::Serializer& serializer, HSBConfiguration& configuration)
{
    size_t start = serializer.length();
    return serializer.append_uint8(configuration.tag)
            && serializer.append_uint8(configuration.tag_length)
            && serializer.append_buffer(configuration.vendor_id, sizeof(configuration.vendor_id))
            && serializer.append_uint8(configuration.data_plane)
            && serializer.append_uint8(configuration.enum_version)
            && serializer.append_uint8(configuration.board_id_lo)
            && serializer.append_uint8(configuration.board_id_hi)
            && serializer.append_buffer(configuration.uuid, sizeof(configuration.uuid))
            && serializer.append_buffer(configuration.serial_num, sizeof(configuration.serial_num))
            && serializer.append_uint16_le(configuration.hsb_ip_version)
            && serializer.append_uint16_le(configuration.fpga_crc)
        ? serializer.length() - start
        : 0;
}

static inline size_t serialize_chaddr(hololink::core::Serializer& serializer, uint8_t* mac_address)
{
    size_t start = serializer.length();
    if (ETHER_ADDR_LEN > BOOTP_CHADDR_SIZE) {
        return serializer.append_buffer(mac_address, BOOTP_CHADDR_SIZE) ? serializer.length() - start : 0;
    }
    return serializer.append_buffer(mac_address, ETHER_ADDR_LEN) ? serializer.length() - start : 0;
}

static inline size_t serialize_bootp_packet(hololink::core::Serializer& serializer, BootpPacket& packet)
{
    size_t start = serializer.length();
    return serializer.append_uint8(packet.op)
            && serializer.append_uint8(packet.htype)
            && serializer.append_uint8(packet.hlen)
            && serializer.append_uint8(packet.hops)
            && serializer.append_uint32_be(packet.xid)
            && serializer.append_uint16_be(packet.secs)
            && serializer.append_uint16_be(packet.flags)
            && serializer.append_uint32_be(packet.ciaddr)
            && serializer.append_uint32_be(packet.yiaddr)
            && serializer.append_uint32_be(packet.siaddr)
            && serializer.append_uint32_be(packet.giaddr)
            && serializer.append_buffer(packet.chaddr, BOOTP_CHADDR_SIZE)
            && serializer.append_buffer(packet.sname, BOOTP_SNAME_SIZE)
            && serializer.append_buffer(packet.file, BOOTP_FILE_SIZE)
            && serializer.append_buffer(packet.vend, BOOTP_VEND_SIZE)
        ? serializer.length() - start
        : 0;
}

// helper function to get the delta in milliseconds between two timespecs and writing to BootpPacket::secs (delta time must fit in 32 bits)
// the timespecs are assumed to be from CLOCK_MONOTONIC
static inline uint32_t get_delta_msecs(const struct timespec& start_time, const struct timespec& end_time)
{
    long long diff_sec = end_time.tv_sec - start_time.tv_sec;
    long long diff_nsec = end_time.tv_nsec - start_time.tv_nsec;
    return static_cast<uint32_t>((diff_sec * 1000) + (diff_nsec / 1000000));
}

// The only field that needs to be updated is BootpPacket::secs
void update_bootp_packet(BootpPacket& bootp_packet, const struct timespec& start_time)
{
    // Get current time from POSIX monotonic clock
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    bootp_packet.secs = (uint16_t)(get_delta_msecs(start_time, ts) / 1000);
}

void init_bootp_packet(BootpPacket& bootp_packet, IPAddress& ip_address, HSBConfiguration& vendor_info)
{
    bootp_packet = (BootpPacket) {
        .op = 1,
        .htype = 1,
        .hlen = 6,
        .hops = 0,
        .xid = vendor_info.data_plane,
        // all other fields initialized to 0
        .ciaddr = ntohl(ip_address.ip_address),
    };

    hololink::core::Serializer chaddr_serializer(bootp_packet.chaddr, sizeof(bootp_packet.chaddr));
    // MacAddress is 6 bytes (std::array<uint8_t, 6> in hololink/core/networking.hpp). Check that it wrote the whole thing
    if (ETHER_ADDR_LEN != serialize_chaddr(chaddr_serializer, ip_address.mac)) {
        throw std::runtime_error("Failed to initialize chaddr in bootp packet with local mac address");
    }

    hololink::core::Serializer vendor_info_serializer(bootp_packet.vend, sizeof(bootp_packet.vend));
    // check if vendor info was written within the buffer
    size_t vendor_info_length = serialize_vendor_info(vendor_info_serializer, vendor_info);
    if (!vendor_info_length) {
        throw std::runtime_error("Failed to initialize vendor information in bootp packet");
    } else if (vendor_info_length > sizeof(bootp_packet.vend)) {
        throw std::runtime_error("buffer overrun serializing vendor info");
    }
}

// for the input parameters, explicitly avoiding sending whole DataPlane object unless needed
int DataPlane::broadcast_bootp()
{
    if (!data_plane_ctxt_->running_) {
        std::this_thread::sleep_for(std::chrono::seconds(BOOTP_INTERVAL_SEC)); // skip an bootp cycle if CPU/compiler sets running out of order
        return -1;
    }

    int bootp_socket_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (bootp_socket_ < 0) {
        fprintf(stderr, "Failed to create UDP socket: %d - %s\n", errno, strerror(errno));
        throw std::runtime_error("Failed to create data channel");
    }

    // Enable address reuse
    int reuse = 1;
    if (setsockopt(bootp_socket_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        fprintf(stderr, "Failed to set reuse address on bootp socket...socket will be set up with this option disabled: %d - %s\n", errno, strerror(errno));
    }

    struct sockaddr_in bind_addr = { 0 };
    bind_addr.sin_family = AF_INET;
    bind_addr.sin_port = htons(BOOTP_REPLY_PORT);
    bind_addr.sin_addr.s_addr = ip_address_.ip_address;

    if (bind(bootp_socket_, (struct sockaddr*)&bind_addr, sizeof(bind_addr)) < 0) {
        fprintf(stderr, "Failed to bind bootp socket: %d - %s\n", errno, strerror(errno));
        close(bootp_socket_);
        bootp_socket_ = -1;
        throw std::runtime_error("Failed to create data channel");
    }

    // Enable broadcast
    int broadcast = 1;
    if (setsockopt(bootp_socket_, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast)) < 0) {
        fprintf(stderr, "Failed to set broadcast option. invalid bootp socket: %d - %s\n", errno, strerror(errno));
        close(bootp_socket_);
        bootp_socket_ = -1;
        throw std::runtime_error("Failed to create data channel");
    }

    // initialize the bootp packet and buffer
    BootpPacket bootp_packet;
    init_bootp_packet(bootp_packet, ip_address_, configuration_);
    uint8_t packet_buffer[BOOTP_SIZE] = { 0 };

    struct iovec iov = {
        .iov_base = &packet_buffer[0],
        .iov_len = BOOTP_SIZE,
    };

    // initialize the socket message header
    // set up broadcast address on ipv4 port 8192
    struct sockaddr_in dest_addr = { 0 };
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(BOOTP_REQUEST_PORT);
    dest_addr.sin_addr.s_addr = get_broadcast_address(ip_address_);
    struct msghdr msg { };
    msg.msg_name = &dest_addr;
    msg.msg_namelen = sizeof(dest_addr);
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    // TODO: implement controlbuf functionality
    // msg.msg_control = controlbuf.data();
    // msg.msg_controllen = controlbuf.size();
    struct timespec start_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // bootp broadcast loop
    while (data_plane_ctxt_->running_) {
        update_bootp_packet(bootp_packet, start_time);
        hololink::core::Serializer packet_serializer(packet_buffer, BOOTP_SIZE);
        serialize_bootp_packet(packet_serializer, bootp_packet);
        ssize_t sent = sendmsg(bootp_socket_, &msg, 0);

        if (sent < 0) {
            fprintf(stderr, "Failed to send bootp packet. error %d - %s\n", errno, strerror(errno));
            break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(BOOTP_INTERVAL_SEC));
    }
    close(bootp_socket_);
    return 0;
}

bool DataPlane::packetizer_enabled() const
{
    constexpr uint32_t PACKETIZER_MODE = 0x0C;
    struct AddressValuePair address_value = { sif_address_ + PACKETIZER_MODE, 0 };
    registers_->read(address_value);
    return (address_value.value >> 28) & 0x1;
}

}