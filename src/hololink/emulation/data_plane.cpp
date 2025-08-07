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

#include <assert.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <time.h>

#include "hololink/core/serializer.hpp"

#include "base_transmitter.hpp"
#include "data_plane.hpp"
#include "hsb_emulator.hpp"
#include "net.hpp"

namespace hololink::emulation {

// constants for bootp packet
#define BOOTP_CHADDR_SIZE 16u
#define BOOTP_SNAME_SIZE 64u
#define BOOTP_FILE_SIZE 128u
#define BOOTP_VEND_SIZE 64u
#define BOOTP_PORT 8192u
#define BOOTP_BUFFER_SIZE 1500u
#define BOOTP_INTERVAL_SEC 1u
#define BOOTP_REQUEST_PORT 12267u
#define BOOTP_REPLY_PORT 12268u
#define BOOTP_SIZE 300u

struct BootpPacket {
    uint8_t op; // 1: bootrequest, 2: bootreply
    uint8_t htype; // 1: 10mb ethernet. See https://www.iana.org/assignments/arp-parameters/arp-parameters.xhtml
    uint8_t hlen; // 6: 10mb ethernet
    uint8_t hops; // 0: no hops set by client
    uint32_t xid; // transaction id
    uint16_t secs; // client specified seconds, e.g. since boot. // this is updated every bootp cycle
    uint16_t flags; // 0: no flags. RFC1542 defines MSB as broadcast bit, requires other bits to be 0.
    uint32_t ciaddr; // client ip address. filled in by client if known
    uint32_t yiaddr; // your ip address. filled by server if client doesn't know
    uint32_t siaddr; // server ip address. filled in bootreply
    uint32_t giaddr; // gateway ip address
    uint8_t chaddr[BOOTP_CHADDR_SIZE]; // client hardware address
    uint8_t sname[BOOTP_SNAME_SIZE]; // server host name
    uint8_t file[BOOTP_FILE_SIZE]; // boot file name
    uint8_t vend[BOOTP_VEND_SIZE]; // vendor specific
};

static inline size_t serialize_chaddr(hololink::core::Serializer& serializer, hololink::core::MacAddress& mac_address)
{
    size_t start = serializer.length();
    if (mac_address.size() > BOOTP_CHADDR_SIZE) {
        return serializer.append_buffer(mac_address.data(), BOOTP_CHADDR_SIZE) ? serializer.length() - start : 0;
    }
    return serializer.append_buffer(mac_address.data(), mac_address.size()) ? serializer.length() - start : 0;
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

// HSBEmulator must remain alive for as long as the longest-lived DataPlane object it was used to construct
// IPAddress is both the source IP address and subnet mask to be able to set the appropriate broadcast address (!cannot use INADDR_BROADCAST!)
DataPlane::DataPlane(HSBEmulator& hsb_emulator, const IPAddress& ip_address, DataPlaneID data_plane_id, SensorID sensor_id)
    : hsb_emulator_(hsb_emulator)
    , registers_(hsb_emulator.get_memory())
    , ip_address_(ip_address)
    , configuration_(hsb_emulator.get_config())
    , sensor_id_(sensor_id)
{
    if (!data_plane_map.count(data_plane_id)) {
        throw std::invalid_argument("Invalid data plane id");
    }
    if (!sensor_map.count(sensor_id)) {
        throw std::invalid_argument("Invalid sensor id");
    }

    configuration_.data_plane = data_plane_id;

    // register the DataPlane with the HSBEmulator so that it can be started/stopped
    hsb_emulator.add_data_plane(*this);
}

DataPlane::~DataPlane()
{
    stop();
}

void DataPlane::start()
{
    if (running_) {
        return;
    }
    running_ = true;
    bootp_thread_ = std::thread(&DataPlane::broadcast_bootp, this);
}

void DataPlane::stop()
{
    if (!is_running()) {
        return;
    }
    running_ = false;
    if (bootp_thread_.joinable()) {
        bootp_thread_.join();
    }
}

bool DataPlane::is_running()
{
    return running_;
}

int64_t DataPlane::send(const DLTensor& tensor)
{
    if (!transmitter_) {
        fprintf(stderr, "DataPlane::send() no transmitter\n");
        return -1;
    }

    std::scoped_lock<std::mutex> lock(metadata_mutex_); // locks metadata for the duration of the send
    update_metadata(); // call the abstract method under the protection of the lock. Transmitter is assured to have synchronized access to the metadata
    int64_t n_bytes = transmitter_->send(metadata_, tensor);
    if (n_bytes < 0) {
        fprintf(stderr, "DataPlane::send() error sending tensor\n");
    }
    return n_bytes;
}

// helper function to get the delta in seconds between two timespecs and writing to BootpPacket::secs (delta time must fit in 16 bits)
// the timespecs are assumed to be from CLOCK_MONOTONIC
static inline uint16_t get_delta_secs(const struct timespec& start_time, const struct timespec& end_time)
{
    time_t delta = end_time.tv_sec - start_time.tv_sec;
    if (start_time.tv_nsec > end_time.tv_nsec) {
        delta--;
    }
    return static_cast<uint16_t>(delta & 0xFFFF);
}

// The only field that needs to be updated is BootpPacket::secs
void update_bootp_packet(BootpPacket& bootp_packet, const struct timespec& start_time)
{
    // Get current time from POSIX monotonic clock
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    bootp_packet.secs = get_delta_secs(start_time, ts);
}

void init_bootp_packet(BootpPacket& bootp_packet, const std::string& ip_address, HSBConfiguration& vendor_info)
{
    bootp_packet = (BootpPacket) {
        .op = 1,
        .htype = 1,
        .hlen = 6,
        .hops = 0,
        .xid = vendor_info.data_plane
        // all other fields initialized to 0
    };

    auto [local_ip, local_device, local_mac] = local_ip_and_mac(ip_address, BOOTP_REQUEST_PORT);
    hololink::core::Serializer chaddr_serializer(bootp_packet.chaddr, sizeof(bootp_packet.chaddr));
    // MacAddress is 6 bytes (std::array<uint8_t, 6> in hololink/core/networking.hpp). Check that it wrote the whole thing
    if (local_mac.size() != serialize_chaddr(chaddr_serializer, local_mac)) {
        throw std::runtime_error("Failed to initialize chaddr in bootp packet with local mac address");
    }

    hololink::core::Serializer vendor_info_serializer(bootp_packet.vend, sizeof(bootp_packet.vend));
    // serialize_hsb_configuration does not necessarily fill the entire buffer, so only check that it didn't fail
    if (!serialize_hsb_configuration(vendor_info_serializer, vendor_info)) {
        throw std::runtime_error("Failed to initialize vendor information in bootp packet");
    }
}

// for the input parameters, explicitly avoiding sending whole DataPlane object unless needed
void DataPlane::broadcast_bootp()
{
    if (!running_) {
        std::this_thread::sleep_for(std::chrono::seconds(BOOTP_INTERVAL_SEC)); // skip an bootp cycle if CPU/compiler sets running out of order
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
    init_bootp_packet(bootp_packet, IPAddress_to_string(ip_address_), configuration_);
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
    dest_addr.sin_addr.s_addr = htonl(get_broadcast_address(ip_address_));
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
    while (running_) {
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
}

} // namespace hololink::emulation
