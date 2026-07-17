/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bootp.hpp"

#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#include <fmt/core.h>

#include "deserializer.hpp"
#include "hololink/module/logging.hpp" // HSB_LOG_*

namespace hololink::module {

// Large enough for a 9k jumbo bootp datagram.
static constexpr size_t UDP_PACKET_SIZE = 10240;
// The control port the device listens on; the bootp payload doesn't
// carry it, so it's a fixed default the legacy enumeration also used.
static constexpr int64_t DEFAULT_CONTROL_PORT = 8192;

// Read n bytes and render them as a lowercase hex string; false on overflow.
static bool next_buffer_as_string(Deserializer& deserializer, std::string& result, unsigned n)
{
    const uint8_t* buffer = nullptr;
    if (!deserializer.pointer(buffer, n)) {
        return false;
    }
    std::stringstream stream;
    for (unsigned i = 0; i < n; ++i) {
        stream << std::setfill('0') << std::setw(2) << std::hex << int(buffer[i]) << std::dec;
    }
    result = stream.str();
    return true;
}

// Read 16 bytes and render them in 8-4-4-4-12 UUID form; false on overflow.
static bool next_uuid_as_string(Deserializer& deserializer, std::string& result)
{
    const uint8_t* buffer = nullptr;
    if (!deserializer.pointer(buffer, 16)) {
        return false;
    }
    std::stringstream stream;
    for (unsigned i = 0; i < 16; ++i) {
        stream << std::setfill('0') << std::setw(2) << std::hex << int(buffer[i]) << std::dec;
        if (i == 3 || i == 5 || i == 7 || i == 9) {
            stream << "-";
        }
    }
    result = stream.str();
    return true;
}

// Pull the arrival-interface details out of the recvmsg ancillary data.
static void deserialize_ancdata(struct msghdr& msg, EnumerationMetadata& metadata)
{
    for (struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg); cmsg != nullptr;
         cmsg = CMSG_NXTHDR(&msg, cmsg)) {
        if ((cmsg->cmsg_level == IPPROTO_IP) && (cmsg->cmsg_type == IP_PKTINFO)) {
            in_pktinfo* pkt_info = (in_pktinfo*)CMSG_DATA(cmsg);
            char interface[IF_NAMESIZE + 1];
            if (if_indextoname(pkt_info->ipi_ifindex, interface) == nullptr) {
                throw std::runtime_error(fmt::format(
                    "if_indextoname failed with errno={}: \"{}\"", errno, strerror(errno)));
            }
            metadata["interface_index"] = static_cast<int64_t>(pkt_info->ipi_ifindex);
            metadata["interface"] = std::string(interface);
            metadata["interface_address"] = std::string(inet_ntoa(pkt_info->ipi_spec_dst));
            metadata["destination_address"] = std::string(inet_ntoa(pkt_info->ipi_addr));
        }
    }
}

// Parse the bootp v2 board body (the only enumeration version the
// module targets). compat_id / fpga_uuid drive module resolution.
static void deserialize_bootp_v2(Deserializer& deserializer, EnumerationMetadata& metadata)
{
    uint16_t compat_id = 0;
    std::string fpga_uuid;
    std::string serial_number;
    uint32_t transmitted_packet_count = 0;
    uint16_t hsb_ip_version = 0;
    uint16_t fpga_crc = 0;
    uint8_t ignored = 0;

    if (!(deserializer.next_uint16_le(compat_id)
            && next_uuid_as_string(deserializer, fpga_uuid)
            && deserializer.next_uint24_be(transmitted_packet_count)
            && deserializer.next_uint8(ignored)
            && next_buffer_as_string(deserializer, serial_number, 7)
            && deserializer.next_uint16_le(hsb_ip_version)
            && deserializer.next_uint16_le(fpga_crc))) {
        static unsigned reports = 0;
        if (reports < 5) {
            HSB_LOG_ERROR("Unable to deserialize bootp request board data.");
            reports++;
        }
        return;
    }

    metadata["compat_id"] = static_cast<int64_t>(compat_id);
    metadata["fpga_uuid"] = fpga_uuid;
    metadata["transmitted_packet_count"] = static_cast<int64_t>(transmitted_packet_count);
    metadata["serial_number"] = serial_number;
    metadata["hsb_ip_version"] = static_cast<int64_t>(hsb_ip_version);
    metadata["fpga_crc"] = static_cast<int64_t>(fpga_crc);
}

// Parse the bootp header + NVDA vendor section into metadata.
static void deserialize_bootp_request(
    const std::vector<uint8_t>& packet, EnumerationMetadata& metadata)
{
    Deserializer deserializer(packet.data(), packet.size());

    uint8_t op = 0;
    uint8_t hardware_type = 0;
    uint8_t hardware_address_length = 0;
    uint8_t hops = 0;
    uint32_t transaction_id = 0;
    uint16_t seconds = 0;
    uint16_t flags = 0;
    uint32_t client_ip_address = 0;
    uint32_t your_ip_address = 0;
    uint32_t server_ip_address = 0;
    uint32_t gateway_ip_address = 0;
    std::vector<uint8_t> hardware_address(16);

    if (!(deserializer.next_uint8(op) && deserializer.next_uint8(hardware_type)
            && deserializer.next_uint8(hardware_address_length)
            && (hardware_address_length <= 16)
            && deserializer.next_uint8(hops)
            && deserializer.next_uint32_be(transaction_id)
            && deserializer.next_uint16_be(seconds) && deserializer.next_uint16_be(flags)
            && deserializer.next_uint32_be(client_ip_address)
            && deserializer.next_uint32_be(your_ip_address)
            && deserializer.next_uint32_be(server_ip_address)
            && deserializer.next_uint32_be(gateway_ip_address)
            && deserializer.next_buffer(hardware_address))) {
        static unsigned reports = 0;
        if (reports < 5) {
            HSB_LOG_ERROR("Unable to deserialize bootp request packet.");
            reports++;
        }
        return;
    }

    std::stringstream mac_id_stream;
    for (int i = 0; i < hardware_address_length; ++i) {
        if (i) {
            mac_id_stream << ":";
        }
        mac_id_stream << fmt::format("{:02X}", hardware_address[i]);
    }
    const std::string mac_id = mac_id_stream.str();

    const uint8_t* ignore = nullptr;
    deserializer.pointer(ignore, 64); // server_hostname
    deserializer.pointer(ignore, 128); // boot_filename

    // The vendor section carries the data-plane index and enumeration version.
    constexpr uint8_t expected_vendor_tag = 0xE0;
    uint8_t vendor_tag = 0;
    uint8_t vendor_tag_length = 0;
    constexpr uint32_t expected_vendor_id = 0x4E564441; // 'NVDA'
    uint32_t vendor_id = 0;
    uint8_t data_plane = 0;
    uint8_t enum_version = 0;
    if (!(deserializer.next_uint8(vendor_tag)
            && (vendor_tag == expected_vendor_tag)
            && deserializer.next_uint8(vendor_tag_length)
            && deserializer.next_uint32_be(vendor_id)
            && (vendor_id == expected_vendor_id)
            && deserializer.next_uint8(data_plane)
            && deserializer.next_uint8(enum_version))) {
        static unsigned reports = 0;
        if (reports < 5) {
            HSB_LOG_ERROR("Unable to deserialize bootp request vendor data.");
            reports++;
        }
        return;
    }

    metadata["type"] = std::string("bootp_request");
    metadata["op"] = static_cast<int64_t>(op);
    metadata["hardware_type"] = static_cast<int64_t>(hardware_type);
    metadata["hardware_address_length"] = static_cast<int64_t>(hardware_address_length);
    metadata["hops"] = static_cast<int64_t>(hops);
    metadata["transaction_id"] = static_cast<int64_t>(transaction_id);
    metadata["seconds"] = static_cast<int64_t>(seconds);
    metadata["flags"] = static_cast<int64_t>(flags);
    metadata["client_ip_address"] = std::string(inet_ntoa({ ntohl(client_ip_address) }));
    metadata["your_ip_address"] = std::string(inet_ntoa({ ntohl(your_ip_address) }));
    metadata["server_ip_address"] = std::string(inet_ntoa({ ntohl(server_ip_address) }));
    metadata["gateway_ip_address"] = std::string(inet_ntoa({ ntohl(gateway_ip_address) }));
    metadata["hardware_address"] = hardware_address;
    metadata["mac_id"] = mac_id;
    metadata["data_plane"] = static_cast<int64_t>(data_plane);
    metadata["enum_version"] = static_cast<int64_t>(enum_version);

    metadata["control_port"] = DEFAULT_CONTROL_PORT;
    metadata["sequence_number_checking"] = static_cast<int64_t>(1);

    // Devices the module targets enumerate with version 2. Older
    // versions are recognized but carry no UUID we can dispatch on; the
    // module's update_metadata supplies any device-specific enrichment.
    metadata["fpga_uuid"] = std::string("N/A");
    if (enum_version == 2) {
        deserialize_bootp_v2(deserializer, metadata);
    }
}

bool configure_bootp_socket(int fd, uint32_t port)
{
    // Learn which interface each request arrived on.
    const int pkt_info = 1;
    if (setsockopt(fd, SOL_IP, IP_PKTINFO, &pkt_info, sizeof(pkt_info)) < 0) {
        throw std::runtime_error(
            fmt::format("setsockopt IP_PKTINFO failed with errno={}: \"{}\"", errno, strerror(errno)));
    }

    sockaddr_in address {};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = INADDR_ANY;
    if (bind(fd, (sockaddr*)&address, sizeof(address)) < 0) {
        throw std::runtime_error(
            fmt::format("bind failed with errno={}: \"{}\"", errno, strerror(errno)));
    }

    return true;
}

std::pair<EnumerationMetadata, std::vector<uint8_t>> receive_bootp(int fd)
{
    std::vector<uint8_t> iobuf(UDP_PACKET_SIZE);
    std::array<uint8_t, 512> controlbuf {};

    struct msghdr msg { };

    sockaddr_in peer_address {};
    peer_address.sin_family = AF_UNSPEC;
    msg.msg_name = &peer_address;
    msg.msg_namelen = sizeof(peer_address);

    iovec iov {};
    iov.iov_base = iobuf.data();
    iov.iov_len = iobuf.size();
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;

    msg.msg_control = controlbuf.data();
    msg.msg_controllen = controlbuf.size();

    const ssize_t received_bytes = recvmsg(fd, &msg, 0);
    if (received_bytes < 0) {
        throw std::runtime_error(
            fmt::format("recvmsg failed with errno={}: \"{}\"", errno, strerror(errno)));
    }
    iobuf.resize(static_cast<std::size_t>(received_bytes));

    char peer_addr_buf[INET_ADDRSTRLEN];
    const std::string peer_address_string(
        inet_ntop(AF_INET, &peer_address.sin_addr, peer_addr_buf, sizeof(peer_addr_buf)));

    EnumerationMetadata metadata;
    metadata["peer_ip"] = peer_address_string;
    metadata["source_port"] = static_cast<int64_t>(ntohs(peer_address.sin_port));
    metadata["_socket_fd"] = static_cast<int64_t>(fd);

    if (received_bytes == 0) {
        // A zero-length datagram carries no bootp request. Return the peer
        // metadata without an fpga_uuid so the caller skips it, rather than
        // looping on recvmsg and stalling the reactor thread.
        HSB_LOG_DEBUG("Received zero-length datagram from {}.", peer_address_string);
        return { std::move(metadata), std::move(iobuf) };
    }

    deserialize_ancdata(msg, metadata);
    deserialize_bootp_request(iobuf, metadata);

    return { std::move(metadata), std::move(iobuf) };
}

} // namespace hololink::module
