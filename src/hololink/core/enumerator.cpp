/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 */

#include "enumerator.hpp"

#include "data_channel.hpp"
#include "deserializer.hpp"
#include "logging_internal.hpp"
#include "metadata.hpp"
#include "networking.hpp"

#include <cmath>
#include <iomanip>
#include <memory>
#include <ratio>
#include <sstream>
#include <stdexcept>
#include <string>

#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace hololink {

namespace {

    class Socket {
    public:
        Socket(const std::string& local_interface, uint32_t port)
        {
            socket_.reset(socket(AF_INET, SOCK_DGRAM, 0));
            if (!socket_) {
                throw std::runtime_error("Failed to create socket");
            }

            // Allow other programs to receive these broadcast packets.
            const int reuse_port = 1;
            if (setsockopt(get(), SOL_SOCKET, SO_REUSEPORT, &reuse_port, sizeof(reuse_port)) < 0) {
                throw std::runtime_error(fmt::format("setsockopt failed with errno={}: \"{}\"", errno, strerror(errno)));
            }

            // Tell us what interface the request came in on, so we reply to the same place
            const int ptk_info = 1;
            if (setsockopt(get(), SOL_IP, IP_PKTINFO, &ptk_info, sizeof(ptk_info)) < 0) {
                throw std::runtime_error(fmt::format("setsockopt failed with errno={}: \"{}\"", errno, strerror(errno)));
            }

            // Maybe listen only on the named interface.
            if (!local_interface.empty()) {
                struct ifreq ifr = {};
                strncpy(ifr.ifr_name, local_interface.c_str(), sizeof(ifr.ifr_name));
                if (setsockopt(get(), SOL_SOCKET, SO_BINDTODEVICE, &ifr, sizeof(ifr)) < 0) {
                    throw std::runtime_error(fmt::format("setsockopt SO_BINDTODEVICE failed with errno={}: \"{}\"", errno, strerror(errno)));
                }
            }

            sockaddr_in address {};
            address.sin_family = AF_INET;
            address.sin_port = htons(port);
            address.sin_addr.s_addr = INADDR_ANY;

            if (bind(get(), (sockaddr*)&address, sizeof(address)) < 0) {
                throw std::runtime_error(fmt::format("bind failed with errno={}: \"{}\"", errno, strerror(errno)));
            }
        }

        int get() const { return socket_.get(); }

    private:
        core::UniqueFileDescriptor socket_;
    };

    /**
     * Get a buffer from the deserializer and return it as a hex string.
     *
     * @param deserializer
     * @param result hex string
     * @param n length of the buffer to read
     *
     * @returns true if result is set; false on buffer overflow.
     */
    static bool next_buffer_as_string(
        core::Deserializer& deserializer, std::string& result, unsigned n)
    {
        const uint8_t* buffer = nullptr;
        if (!deserializer.pointer(buffer, n)) {
            return false;
        }
        std::stringstream stream;
        for (unsigned i = 0; i < n; ++i) {
            stream << std::setfill('0') << std::setw(2) << std::hex << int(buffer[i]);
        }
        result = stream.str();
        return true;
    }

    /**
     * Get the next 16 bytes from the deserializer and return it as a UUID-formatted string.
     *
     * @param deserializer
     * @param result UUID formatted string
     *
     * @returns true if result is set; false on buffer overflow.
     */
    static bool next_uuid_as_string(core::Deserializer& deserializer, std::string& result)
    {
        const uint8_t* buffer = nullptr;
        if (!deserializer.pointer(buffer, 16)) {
            return false;
        }
        std::stringstream stream;
        // Format: 8-4-4-4-12 characters
        for (unsigned i = 0; i < 16; ++i) {
            stream << std::setfill('0') << std::setw(2) << std::hex << int(buffer[i]);
            // Add dashes in the appropriate spots
            if (i == 3 || i == 5 || i == 7 || i == 9) {
                stream << "-";
            }
        }
        result = stream.str();
        return true;
    }

    static void deserialize_ancdata(struct msghdr& msg, Metadata& metadata)
    {
        for (struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg); cmsg != NULL;
             cmsg = CMSG_NXTHDR(&msg, cmsg)) {
            HSB_LOG_TRACE(fmt::format("cmsg_level={} cmsg_type={} cmsg_data_len={}",
                cmsg->cmsg_level, cmsg->cmsg_type, cmsg->cmsg_len));

            if ((cmsg->cmsg_level == IPPROTO_IP) && (cmsg->cmsg_type == IP_PKTINFO)) {
                in_pktinfo* pkt_info = (in_pktinfo*)CMSG_DATA(cmsg);
                char interface[IF_NAMESIZE + 1];

                if (if_indextoname(pkt_info->ipi_ifindex, interface) == 0) {
                    throw std::runtime_error(
                        fmt::format("if_indextoname failed with errno={}: \"{}\"", errno, strerror(errno)));
                }

                metadata["interface_index"] = pkt_info->ipi_ifindex;
                metadata["interface"] = std::string(interface),
                metadata["interface_address"] = std::string(inet_ntoa(pkt_info->ipi_spec_dst)),
                metadata["destination_address"] = std::string(inet_ntoa(pkt_info->ipi_addr));

                HSB_LOG_TRACE(
                    fmt::format("ipi_ifindex={} interface={} ipi_spec_dst={} ipi_addr={}",
                        metadata["interface_index"], metadata["interface"],
                        metadata["interface_address"], metadata["destination_address"]));
            }
        }
    }

    /**
     * Older BOOTP vendor data doesn't include UUIDs, so emulate that here.
     */
    static void deserialize_bootp_v1(core::Deserializer& deserializer, Metadata& metadata)
    {
        uint16_t board_id = 0;
        std::string board_version;
        std::string serial_number;
        uint16_t hsb_ip_version = 0;
        uint16_t fpga_crc = 0;

        if (!(deserializer.next_uint16_le(board_id)
                && next_buffer_as_string(deserializer, board_version, 20)
                && next_buffer_as_string(deserializer, serial_number, 7)
                && deserializer.next_uint16_le(hsb_ip_version)
                && deserializer.next_uint16_le(fpga_crc))) {
            // Don't flood the log.
            static unsigned reports = 0;
            if (reports < 5) {
                HSB_LOG_ERROR("Unable to deserialize bootp request board data.");
                reports++;
            }
            return;
        }

        metadata["board_id"] = board_id;
        metadata["board_version"] = board_version;
        metadata["serial_number"] = serial_number;
        metadata["hsb_ip_version"] = hsb_ip_version;
        metadata["fpga_crc"] = fpga_crc;

        // In enum-v1, we don't get fpga-uuid, so let's emulate that here.
        if (board_id == HOLOLINK_LITE_BOARD_ID) {
            metadata["fpga_uuid"] = HOLOLINK_LITE_UUID;
        } else if (board_id == HOLOLINK_NANO_BOARD_ID) {
            metadata["fpga_uuid"] = HOLOLINK_NANO_UUID;
        } else if (board_id == HOLOLINK_100G_BOARD_ID) {
            metadata["fpga_uuid"] = HOLOLINK_100G_UUID;
        } else if (board_id == MICROCHIP_POLARFIRE_BOARD_ID) {
            metadata["fpga_uuid"] = MICROCHIP_POLARFIRE_UUID;
        } else if (board_id == LEOPARD_EAGLE_BOARD_ID) {
            metadata["fpga_uuid"] = LEOPARD_EAGLE_UUID;
        }
    }

    static void deserialize_bootp_v2(core::Deserializer& deserializer, Metadata& metadata)
    {
        uint16_t reserved_1 = 0;
        std::string fpga_uuid;
        std::string serial_number;
        uint32_t reserved_2 = 0;
        uint16_t hsb_ip_version = 0;
        uint16_t fpga_crc = 0;

        if (!(deserializer.next_uint16_le(reserved_1)
                && next_uuid_as_string(deserializer, fpga_uuid)
                && deserializer.next_uint32_le(reserved_2)
                && next_buffer_as_string(deserializer, serial_number, 7)
                && deserializer.next_uint16_le(hsb_ip_version)
                && deserializer.next_uint16_le(fpga_crc))) {
            // Don't flood the log.
            static unsigned reports = 0;
            if (reports < 5) {
                HSB_LOG_ERROR("Unable to deserialize bootp request board data.");
                reports++;
            }
            return;
        }

        metadata["fpga_uuid"] = fpga_uuid;
        metadata["serial_number"] = serial_number;
        metadata["hsb_ip_version"] = hsb_ip_version;
        metadata["fpga_crc"] = fpga_crc;
    }

    static void deserialize_bootp_request(const std::vector<uint8_t>& packet, Metadata& metadata)
    {
        core::Deserializer deserializer(packet.data(), packet.size());

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
                && deserializer.next_uint32_be(client_ip_address) && // current IP address
                deserializer.next_uint32_be(your_ip_address)
                && // host IP that assigned the IP address
                deserializer.next_uint32_be(server_ip_address) && // expected to be 0s
                deserializer.next_uint32_be(gateway_ip_address)
                && deserializer.next_buffer(hardware_address))) {
            // Don't flood the log.
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

        //  Vendor information has more for us.
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
            // Don't flood the log.
            static unsigned reports = 0;
            if (reports < 5) {
                HSB_LOG_ERROR("Unable to deserialize bootp request vendor data.");
                reports++;
            }
            return;
        }

        metadata["type"] = "bootp_request";
        metadata["op"] = op;
        metadata["hardware_type"] = hardware_type;
        metadata["hardware_address_length"] = hardware_address_length;
        metadata["hops"] = hops;
        metadata["transaction_id"] = transaction_id;
        metadata["seconds"] = seconds;
        metadata["flags"] = flags;
        metadata["client_ip_address"] = std::string(inet_ntoa({ ntohl(client_ip_address) }));
        metadata["your_ip_address"] = std::string(inet_ntoa({ ntohl(your_ip_address) }));
        metadata["server_ip_address"] = std::string(inet_ntoa({ ntohl(server_ip_address) }));
        metadata["gateway_ip_address"] = std::string(inet_ntoa({ ntohl(gateway_ip_address) }));
        metadata["hardware_address"] = hardware_address;
        metadata["mac_id"] = mac_id;
        metadata["data_plane"] = static_cast<int64_t>(data_plane);
        metadata["enum_version"] = enum_version;

        constexpr int default_control_port = 8192;
        metadata["control_port"] = default_control_port;
        metadata["sequence_number_checking"] = 1;

        if (enum_version == 1) {
            deserialize_bootp_v1(deserializer, metadata);
        } else if (enum_version == 2) {
            deserialize_bootp_v2(deserializer, metadata);
        }

        // At this point, metadata["fpga_uuid"] can be used to update configurations.
        // Note that for now, we're just accommodating the HSB enum-v2 BOOTP format;
        // this will become an extensible framework in an upcoming
        // release.
        metadata["board_description"] = "N/A";
        auto fpga_uuid_value = metadata.get<std::string>("fpga_uuid");
        if (fpga_uuid_value) {
            std::string fpga_uuid = fpga_uuid_value.value();
            if (fpga_uuid == "889b7ce3-65a5-4247-8b05-4ff1904c3359") {
                // HOLOLINK_LITE_BOARD_ID
                metadata["board_description"] = "hololink-lite";
                metadata["gpio_pin_count"] = 16;
            } else if (fpga_uuid == "d0f015e0-93b6-4473-b7d1-7dbd01cbeab5") {
                // HOLOLINK_NANO_BOARD_ID
                metadata["board_description"] = "hololink-nano";
                metadata["gpio_pin_count"] = 54;
            } else if (fpga_uuid == "7a377bf7-76cb-4756-a4c5-7dddaed8354b") {
                // HOLOLINK_100G_BOARD_ID
                metadata["board_description"] = "hololink 100G";
            } else if (fpga_uuid == "ed6a9292-debf-40ac-b603-a24e025309c1") {
                // MICROCHIP_POLARFIRE_BOARD_ID
                metadata["board_description"] = "Microchip Polarfire";
            } else if (fpga_uuid == "f1627640-b4dc-48af-a360-c55b09b3d230") {
                // LEOPARD_EAGLE_BOARD_ID
                metadata["board_description"] = "Leopard Eagle";
            }
        }
    }

} // anonymous namespace

Enumerator::Enumerator(const std::string& local_interface,
    uint32_t bootp_request_port, uint32_t bootp_reply_port)
    : local_interface_(local_interface)
    , bootp_request_port_(bootp_request_port)
    , bootp_reply_port_(bootp_reply_port)
{
}

/*static*/ void Enumerator::enumerated(
    const std::function<bool(Metadata&)>& call_back, const std::shared_ptr<Timeout>& timeout)
{
    Enumerator enumerator("");

    enumerator.enumeration_packets(
        [call_back](
            Enumerator&, const std::vector<uint8_t>& packet, Metadata& metadata) -> bool {
            HSB_LOG_DEBUG(fmt::format("Enumeration metadata={}", metadata));
            auto peer_ip = metadata.get<std::string>("peer_ip");
            if (!peer_ip) {
                return true;
            }
            // Add some supplemental data.
            auto opt_data_plane = metadata.get<int64_t>("data_plane");
            if (!opt_data_plane.has_value()) {
                // 2410 and later always provide this.
                return true;
            }
            int data_plane = opt_data_plane.value();
            DataChannel::use_data_plane_configuration(metadata, data_plane);
            // By default, use the data_plane ID to select the sensor.
            int sensor_number = data_plane;
            DataChannel::use_sensor(metadata, sensor_number);
            // Do we have the information we need?
            HSB_LOG_DEBUG(fmt::format("metadata={}", metadata));
            if (DataChannel::enumerated(metadata)) {
                if (!call_back(metadata)) {
                    return false;
                }
            }
            return true;
        },
        timeout);
}

/*static*/ Metadata Enumerator::find_channel(
    const std::string& channel_ip, const std::shared_ptr<Timeout>& timeout)
{
    Metadata channel_metadata;
    bool found = false;

    enumerated(
        [&channel_ip, &channel_metadata, &found](Metadata& metadata) -> bool {
            auto peer_ip = metadata.get<std::string>("peer_ip");
            if (peer_ip && (peer_ip == channel_ip)) {
                channel_metadata = metadata;
                found = true;
                return false;
            }
            return true;
        },
        timeout);

    if (found) {
        return channel_metadata;
    }

    // We only get here if we time out or channel had not been found.
    throw std::runtime_error(fmt::format("Device with {} not found.", channel_ip));
}

void Enumerator::enumeration_packets(
    const std::function<bool(Enumerator&, const std::vector<uint8_t>&, Metadata&)>& call_back,
    const std::shared_ptr<Timeout>& timeout)
{
    Socket bootp_socket(local_interface_, bootp_request_port_);

    constexpr size_t receive_message_size = hololink::core::UDP_PACKET_SIZE;
    std::vector<uint8_t> iobuf(receive_message_size);
    std::array<uint8_t, CMSG_ALIGN(receive_message_size)> controlbuf;

    timeval timeout_value {};
    bool done = false;
    while (!done) {
        // if there had been a timeout provided check if it had expired, if not set the timeout of
        // the select() call to the remaining time
        timeval* select_timeout;
        if (timeout) {
            if (timeout->expired()) {
                return;
            }
            // convert floating point time to integer seconds and microseconds
            const double trigger_s = timeout->trigger_s();
            double integral = 0.f;
            double fractional = std::modf(trigger_s, &integral);
            timeout_value.tv_sec = (__time_t)(integral);
            timeout_value.tv_usec
                = (__suseconds_t)(fractional * std::micro().den / std::micro().num);
            select_timeout = &timeout_value;
        } else {
            select_timeout = nullptr;
        }

        std::array<int, 1> fds { { bootp_socket.get() } };
        fd_set r;
        FD_ZERO(&r);
        for (auto&& fd : fds) {
            FD_SET(fd, &r);
        }
        fd_set x = r;
        int num_fds = fds[0];
        for (size_t index = 1; index < fds.size(); ++index) {
            num_fds = std::max(num_fds, fds[index]);
        }
        num_fds++;
        const int result = select(num_fds, &r, nullptr, &x, select_timeout);
        if (result == -1) {
            throw std::runtime_error(fmt::format("select failed with errno={}: \"{}\"", errno, strerror(errno)));
        }
        for (auto&& fd : fds) {
            if (FD_ISSET(fd, &x)) {
                HSB_LOG_ERROR("Error reading enumeration sockets.");
                return;
            }
        }
        if (result == 0) {
            // Timed out
            return;
        }

        for (auto&& fd : fds) {
            if (!FD_ISSET(fd, &r)) {
                continue;
            }

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

            ssize_t received_bytes;
            do {
                received_bytes = recvmsg(fd, &msg, 0);
                if (received_bytes == -1) {
                    throw std::runtime_error(fmt::format("recvmsg failed with errno={}: \"{}\"", errno, strerror(errno)));
                }
            } while (received_bytes <= 0);

            const std::string peer_address_string(inet_ntoa(peer_address.sin_addr));
            HSB_LOG_TRACE(fmt::format(
                "enumeration peer_address \"{}:{}\", ancdata size {}, msg_flags {}, packet size {}",
                peer_address_string, ntohs(peer_address.sin_port), msg.msg_controllen, msg.msg_flags, received_bytes));

            Metadata metadata;
            metadata["peer_ip"] = peer_address_string;
            metadata["source_port"] = ntohs(peer_address.sin_port);
            metadata["_socket_fd"] = fd;
            deserialize_ancdata(msg, metadata);

            if (fd == bootp_socket.get()) {
                deserialize_bootp_request(iobuf, metadata);
            }

            if (!call_back(*this, iobuf, metadata)) {
                done = true;
                break;
            }
        }
    }
}

void Enumerator::send_bootp_reply(
    const std::string& peer_address,
    const std::string& reply_packet,
    Metadata& metadata)
{
    auto socket_fd = metadata.get<int64_t>("_socket_fd");
    if (!socket_fd) {
        throw std::runtime_error("Metadata is missing the _socket_fd element.");
    }
    auto interface_index = metadata.get<int64_t>("interface_index");
    if (!interface_index) {
        throw std::runtime_error("Metadata is missing the interface_index element.");
    }
    // reply_packet is const but iovec isn't; work around that
    std::string reply(reply_packet);
    struct iovec iov = {
        .iov_base = reply.data(),
        .iov_len = reply.size(),
    };
    // which port to send this message out?
    char adata[256] = { 0 }; // C fills the rest with 0s.
    struct cmsghdr* cmsghdr = (struct cmsghdr*)adata;
    struct in_pktinfo* pktinfo = (struct in_pktinfo*)CMSG_DATA(cmsghdr);
    static_assert(CMSG_SPACE(sizeof(pktinfo[0])) <= sizeof(adata));
    cmsghdr->cmsg_len = CMSG_LEN(sizeof(pktinfo[0]));
    cmsghdr->cmsg_level = IPPROTO_IP;
    cmsghdr->cmsg_type = IP_PKTINFO;
    pktinfo->ipi_ifindex = interface_index.value();
    // (rest of pktinfo is 0).
    // who do we send this to?
    struct sockaddr_in sin = {
        .sin_family = AF_INET,
        .sin_port = htons(bootp_reply_port_),
    };
    if (inet_aton(peer_address.c_str(), &sin.sin_addr) != 1) {
        throw std::runtime_error(fmt::format("inet_aton failed with peer_address={}", peer_address));
    }
    // send it
    struct msghdr message = {
        .msg_name = &sin,
        .msg_namelen = sizeof(sin),
        .msg_iov = &iov,
        .msg_iovlen = 1,
        .msg_control = adata,
        .msg_controllen = CMSG_SPACE(sizeof(pktinfo[0])),
    };
    int flags = 0;
    ssize_t sendmsg_sent = sendmsg(socket_fd.value(), &message, flags);
    if (sendmsg_sent == -1) {
        // Don't flood the console.
        static bool first = true;
        if (first) {
            HSB_LOG_DEBUG(fmt::format("sendmsg failed with errno={}: \"{}\"", errno, strerror(errno)));
            first = false;
        }
    }
}

} // namespace hololink
