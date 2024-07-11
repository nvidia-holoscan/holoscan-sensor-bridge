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
#include "metadata.hpp"

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

#include <hololink/native/deserializer.hpp>
#include <hololink/native/networking.hpp>

#include <holoscan/logger/logger.hpp>

namespace {

/** Hololink-lite data plane configuration is implied by the value
 * passed in the bootp transaction_id field, which is coopted
 * by FPGA to imply which port is publishing the request.  We use
 * that port ID to figure out what the address of the port's
 * configuration data is; which is the value listed here.
 */
struct HololinkChannelConfiguration {
    uint32_t configuration_address;
    uint32_t vip_mask;
};
static const std::map<int, HololinkChannelConfiguration> BOOTP_TRANSACTION_ID_MAP {
    { 0, HololinkChannelConfiguration { 0x02000000, 0x1 } },
    { 1, HololinkChannelConfiguration { 0x02010000, 0x2 } },
};

} // anonynous namespace

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

            sockaddr_in address {};
            address.sin_family = AF_INET;
            address.sin_port = htons(port);
            // host address: '' represents INADDR_ANY
            if (local_interface.empty()) {
                address.sin_addr.s_addr = INADDR_ANY;
            } else {
                if (inet_pton(AF_INET, local_interface.c_str(), &address.sin_addr) == 0) {
                    throw std::runtime_error(
                        fmt::format("Failed to convert address {}", local_interface));
                }
            }

            if (bind(get(), (sockaddr*)&address, sizeof(address)) < 0) {
                throw std::runtime_error(fmt::format("bind failed with errno={}: \"{}\"", errno, strerror(errno)));
            }
        }

        int get() const { return socket_.get(); }

    private:
        native::UniqueFileDescriptor socket_;
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
        native::Deserializer& deserializer, std::string& result, unsigned n)
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

    static void deserialize_ancdata(struct msghdr& msg, Metadata& metadata)
    {
        for (struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg); cmsg != NULL;
             cmsg = CMSG_NXTHDR(&msg, cmsg)) {
            HOLOSCAN_LOG_TRACE(fmt::format("cmsg_level={} cmsg_type={} cmsg_data_len={}",
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

                HOLOSCAN_LOG_TRACE(
                    fmt::format("ipi_ifindex={} interface={} ipi_spec_dst={} ipi_addr={}",
                        metadata["interface_index"], metadata["interface"],
                        metadata["interface_address"], metadata["destination_address"]));
            }
        }
    }

    static void deserialize_enumeration(const std::vector<uint8_t>& packet, Metadata& metadata)
    {
        native::Deserializer deserializer(packet.data(), packet.size());

        uint8_t board_id = 0;
        if (!deserializer.next_uint8(board_id)) {
            throw std::runtime_error("Unable to deserialize enumeration packet.");
        }

        metadata["type"] = "enumeration";
        metadata["board_id"] = board_id;
        if (board_id == HOLOLINK_LITE_BOARD_ID) {
            metadata["board_description"] = "hololink-lite";
        } else if (board_id == HOLOLINK_BOARD_ID) {
            metadata["board_description"] = "hololink";
        } else if (board_id == HOLOLINK_100G_BOARD_ID) {
            metadata["board_description"] = "hololink 100G";
        } else {
            metadata["board_description"] = "N/A";
        }

        if ((board_id == HOLOLINK_LITE_BOARD_ID)
            || (board_id == HOLOLINK_BOARD_ID)
            || (board_id == HOLOLINK_100G_BOARD_ID)) {
            std::string board_version;
            std::string serial_number;
            uint16_t cpnx_version = 0;
            uint16_t cpnx_crc = 0;
            uint16_t clnx_version = 0;
            uint16_t clnx_crc = 0;

            if (!(next_buffer_as_string(deserializer, board_version, 20)
                    && next_buffer_as_string(deserializer, serial_number, 7)
                    && deserializer.next_uint16_le(cpnx_version)
                    && deserializer.next_uint16_le(cpnx_crc)
                    && deserializer.next_uint16_le(clnx_version)
                    && deserializer.next_uint16_le(clnx_crc))) {
                throw std::runtime_error("Unable to deserialize enumeration packet.");
            }

            constexpr int default_control_port = 8192;

            metadata["board_version"] = board_version;
            metadata["serial_number"] = serial_number;
            metadata["cpnx_version"] = cpnx_version;
            metadata["cpnx_crc"] = cpnx_crc;
            metadata["clnx_version"] = clnx_version;
            metadata["clnx_crc"] = clnx_crc;
            metadata["control_port"] = default_control_port;
        }
    }

    static void deserialize_bootp_request(const std::vector<uint8_t>& packet, Metadata& metadata)
    {
        native::Deserializer deserializer(packet.data(), packet.size());

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
            throw std::runtime_error("Unable to deserialize bootp request packet.");
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
        deserializer.pointer(ignore, 64); // vendor information

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
    }

} // anonymous namespace

Enumerator::Enumerator(const std::string& local_interface, uint32_t enumeration_port,
    uint32_t bootp_request_port, uint32_t bootp_reply_port)
    : local_interface_(local_interface)
    , enumeration_port_(enumeration_port)
    , bootp_request_port_(bootp_request_port)
    , bootp_reply_port_(bootp_reply_port)
{
}

/*static*/ void Enumerator::enumerated(
    const std::function<bool(const Metadata&)>& call_back, const std::shared_ptr<Timeout>& timeout)
{
    Enumerator enumerator("");
    std::map<std::string, Metadata> data_plane_by_peer_ip;

    enumerator.enumeration_packets(
        [call_back, &data_plane_by_peer_ip](
            Enumerator&, const std::vector<uint8_t>& packet, const Metadata& metadata) -> bool {
            HOLOSCAN_LOG_DEBUG(fmt::format("Enumeration metadata={}", metadata));
            auto peer_ip = metadata.get<std::string>("peer_ip");
            if (!peer_ip) {
                return true;
            }
            Metadata& channel_metadata = data_plane_by_peer_ip[*peer_ip];
            channel_metadata.update(metadata);
            // transaction_id actually indicates which data plane instance we're talking to
            auto transaction_id = metadata.get<int64_t>("transaction_id"); // may not exist
            if (transaction_id) {
                HOLOSCAN_LOG_TRACE(fmt::format("transaction_id={}", transaction_id.value()));
                auto channel_configuration = BOOTP_TRANSACTION_ID_MAP.find(transaction_id.value());
                if (channel_configuration != BOOTP_TRANSACTION_ID_MAP.cend()) {
                    channel_metadata["configuration_address"]
                        = channel_configuration->second.configuration_address;
                    channel_metadata["vip_mask"] = channel_configuration->second.vip_mask;
                }
            }

            // Do we have the information we need?
            HOLOSCAN_LOG_DEBUG(fmt::format("channel_metadata={}", channel_metadata));
            if (DataChannel::enumerated(channel_metadata)) {
                if (!call_back(channel_metadata)) {
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
        [&channel_ip, &channel_metadata, &found](const Metadata& metadata) -> bool {
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
    const std::function<bool(Enumerator&, const std::vector<uint8_t>&, const Metadata&)>& call_back,
    const std::shared_ptr<Timeout>& timeout)
{
    Socket enumeration_socket(local_interface_, enumeration_port_);
    Socket bootp_socket(local_interface_, bootp_request_port_);

    constexpr size_t receive_message_size = 8192;
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

        std::array<int, 2> fds { enumeration_socket.get(), bootp_socket.get() };
        fd_set r;
        FD_ZERO(&r);
        for (auto&& fd : fds) {
            FD_SET(fd, &r);
        }
        fd_set x = r;
        int num_fds = fds[0];
        for (int index = 1; index < fds.size(); ++index) {
            num_fds = std::max(num_fds, fds[index]);
        }
        num_fds++;
        const int result = select(num_fds, &r, nullptr, &x, select_timeout);
        if (result == -1) {
            if (errno == EINTR) {
                // retry
                continue;
            }
            throw std::runtime_error(fmt::format("select failed with errno={}: \"{}\"", errno, strerror(errno)));
        }
        for (auto&& fd : fds) {
            if (FD_ISSET(fd, &x)) {
                HOLOSCAN_LOG_ERROR("Error reading enumeration sockets.");
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
                if ((received_bytes == -1) && (errno != EINTR)) {
                    throw std::runtime_error(fmt::format("recvmsg failed with errno={}: \"{}\"", errno, strerror(errno)));
                }
            } while (received_bytes <= 0);

            const std::string peer_address_string(inet_ntoa(peer_address.sin_addr));
            HOLOSCAN_LOG_TRACE(fmt::format(
                "enumeration peer_address \"{}:{}\", ancdata size {}, msg_flags {}, packet size {}",
                peer_address_string, ntohs(peer_address.sin_port), msg.msg_controllen, msg.msg_flags, received_bytes));

            Metadata metadata;
            metadata["peer_ip"] = peer_address_string;
            metadata["source_port"] = ntohs(peer_address.sin_port);
            metadata["_socket_fd"] = fd;
            deserialize_ancdata(msg, metadata);

            if (fd == enumeration_socket.get()) {
                deserialize_enumeration(iobuf, metadata);
            } else if (fd == bootp_socket.get()) {
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
            HOLOSCAN_LOG_DEBUG(fmt::format("sendmsg failed with errno={}: \"{}\"", errno, strerror(errno)));
            first = false;
        }
    }
}

} // namespace hololink
