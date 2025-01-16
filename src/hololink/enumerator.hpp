/**
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
 *
 * See README.md for detailed information.
 */

#ifndef SRC_HOLOLINK_ENUMERATOR
#define SRC_HOLOLINK_ENUMERATOR

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "metadata.hpp"
#include "timeout.hpp"

namespace hololink {

/**
 * @brief Handle device discovery and IP address assignment.
 */
class Enumerator {
public:
    /**
     * @brief Construct a new Enumerator object
     *
     * @param local_interface blank for all local interfaces
     * @param bootp_request_port
     * @param bootp_reply_port
     */
    explicit Enumerator(const std::string& local_interface = std::string(),
        uint32_t bootp_request_port = 12267u,
        uint32_t bootp_reply_port = 12268u);
    Enumerator() = delete;

    /**
     * @brief Calls the provided call back function with metadata about the connection for every
     * enumeration message received. Note that if data changes, e.g. board reset, this routine won't
     * know to invalidate the old data; so you may get one or two stale messages around reset.
     *
     * @param call_back
     * @param timeout
     */
    static void enumerated(const std::function<bool(Metadata&)>& call_back,
        const std::shared_ptr<Timeout>& timeout = std::shared_ptr<Timeout>());

    /**
     * @brief
     *
     * @param channel_ip
     * @param timeout
     * @return Metadata&
     */
    static Metadata find_channel(const std::string& channel_ip,
        const std::shared_ptr<Timeout>& timeout = std::make_shared<Timeout>(20.f));

    /**
     * @brief Calls the provided call back function a pair of (packet, metadata) about the
     * connection for each received enumeration or bootp request packet. This function terminates
     * after the given timeout; provide no timeout to run forever.
     *
     * @param call_back
     * @param timeout
     */
    void enumeration_packets(
        const std::function<bool(Enumerator&, const std::vector<uint8_t>&, Metadata&)>& call_back,
        const std::shared_ptr<Timeout>& timeout = std::shared_ptr<Timeout>());

    /**
     *
     */
    void send_bootp_reply(
        const std::string& peer_address,
        const std::string& reply_packet,
        Metadata& metadata);

private:
    const std::string local_interface_;
    const uint32_t bootp_request_port_;
    const uint32_t bootp_reply_port_;
};

} // namespace hololink

#endif /* SRC_HOLOLINK_ENUMERATOR */
