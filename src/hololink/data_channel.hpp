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

#ifndef SRC_HOLOLINK_DATA_CHANNEL
#define SRC_HOLOLINK_DATA_CHANNEL

#include <memory>
#include <string>

#include "hololink.hpp"
#include "metadata.hpp"

#include <hololink/native/networking.hpp>

namespace hololink {

// Note that these are offsets enumeration metadata "configuration_address".
constexpr uint32_t DP_PACKET_SIZE = 0x304;
constexpr uint32_t DP_VIP_MASK = 0x30C;

// DMA descriptor registers.
constexpr uint32_t DP_QP = 0x1000;
constexpr uint32_t DP_RKEY = 0x1004;
// these are all page addresses; the actual byte address in the
// packet will be this page address * 128.
constexpr uint32_t DP_ADDRESS_0 = 0x1008;
constexpr uint32_t DP_ADDRESS_1 = 0x100C;
constexpr uint32_t DP_ADDRESS_2 = 0x1010;
constexpr uint32_t DP_ADDRESS_3 = 0x1014;
constexpr uint32_t DP_BUFFER_LENGTH = 0x1018; // this is in bytes
constexpr uint32_t DP_BUFFER_MASK = 0x101C; // each bit enables a buffer
constexpr uint32_t DP_HOST_MAC_LOW = 0x1020;
constexpr uint32_t DP_HOST_MAC_HIGH = 0x1024;
constexpr uint32_t DP_HOST_IP = 0x1028;
constexpr uint32_t DP_HOST_UDP_PORT = 0x102C;

class Enumerator;

class DataChannel {
    // Enumerator calls our methods to reconfigure Metadata.
    friend class Enumerator;

public:
    /**
     * @brief Construct a new DataChannel object
     *
     * @param metadata
     */
    DataChannel(const Metadata& metadata,
        const std::function<std::shared_ptr<Hololink>(const Metadata& metadata)>& create_hololink
        = Hololink::from_enumeration_metadata);

    /**
     * @brief
     *
     * @param metadata
     * @return true
     * @return false
     */
    static bool enumerated(const Metadata& metadata);

    /**
     * Update the enumeration metadata to use a multicast destination address.
     */
    static void use_multicast(Metadata& metadata, std::string address, uint16_t port);

    /**
     * Update the enumeration metadata to use a broadcast destination address.
     */
    static void use_broadcast(Metadata& metadata, uint16_t port);

    /**
     * @brief
     *
     * @return std::shared_ptr<Hololink>
     */
    std::shared_ptr<Hololink> hololink() const;

    /**
     * @brief
     *
     * @return std::string
     */
    const std::string& peer_ip() const;

    /**
     * @brief
     *
     * @param qp_number
     * @param rkey
     */
    void authenticate(uint32_t qp_number, uint32_t rkey);

    /**
     * @brief
     *
     * @param frame_address
     * @param frame_size
     * @param local_data_port
     */
    void configure(uint64_t frame_memory, size_t frame_size, size_t page_size, unsigned pages, uint32_t local_data_port);

    /**
     * Clear the operating state set up by configure().
     */
    void unconfigure();

    /**
     * Configure the receiver to handle this traffic; this
     * is useful when using, say, multicast.
     */
    void configure_socket(int socket_fd);

    /**
     * Configure the given metadata to exchange data from the
     * given sensor port.  Given enumeration data for a specific
     * IP address, use this call to configure the sensor to listen
     * to.  Multiple sensors can transmit using the same IP address.
     */
    static void use_sensor(Metadata& metadata, int64_t sensor_number);

protected:
    /**
     * Configure the given metadata to send data to the given
     * host interface.
     */
    static void use_data_plane_configuration(Metadata& metadata, int64_t data_plane);

private:
    std::shared_ptr<Hololink> hololink_;
    uint32_t network_configuration_address_;
    uint32_t sensor_configuration_address_;
    std::string peer_ip_;
    uint32_t vip_mask_;
    uint32_t qp_number_;
    uint32_t rkey_;
    uint32_t data_plane_;
    uint32_t sensor_;
    std::string multicast_;
    uint16_t multicast_port_;
    uint16_t broadcast_port_;
};

} // namespace hololink

#endif /* SRC_HOLOLINK_DATA_CHANNEL */
