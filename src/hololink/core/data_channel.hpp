/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "networking.hpp"

namespace hololink {

// HIF: Packet metadata
constexpr uint32_t DP_PACKET_SIZE = 0x04;
constexpr uint32_t DP_PACKET_UDP_PORT = 0x08;
constexpr uint32_t DP_VP_MASK = 0x0C;

// VP: DMA descriptor registers.
constexpr uint32_t DP_QP = 0x00;
constexpr uint32_t DP_RKEY = 0x04;
// these are all page addresses; the actual byte address in the
// packet will be this page address * 128.
constexpr uint32_t DP_ADDRESS_0 = 0x08;
constexpr uint32_t DP_ADDRESS_1 = 0x0C;
constexpr uint32_t DP_ADDRESS_2 = 0x10;
constexpr uint32_t DP_ADDRESS_3 = 0x14;
constexpr uint32_t DP_BUFFER_LENGTH = 0x18; // this is in bytes
constexpr uint32_t DP_BUFFER_MASK = 0x1C; // each bit enables a buffer
constexpr uint32_t DP_HOST_MAC_LOW = 0x20;
constexpr uint32_t DP_HOST_MAC_HIGH = 0x24;
constexpr uint32_t DP_HOST_IP = 0x28;
constexpr uint32_t DP_HOST_UDP_PORT = 0x2C;

// SIF: packetizer configuration.
constexpr uint32_t PACKETIZER_MODE = 0x0C;
constexpr uint32_t PACKETIZER_RAM = 0x04;
constexpr uint32_t PACKETIZER_DATA = 0x08;

class Enumerator;
class Programmer;

class DataChannel {
    // Enumerator calls our methods to reconfigure Metadata.
    friend class Enumerator;
    friend class Programmer;

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
    void configure_roce(uint64_t frame_memory, size_t frame_size, size_t page_size, unsigned pages, uint32_t local_data_port);

    /**
     * Configure the data channel for 1722 COE packets.
     */
    void configure_coe(uint8_t channel, size_t frame_size, uint32_t pixel_width, bool vlan_enabled = false);

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

    /**
     * Disable the packetizer.
     */
    void disable_packetizer();

    /**
     * Enable the packetizer for 10-bit data.
     */
    void enable_packetizer_10();

    /**
     * Enable the packetizer for 12-bit data.
     */
    void enable_packetizer_12();

    /**
     * Get a Sequencer that's connected to the frame-end input.
     */
    std::shared_ptr<Hololink::Sequencer> frame_end_sequencer();

    /**
     * Configure the given metadata to send data to the given
     * host interface.
     */
    static void use_data_plane_configuration(Metadata& metadata, int64_t data_plane);

    Metadata& enumeration_metadata() { return enumeration_metadata_; }

protected:
    /**
     * Programming tools can use this constructor to avoid version checking.
     */
    DataChannel(const Metadata& metadata, std::shared_ptr<Hololink> hololink);

    /**
     * Common routine for constructors
     */
    void initialize(const Metadata& metadata, std::shared_ptr<Hololink> hololink);

    /**
     * Write the common data channel configuration shared between RoCe and COE w/1722B.
     */
    void configure_common(uint32_t frame_size, uint32_t header_size, uint32_t local_data_port = 0);

private:
    std::shared_ptr<Hololink> hololink_;
    Metadata enumeration_metadata_; // we keep a copy of what we were instantiated with
    std::string peer_ip_;
    uint32_t vp_mask_;
    uint32_t qp_number_;
    uint32_t rkey_;
    uint32_t data_plane_;
    uint32_t sensor_;
    std::string multicast_;
    uint16_t multicast_port_;
    uint16_t broadcast_port_;
    bool frame_end_event_valid_;
    Hololink::Event frame_end_event_;
    uint32_t vp_address_;
    uint32_t hif_address_;
    uint32_t sif_address_;
    std::string fpga_uuid_;
};

} // namespace hololink

#endif /* SRC_HOLOLINK_DATA_CHANNEL */
