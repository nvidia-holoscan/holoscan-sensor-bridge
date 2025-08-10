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

namespace core {
    class Deserializer;
}

// FPGA UUIDs for some legacy bootp-v1 configurations.  All newer devices,
// specifically producing bootp-v2, provide us their UUIDs directly.
const std::string HOLOLINK_LITE_UUID = "889b7ce3-65a5-4247-8b05-4ff1904c3359";
const std::string HOLOLINK_NANO_UUID = "d0f015e0-93b6-4473-b7d1-7dbd01cbeab5";
const std::string HOLOLINK_100G_UUID = "7a377bf7-76cb-4756-a4c5-7dddaed8354b";
const std::string MICROCHIP_POLARFIRE_UUID = "ed6a9292-debf-40ac-b603-a24e025309c1";
const std::string LEOPARD_EAGLE_UUID = "f1627640-b4dc-48af-a360-c55b09b3d230";

/**
 * Strategy that adjusts enumeration data based on FPGA UUID.
 */
class EnumerationStrategy {
public:
    // All ABCs have virtual destructors.
    virtual ~EnumerationStrategy();

    // Allow a hook to configure metadata before it's returned to the
    // application.
    virtual void update_metadata(Metadata& metadata, hololink::core::Deserializer& deserializer) = 0;

    // Implements the work of DataChannel::use_sensor.
    virtual void use_sensor(Metadata& metadata, int64_t sensor_number) = 0;

    // Implements the work of DataChannel::use_data_plane_configuration.
    virtual void use_data_plane_configuration(Metadata& metadata, int64_t data_plane) = 0;
};

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

    /**
     * Configure enumeration based on the UUID found during enumeration.
     * @returns any previous registration for this UUID or nullptr.
     */
    static std::shared_ptr<EnumerationStrategy> set_uuid_strategy(std::string uuid, std::shared_ptr<EnumerationStrategy> enumeration_strategy);

    static EnumerationStrategy& get_uuid_strategy(std::string uuid);

protected:
    static void configure_default_enumeration_strategies();

private:
    const std::string local_interface_;
    const uint32_t bootp_request_port_;
    const uint32_t bootp_reply_port_;

    // Static map to store UUID strategies
    static std::map<std::string, std::shared_ptr<EnumerationStrategy>> uuid_strategies_;
};

/**
 * Supports HSB devices with the given count of
 * sensors and SIFs for each sensor.
 */
class BasicEnumerationStrategy
    : public EnumerationStrategy {
public:
    BasicEnumerationStrategy(const Metadata& additional_metadata, unsigned total_sensors = 2, unsigned total_dataplanes = 2, unsigned sifs_per_sensor = 2);

    // Changes whether ptp_enable is true or not.
    void ptp_enable(bool enable);
    // Changes whether vsync_enable is true or not.
    void vsync_enable(bool enable);

    void update_metadata(Metadata& metadata, hololink::core::Deserializer& deserializer) override;
    void use_sensor(Metadata& metadata, int64_t sensor_number) override;
    void use_data_plane_configuration(Metadata& metadata, int64_t data_plane) override;

protected:
    Metadata additional_metadata_;
    unsigned total_sensors_;
    unsigned total_dataplanes_;
    unsigned sifs_per_sensor_;
    bool ptp_enable_;
    bool vsync_enable_;
};

} // namespace hololink

#endif /* SRC_HOLOLINK_ENUMERATOR */
