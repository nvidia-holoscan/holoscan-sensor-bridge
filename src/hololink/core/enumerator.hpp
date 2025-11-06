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
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "metadata.hpp"
#include "timeout.hpp"

namespace hololink {

namespace core {
    class Deserializer;
    class Reactor;
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
     * @brief Handle for managing registered callbacks
     */
    class CallbackHandle {
    public:
        CallbackHandle(uint64_t id, const std::string& ip, const std::function<void(Metadata&)>& callback)
            : id_(id)
            , ip_(ip)
            , callback_(callback)
        {
        }
        uint64_t id() const { return id_; }
        const std::string& ip() const { return ip_; }
        const std::function<void(Metadata&)>& callback() const { return callback_; }

    private:
        uint64_t id_;
        std::string ip_;
        std::function<void(Metadata&)> callback_;
    };

    /**
     * @brief Inner ReactorEnumerator class for managing IP callbacks
     */
    class ReactorEnumerator {
    public:
        static std::shared_ptr<ReactorEnumerator> get_reactor_enumerator();

        ReactorEnumerator();
        ~ReactorEnumerator();

        void start();
        std::shared_ptr<CallbackHandle> register_ip(const std::string& ip, const std::function<void(Metadata&)>& callback);
        void unregister_ip(const std::shared_ptr<CallbackHandle>& handle);

    private:
        void fd_callback(int fd, int events);

        std::map<std::string, std::vector<std::shared_ptr<CallbackHandle>>> ip_callback_map_;
        std::recursive_mutex lock_;
        int socket_fd_;
        uint64_t next_callback_id_;
        std::shared_ptr<core::Reactor> reactor_;
    };

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
     * @brief Register a callback for a specific IP address
     *
     * @param ip IP address to monitor
     * @param callback Function to call when enumeration packets are received from this IP
     * @return CallbackHandle for unregistering
     */
    static std::shared_ptr<CallbackHandle> register_ip(const std::string& ip, const std::function<void(Metadata&)>& callback);

    /**
     * @brief Unregister a callback
     *
     * @param handle The handle returned by register_ip
     */
    static void unregister_ip(const std::shared_ptr<CallbackHandle>& handle);

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

    /**
     * Configure an AF_INET SOCK_DGRAM socket
     * to listen to bootp on the given interface
     * (or all if not specified)
     */
    static bool configure_socket(int fd, uint32_t port = 12267u, const std::string& local_interface = "");

    static std::tuple<Metadata, std::vector<uint8_t>> handle_bootp_fd(int fd);

protected:
    static void configure_default_enumeration_strategies();

private:
    const std::string local_interface_;
    const uint32_t bootp_request_port_;
    const uint32_t bootp_reply_port_;

    // Static map to store UUID strategies
    static std::map<std::string, std::shared_ptr<EnumerationStrategy>> uuid_strategies_;

    // Static ReactorEnumerator instance
    static std::shared_ptr<ReactorEnumerator> reactor_enumerator_;
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
    // Changes whether block_enable is true or not.
    void block_enable(bool enable);

    void update_metadata(Metadata& metadata, hololink::core::Deserializer& deserializer) override;
    void use_sensor(Metadata& metadata, int64_t sensor_number) override;
    void use_data_plane_configuration(Metadata& metadata, int64_t data_plane) override;

protected:
    Metadata additional_metadata_;
    unsigned total_sensors_;
    unsigned total_dataplanes_;
    unsigned sifs_per_sensor_;
    bool ptp_enable_;
    bool block_enable_;
};

} // namespace hololink

#endif /* SRC_HOLOLINK_ENUMERATOR */
