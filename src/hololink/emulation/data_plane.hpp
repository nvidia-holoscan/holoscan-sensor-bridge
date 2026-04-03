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

#ifndef EMULATION_DATA_PLANE_H
#define EMULATION_DATA_PLANE_H

#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>

#include "dlpack/dlpack.h"

#include "base_transmitter.hpp"
#include "hsb_config.hpp"

#include "hsb_emulator.hpp"
#include "net.hpp"

// registers (taken from data_channel.hpp)
#define DP_QP 0x00u
#define DP_RKEY 0x04u
#define DP_PAGE_LSB 0x08u
#define DP_PAGE_MSB 0x0Cu
#define DP_PAGE_INC 0x10u
#define DP_MAX_BUFF 0x14u
#define DP_BUFFER_LENGTH 0x18u
#define DP_HOST_MAC_LOW 0x20u
#define DP_HOST_MAC_HIGH 0x24u
#define DP_HOST_IP 0x28u
#define DP_HOST_UDP_PORT 0x2Cu
#define DP_PACKET_SIZE 0x04u
#define DP_PACKET_UDP_PORT 0x08u
#define DP_VP_MASK 0x0Cu

#define BOOTP_PORT 8192u
#define BOOTP_BUFFER_SIZE 1500u
#define BOOTP_INTERVAL_SEC 1u
#define BOOTP_REQUEST_PORT 12267u
#define BOOTP_REPLY_PORT 12268u

// constants for bootp packet
#define BOOTP_CHADDR_SIZE 16u
#define BOOTP_SNAME_SIZE 64u
#define BOOTP_FILE_SIZE 128u
#define BOOTP_VEND_SIZE 64u
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

static_assert(sizeof(BootpPacket) == 300, "BootpPacket must be 300 bytes to ensure it is packed");

namespace hololink::emulation {

// Forward declarations
class BaseTransmitter;
struct TransmissionMetadata;
class AddressMemory;
struct DataPlaneCtxt;

/**
 * @brief DataPlane is the partially implemented Abstract Base Class with which HSB Emulator applications control
 * data transmission and HSB Host applications establish "connections" and data
 * flow via Bootp enumeration. The parent instance of any `DataPlane` implementation will manage Bootp broadcasting.
 * Each implementation must interface with the HSB Emulator's internal memory model (`AddressMemory`) to update and manage
 * metadata and any state needed for the metadata or transport layer for the corresponding `Transmitter`.
 *
 * @note The DataPlane lifecycle operations are managed by the `HSBEmulator` instance it is attached to and therefore `DataPlane` lifetime must be at least as long as the `HSBEmulator` instance or at least until the final `HSBEmulator::stop()` is called.
 */
class DataPlane {
public:
    /**
     * @brief Constructs a DataPlane object.
     * @param hsb_emulator A reference to an HSBEmulator instance that the DataPlane will configure itself from.
     * @param ip_address The IP address of the DataPlane.
     * @param data_plane_id The data plane index of the DataPlane.
     * @param sensor_id The sensor index of the DataPlane to associate with the DataPlane.
     * The data_plane_id and sensor_id are used to identify registers needed to compile metadata.
     */
    DataPlane(HSBEmulator& hsb_emulator, const IPAddress& ip_address, uint8_t data_plane_id, uint8_t sensor_id);
    virtual ~DataPlane();

    /**
     * @brief Start the DataPlane by initiating the BootP broadcast
     */
    void start();

    /**
     * @brief Initialize the DataPlane by validating the configuration and registers.
     * @return 0 on success, < 0 on failure.
     */
    int init();

    /**
     * @brief Stop the DataPlane by stopping the BootP broadcast
     */
    void stop();
    /**
     * @brief This is a clear alias for stop()
     */
    void stop_bootp() { stop(); }; // clear alias for stop()

    /**
     * @brief Check if the DataPlane is running. (Bootp is broadcasting)
     * @return True if the DataPlane is running, false otherwise.
     */
    bool is_running();

    /**
     * @brief Send a tensor over the DataPlane.
     * @param tensor The tensor object reference to send. Supported device types are kDLCPU, kDLCUDA, kDLCUDAHost (host pinned), and kDLCUDAManaged (Unified Memory)
     * @param frame_metadata The frame metadata to send. Acts as a buffer flush. If nullptr, the data is by default buffered until the next send command that fills an MTU or a non-nullptr frame_metadata is provided.
     * @return The number of bytes sent or < 0 if error occurred.
     *
     * @note This method is synchronous. It will block and metadata will be protected by a mutex until the send is complete.
     */
    int64_t send(const DLTensor& tensor, FrameMetadata* frame_metadata = DEFAULT_FRAME_METADATA);

    /**
     * @brief Send a buffer over the DataPlane.
     * @param buffer The buffer to send.
     * @param buffer_size The size of the buffer.
     * @param frame_metadata The frame metadata to send. Acts as a buffer flush. If nullptr, the data is by default buffered until the next send command that fills an MTU or a non-nullptr frame_metadata is provided.
     * @return The number of bytes sent or < 0 if error occurred.
     */
    int64_t send(const uint8_t* buffer, size_t buffer_size, FrameMetadata* frame_metadata = nullptr);

    /**
     * @brief Get the sensor ID associated with the DataPlane.
     * @return The sensor ID.
     */
    uint8_t get_sensor_id() const { return sensor_id_; }

    /**
     * @brief Check if the packetizer is enabled.
     * @return True if the packetizer is enabled, false otherwise.
     */
    bool packetizer_enabled() const;

    int broadcast_bootp();

protected:
    /**
     * @brief method to update metadata model for the appropriate transmitter
     * subclass can assume that appropriate locks are held while accessing memory registers and the transmitter metadata
     */
    virtual void update_metadata(); // to be overridden by subclasses

    UniqueDel<AddressMemory>& registers_; // keep a reference of the AddressMemory alive and inheritable
    IPAddress ip_address_;
    HSBConfiguration configuration_; /* this is separate copy from what is in hsb_emulator because it may have different values as transmitted over Bootp*/
    uint8_t sensor_id_;
    uint8_t data_plane_id_;

    // transmitter data
    // lifecycle of these objects must be managed by the subclass implementation
    // since TransmissionMetadata is POD, should malloc/calloc and free
    TransmissionMetadata* metadata_ { nullptr };
    // BaseTransmitter has a virtual destructor, so subclasses can new/delete
    BaseTransmitter* transmitter_ { nullptr };

    // DataPlaneConfiguration
    uint32_t hif_address_ { 0 };
    // SensorConfiguration
    uint32_t vp_mask_ { 0 };
    Event frame_end_event_ { Event::EVENT_SIF_0_FRAME_END };
    uint32_t sif_address_ { 0 };
    uint32_t vp_address_ { 0 };

    std::unique_ptr<struct DataPlaneCtxt, std::function<void(DataPlaneCtxt*)>> data_plane_ctxt_ { nullptr };
};

}

#endif
