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

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <unistd.h>

#include "dlpack/dlpack.h"

#include "hololink/core/data_channel.hpp"

#include "base_transmitter.hpp"
#include "hsb_config.hpp"

#include "hsb_emulator.hpp"
#include "net.hpp"

namespace hololink::emulation {

extern std::map<uint32_t, uint32_t> ADDRESS_MAP;

// Forward declarations
class BaseTransmitter;
struct TransmissionMetadata;
class MemRegister;

/**
 * @brief DataPlane is the partially implemented Abstract Base Class with which HSB Emulator applications control
 * data transmission and HSB Host applications establish "connections" and data
 * flow via Bootp enumeration. The parent instance of any `DataPlane` implementation will manage Bootp broadcasting.
 * Each implementation must interface with the HSB Emulator's internal memory model (`MemRegister`) to update and manage
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
     * @return The number of bytes sent or < 0 if error occurred.
     *
     * @note This method is synchronous. It will block and metadata will be protected by a mutex until the send is complete.
     */
    int64_t send(const DLTensor& tensor);

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

protected:
    /**
     * @brief method to update metadata model for the appropriate transmitter
     * subclass can assume that appropriate locks are held while accessing memory registers and the transmitter metadata
     */
    virtual void update_metadata() = 0; // to be overridden by subclasses

    std::shared_ptr<MemRegister> registers_; // keep a reference of the MemRegister alive and inheritable
    IPAddress ip_address_;
    HSBConfiguration configuration_; /* this is separate copy from what is in hsb_emulator because it may have different values as transmitted over Bootp*/
    uint8_t sensor_id_;

    // transmitter data
    std::mutex metadata_mutex_; // protects against multiple threads updating metadata_ at the same time
    // lifecycle of these objects must be managed by the subclass implementation
    // since TransmissionMetadata is POD, should malloc/calloc and free
    TransmissionMetadata* metadata_ { nullptr };
    // BaseTransmitter has a virtual destructor, so subclasses can new/delete
    BaseTransmitter* transmitter_ { nullptr };

    // DataPlaneConfiguration
    uint32_t hif_address_ { 0 };
    // SensorConfiguration
    uint32_t vp_mask_ { 0 };
    hololink::Hololink::Event frame_end_event_ { hololink::Hololink::Event::SIF_0_FRAME_END };
    uint32_t sif_address_ { 0 };
    uint32_t vp_address_ { 0 };

private:
    // bootp thread control is private. not intended to be accessed by subclasses

    /**
     * @brief method for separate thread to Broadcast the bootp packet to the network.
     * @note This method is called by the DataPlane constructor and is not thread-safe. This is why I currently do not allow updating IP address after construction.
     */
    void broadcast_bootp();

    std::atomic<bool> running_ { false };
    std::thread bootp_thread_;
};

}

#endif
