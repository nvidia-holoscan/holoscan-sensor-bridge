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
#include <memory>
#include <mutex>
#include <thread>
#include <unistd.h>

#include "base_transmitter.hpp"
#include "dlpack/dlpack.h"
#include "hsb_config.hpp"
#include "hsb_emulator.hpp"
#include "net.hpp"

namespace hololink::emulation {

// Forward declarations
class BaseTransmitter;
struct TransmissionMetadata;
class MemRegister;

/**
 * @brief DataPlane is the class with which HSB Emulator applications control
 * data transmission and HSB Host applications establish "connections" and data
 * flow via bootp enumeration. Calling start() will initiate BootP broadcasting
 * so that the HSB Host applications can detect the emulator and configure their
 * data receivers. Calling stop() will stop the BootP broadcasting. Data
 * transmission will still be available, but metadata in general cannot be
 * updated. Data transmission will cease when the DataPlane object itself is
 * destroyed.
 *
 * @note Internally, the DataPlane object holds a reference to the HSBEmulator
 * object it was constructed from. If the HSBEmulator object is destructed, any
 * accesses to that object by subclasses are UB
 *
 * @note The DataPlane object is not yet fully featured. Most actions that would be
 * supported by a real HSB to change its configuration on the fly are not supported,
 * e.g. updating IP address or any part of HSBConfiguration for Bootp
 */
class DataPlane {
public:
    /**
     * @brief Constructs a DataPlane object..
     * @param hsb_emulator A reference to an HSBEmulator instance that the DataPlane will configure itself from.
     * @param ip_address The IP address of the DataPlane.
     * @param data_plane_id The DataPlaneID of the DataPlane.
     * @param sensor_id The SensorID of the DataPlane.
     * The data_plane_id and sensor_id are used to identify registers needed to compile metadata.
     */
    DataPlane(HSBEmulator& hsb_emulator, const IPAddress& ip_address, DataPlaneID data_plane_id, SensorID sensor_id);
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
     * @brief Check if the DataPlane is running. (Bootp is broadcasting)
     * @return True if the DataPlane is running, false otherwise.
     */
    bool is_running();

    /**
     * @brief Send a tensor over the DataPlane.
     * @param tensor The tensor object reference to send. Supported device types are kDLCPU, kDLCUDA, kDLCUDAHost (host pinned), and kDLCUDAManaged (Unified Memory)
     * @return The number of bytes sent or < 0 if error occurred.
     *
     * @note This method is synchronous. It will block until the send is complete.
     */
    int64_t send(const DLTensor& tensor);

protected:
    /**
     * @brief method to update metadata model for the appropriate transmitter
     * subclass can assume that appropriate locks are held while accessing memory registers and the transmitter metadata
     */
    virtual void update_metadata() = 0; // to be overridden by subclasses

    HSBEmulator& hsb_emulator_;
    std::shared_ptr<MemRegister> registers_; // keep a reference of the MemRegister alive and inheritable
    IPAddress ip_address_;
    HSBConfiguration configuration_; /* this is separate copy from what is in hsb_emulator because it may have different values as transmitted over Bootp*/
    SensorID sensor_id_;

    // transmitter data
    std::mutex metadata_mutex_; // protects against multiple threads updating metadata_ at the same time
    // lifecycle of these objects must be managed by the subclass implementation
    // since TransmissionMetadata is POD, should malloc/calloc and free
    TransmissionMetadata* metadata_ { nullptr };
    // BaseTransmitter has a virtual destructor, so subclasses can new/delete
    BaseTransmitter* transmitter_ { nullptr };

private:
    // bootp thread control is private. not intended to be accessed by subclasses

    /**
     * @brief method for separate thread to Broadcast the bootp packet to the network.
     * @note This method is called by the DataPlane constructor and is not thread-safe. This is why I currently do not allow updating IP address after construction.
     * @note This method is protected because it is only called by the DataPlane constructor and is not thread-safe.
     */
    void broadcast_bootp();

    std::atomic<bool> running_ { false };
    std::thread bootp_thread_;
};

}

#endif
