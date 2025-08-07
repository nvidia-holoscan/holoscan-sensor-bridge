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

#ifndef LINUX_DATA_PLANE_HPP
#define LINUX_DATA_PLANE_HPP

#include "data_plane.hpp"

namespace hololink::emulation {

/**
 * @brief LinuxDataPlane is a DataPlane that uses LinuxTransmitter to send data.
 *
 */
class LinuxDataPlane : public DataPlane {
public:
    LinuxDataPlane(HSBEmulator& hsb_emulator, const IPAddress& source_ip, uint16_t source_port, DataPlaneID data_plane_id, SensorID sensor_id);
    ~LinuxDataPlane();

protected:
    /**
     * @brief Update the TransmissionMetadata for the LinuxDataPlane.
     *
     * @note Only one thread may call LinuxDataPlane::send at a time. DataPlane::send() ensures thread safety.
     */
    void update_metadata() override;

    // cache for updating the next_page_
    uint32_t next_page_ { 0 };
};

} // namespace hololink::emulation

#endif // LINUX_DATA_PLANE_HPP