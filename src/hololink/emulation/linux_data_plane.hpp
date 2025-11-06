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
 * @brief The DataPlane implementation for RoCEv2 UDP transport.
 */
class LinuxDataPlane : public DataPlane {
public:
    /**
     * python:
     *
     * `def __init__(self: hemu.LinuxDataPlane, hsb_emulator: hemu.HSBEmulator, source_ip: hemu.IPAddress, data_plane_id: int, sensor_id: int)`
     *
     * @brief Construct a new LinuxDataPlane object
     * @param hsb_emulator The HSBEmulator object to attach to.
     * @param source_ip The IP address of the DataPlane.
     * @param data_plane_id The identifying index of the DataPlane.
     * @param sensor_id The identifying index of the sensor interface associated with the DataPlane.
     */
    LinuxDataPlane(HSBEmulator& hsb_emulator, const IPAddress& source_ip, uint8_t data_plane_id, uint8_t sensor_id);
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