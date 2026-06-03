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

#include <assert.h>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "base_transmitter.hpp"
#include "data_plane.hpp"
#include "hsb_emulator.hpp"
#include "net.hpp"
#include "utils.hpp"

namespace hololink::emulation {

int DataPlane::init()
{
    // validate against configuration

    if (data_plane_id_ >= configuration_.data_plane_count) {
        THROW_EXCEPTION(-1, "data_plane_id exceeds configured data_plane_count");
    }
    if (sensor_id_ >= configuration_.sensor_count) {
        THROW_EXCEPTION(-2, "sensor_id exceeds configured sensor count");
    }
    if (!(ip_address_.flags & IPADDRESS_HAS_BROADCAST)) {
        THROW_EXCEPTION(-3, "Broadcast address not found for ip address %s", IPAddress_to_string(ip_address_).c_str());
    }
    // strictly speaking, this is just a warning.
    if (!(ip_address_.flags & IPADDRESS_HAS_MAC)) {
        fprintf(stderr, "MAC address not found/recognized for ip address %s. For non-COE applications, this is non-fatal\n", IPAddress_to_string(ip_address_).c_str());
    }
    return 0;
}

// since we use unique_ptr for DataPlaneCtxt, we need to define the destructor for the holding class after the definition of DataPlaneCtxt
DataPlane::~DataPlane()
{
    stop();
}

} // namespace hololink::emulation
