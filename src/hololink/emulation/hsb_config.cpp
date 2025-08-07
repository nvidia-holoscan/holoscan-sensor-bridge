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

#include "hsb_config.hpp"

namespace hololink::emulation {
std::map<DataPlaneID, DataPlaneConfiguration> data_plane_map {
    { DATA_PLANE_0, DataPlaneConfiguration { 0x02000300 } },
    { DATA_PLANE_1, DataPlaneConfiguration { 0x02010300 } },
};
std::map<SensorID, SensorConfiguration> sensor_map {
    { SENSOR_0, SensorConfiguration { 0, 0x1, hololink::Hololink::Event::SIF_0_FRAME_END, 0x01000000, 0x1000 } },
    { SENSOR_1, SensorConfiguration { 2, 0x4, hololink::Hololink::Event::SIF_1_FRAME_END, 0x01010000, 0x1080 } },
};

std::map<uint32_t, uint32_t> address_map = {
    { 0, hololink::DP_ADDRESS_0 },
    { 1, hololink::DP_ADDRESS_1 },
    { 2, hololink::DP_ADDRESS_2 },
    { 3, hololink::DP_ADDRESS_3 },
};
} // namespace hololink::emulation
