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
#include "utils.hpp"

namespace hololink::emulation {

const HSBConfiguration HSB_EMULATOR_CONFIG = {
    .tag = HSB_EMULATOR_TAG,
    .tag_length = HSB_EMULATOR_TAG_LENGTH,
    .vendor_id = HSB_EMULATOR_VENDOR_ID,
    .enum_version = HSB_EMULATOR_ENUM_VERSION,
    .uuid = HSB_EMULATOR_UUID,
    .serial_num = HSB_EMULATOR_SERIAL_NUM,
    .hsb_ip_version = HSB_EMULATOR_HSB_IP_VERSION,
    .fpga_crc = HSB_EMULATOR_FPGA_CRC,
    .sensor_count = HSB_DEFAULT_SENSOR_COUNT,
    .data_plane_count = HSB_DEFAULT_DATA_PLANE_COUNT,
    .sifs_per_sensor = HSB_DEFAULT_SIFS_PER_SENSOR,
};
const HSBConfiguration HSB_LEOPARD_EAGLE_CONFIG = {
    .tag = HSB_EMULATOR_TAG,
    .tag_length = HSB_EMULATOR_TAG_LENGTH,
    .vendor_id = HSB_EMULATOR_VENDOR_ID,
    .enum_version = HSB_EMULATOR_ENUM_VERSION,
    .uuid = { 0xf1, 0x62, 0x76, 0x40,
        0xb4, 0xdc,
        0x48, 0xaf,
        0xa3, 0x60,
        0xc5, 0x5b, 0x09, 0xb3, 0xd2, 0x30 },
    .serial_num = HSB_EMULATOR_SERIAL_NUM,
    .hsb_ip_version = HSB_EMULATOR_HSB_IP_VERSION,
    .fpga_crc = HSB_EMULATOR_FPGA_CRC,
    .sensor_count = 3,
    .data_plane_count = 1,
    .sifs_per_sensor = 1,
};

int hsb_config_set_uuid(HSBConfiguration& config, const char* uuid_str)
{
    if (uuid_parse(uuid_str, config.uuid) != 0) {
        return 1;
    }
    return 0;
}

} // namespace hololink::emulation
