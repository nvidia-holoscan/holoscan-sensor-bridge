/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 */

#pragma once

#include <cstdint>
#include <string>

namespace hololink {

/**
 * Flash firmware using legacy connection mode.
 * For older FPGA versions that need special connection handling.
 *
 * @param ip_address IP address of the device
 * @param clnx_path Path to CLNX firmware file
 * @param cpnx_path Path to CPNX firmware file
 * @param current_version Current firmware version (for metadata construction)
 * @param fpga_uuid FPGA UUID
 * @param serial_number Device serial number
 * @return true if flash succeeded, false otherwise
 */
bool hsb_lite_flash_2504(const std::string& ip_address,
    const std::string& clnx_path,
    const std::string& cpnx_path,
    int64_t current_version,
    const std::string& fpga_uuid,
    const std::string& serial_number);

} // namespace hololink
