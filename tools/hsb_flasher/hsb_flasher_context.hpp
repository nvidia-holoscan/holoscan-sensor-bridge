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

#include <iostream>
#include <string>

#include <hololink/core/metadata.hpp>

namespace hsb_flasher {

enum class hsb_flasher_log_level {
    INFO, // Basic information
    DEBUG, // Detailed breakdown
};

inline void log_info(const std::string& message)
{
    std::cout << message << std::endl;
}

inline void log_debug(hsb_flasher_log_level log_level, const std::string& message)
{
    if (log_level == hsb_flasher_log_level::DEBUG) {
        std::cout << message << std::endl;
    }
}

struct target_firmware_info {
    // Location of the target firmware; loaded from the YAML file
    std::string location;
    // MD5 hash of the target firmware; loaded from the YAML file
    std::string md5;
    // Size of the target firmware; loaded from the YAML file
    size_t size = 0;
    // Location of the target firmware in local filesystem; will be used for actual flash
    std::string local_location;
};

struct direct_flash_config {
    // Path to CLNX firmware file
    std::string clnx_path;
    // Path to CPNX firmware file
    std::string cpnx_path;
    // Version override for flash strategy selection (e.g. "2504")
    // When set, injected into enumeration_metadata as hsb_ip_version
    // to control which flash strategy class is selected.
    std::string flash_version;

    bool enabled() const
    {
        return !cpnx_path.empty();
    }

    bool has_any() const
    {
        return !clnx_path.empty() || !cpnx_path.empty() || !flash_version.empty();
    }
};

struct hsb_flasher_context {
    // Verbosity level
    // User provided: Optional. Default is INFO.
    hsb_flasher_log_level log_level;
    // IP address of the HSB device
    // User provided: Required.
    std::string hololink_ip;
    // Target version to flash
    // User provided: Required.
    std::string target_version;
    // FPGA UUID override for legacy devices that don't report UUID
    // User provided: Optional. Used if enumeration returns fpga_uuid as "N/A".
    std::string fpga_uuid_override;
    // Timeout for enumeration in seconds
    // User provided: Optional. Default is 3.0 seconds.
    float timeout;
    // Enumeration metadata for the matched device.
    // Populated by discover_device(); contains at minimum:
    //   "peer_ip"        (std::string) - IP address
    //   "mac_id"         (std::string) - MAC address
    //   "hsb_ip_version" (int64_t)     - firmware version
    //   "fpga_uuid"      (std::string) - FPGA UUID (may be "N/A")
    hololink::Metadata enumeration_metadata;
    // Absolute path to the matching firmware information YAML file
    std::string firmware_info_path;
    // Target firmware info
    struct target_firmware_info clnx;
    struct target_firmware_info cpnx;

    // Direct flash configuration (bypass YAML lookup + firmware fetch)
    struct direct_flash_config direct_flash;
};

} // namespace hsb_flasher