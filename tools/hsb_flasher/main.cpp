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

#include "hsb_discovery.hpp"
#include "hsb_flasher.hpp"
#include "hsb_flasher_args.hpp"
#include "hsb_fw_lookup.hpp"

#include <cstdlib>

using namespace hsb_flasher;

int main(int argc, char* argv[])
{
    hsb_flasher_context context {};

    if (!parse_arguments(argc, argv, context)) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (!discover_device(context)) {
        log_info("Error: Target device discovery failed.");
        return EXIT_FAILURE;
    }

    // Use UUID override for legacy devices that don't report UUID in BOOTP
    std::string fpga_uuid = context.enumeration_metadata.get<std::string>("fpga_uuid").value_or("N/A");
    if (fpga_uuid == "N/A" && !context.fpga_uuid_override.empty()) {
        log_info("Using provided UUID override: " + context.fpga_uuid_override);
        context.enumeration_metadata["fpga_uuid"] = context.fpga_uuid_override;
    }

    // --- Direct flash mode: skip YAML lookup and firmware fetch ---
    if (context.direct_flash.enabled()) {
        if (!context.direct_flash.flash_version.empty()) {
            int64_t flash_version = std::stoi(context.direct_flash.flash_version, nullptr, 16);
            context.enumeration_metadata["hsb_ip_version"] = flash_version;
        }

        auto flasher = get_flasher(context);
        if (!flasher) {
            log_info("Error: Failed to get flasher");
            return EXIT_FAILURE;
        }

        if (!flasher->flash(context.direct_flash.clnx_path,
                context.direct_flash.cpnx_path)) {
            log_info("Error: Failed to flash");
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

    // --- Standard mode: YAML lookup + firmware fetch ---
    int64_t current_version = context.enumeration_metadata.get<int64_t>("hsb_ip_version").value_or(0);
    if (current_version == std::stoi(context.target_version, nullptr, 16)) {
        log_info("Device is already at the target version");
        return EXIT_SUCCESS;
    }

    if (!find_manifest_by_uuid(context)) {
        log_info("Error: No firmware information found for FPGA UUID: " + context.enumeration_metadata.get<std::string>("fpga_uuid").value_or("N/A"));
        return EXIT_FAILURE;
    }

    if (!verify_firmware_details(context)) {
        log_info("Error: Target firmware 0x" + context.target_version + " not found in YAML file: " + context.firmware_info_path);
        return EXIT_FAILURE;
    }

    if (!fetch_target_firmware(context)) {
        log_info("Error: Failed to fetch target firmware");
        return EXIT_FAILURE;
    }

    auto flasher = get_flasher(context);
    if (!flasher) {
        log_info("Error: Failed to get flasher");
        return EXIT_FAILURE;
    }

    if (!flasher->flash(context.clnx.local_location, context.cpnx.local_location)) {
        log_info("Error: Failed to flash");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}