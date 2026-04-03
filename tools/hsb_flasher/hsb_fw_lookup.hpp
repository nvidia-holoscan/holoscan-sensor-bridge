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

#include "hsb_flasher_context.hpp"

namespace hsb_flasher {

/**
 * @brief Find the firmware manifest (YAML file) matching the device's FPGA UUID.
 *
 * Scans the @c firmware_information/ directory (adjacent to the executable) for
 * YAML files whose @c fpga_uuid field matches the value in
 * @c context.enumeration_metadata. On success, stores the absolute path of the
 * matching file in @p context.firmware_info_path.
 *
 * @param[in,out] context  Manager context with a valid @c enumeration_metadata
 *                         containing @c "fpga_uuid".
 * @return true if a matching YAML file was found, false otherwise.
 */
bool find_manifest_by_uuid(hsb_flasher_context& context);

/**
 * @brief Verify the target firmware version exists in the YAML file and load its metadata.
 *
 * Parses the YAML file at @p context.firmware_info_path and searches for a
 * @c firmware_versions entry matching @p context.target_version. On success,
 * populates @p context.clnx and @p context.cpnx with the location, MD5, and
 * size of the CLNX and CPNX firmware files.
 *
 * Must be called after a successful find_manifest_by_uuid().
 *
 * @param[in,out] context  Manager context with a valid @c firmware_info_path
 *                         and @c target_version.
 * @return true if the target version was found, false otherwise.
 */
bool verify_firmware_details(hsb_flasher_context& context);

/**
 * @brief Fetch and verify the target CLNX and CPNX firmware files.
 *
 * For each firmware file (CLNX and CPNX), this function:
 *   1. Resolves the local path — if the location begins with @c "http",
 *      downloads the file only if it does not already exist locally.
 *      If the location is a local path, uses it directly.
 *   2. Verifies the file exists, its size matches the manifest, and its
 *      MD5 hash matches the expected value.
 *
 * On success, @p context.clnx.local_location and
 * @p context.cpnx.local_location are set to the verified local paths.
 *
 * Must be called after a successful verify_firmware_details().
 *
 * @param[in,out] context  Manager context with populated @c clnx and @c cpnx
 *                         firmware info (location, md5, size).
 * @return true if both firmware files were fetched and verified, false on
 *         download or verification failure.
 */
bool fetch_target_firmware(hsb_flasher_context& context);

} // namespace hsb_flasher
