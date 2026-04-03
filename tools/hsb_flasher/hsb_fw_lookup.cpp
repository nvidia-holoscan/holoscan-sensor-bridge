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

#include "hsb_fw_lookup.hpp"

#include <algorithm>
#include <cstdlib>
#include <curl/curl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <openssl/evp.h>
#include <sstream>
#include <yaml-cpp/yaml.h>

namespace hsb_flasher {

static std::filesystem::path get_executable_directory()
{
    // On Linux, /proc/self/exe is a symlink to the current executable
    return std::filesystem::read_symlink("/proc/self/exe").parent_path();
}

bool find_manifest_by_uuid(hsb_flasher_context& context)
{
    std::filesystem::path exe_dir = get_executable_directory();
    std::filesystem::path firmware_info_dir = exe_dir / "firmware_information";

    // Sanity check "firmware_information" directory exists
    if (!std::filesystem::exists(firmware_info_dir) || !std::filesystem::is_directory(firmware_info_dir)) {
        log_info("Error: firmware_information directory not found at " + firmware_info_dir.string());
        return false;
    }

    // Iterate through all YAML files and find one with matching fpga_uuid
    for (const auto& entry : std::filesystem::directory_iterator(firmware_info_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        std::string extension = entry.path().extension().string();
        if (extension != ".yaml" && extension != ".yml") {
            continue;
        }

        log_debug(context.log_level, "Checking firmware file: " + entry.path().string());

        try {
            YAML::Node config = YAML::LoadFile(entry.path().string());

            if (!config["fpga_uuid"]) {
                log_debug(context.log_level, "No fpga_uuid found in " + entry.path().string());
                continue;
            }

            std::string device_uuid = context.enumeration_metadata.get<std::string>("fpga_uuid").value_or("N/A");
            const YAML::Node& uuid_node = config["fpga_uuid"];

            if (uuid_node.IsSequence()) {
                for (const auto& item : uuid_node) {
                    if (item.as<std::string>() == device_uuid) {
                        log_info("Found matching firmware information: " + entry.path().string());
                        context.firmware_info_path = entry.path().string();
                        return true;
                    }
                }
            } else {
                if (uuid_node.as<std::string>() == device_uuid) {
                    log_info("Found matching firmware information: " + entry.path().string());
                    context.firmware_info_path = entry.path().string();
                    return true;
                }
            }
        } catch (const YAML::Exception& e) {
            log_debug(context.log_level, "Error parsing " + entry.path().string() + ": " + e.what());
            continue;
        }
    }

    return false;
}

bool verify_firmware_details(hsb_flasher_context& context)
{
    YAML::Node config = YAML::LoadFile(context.firmware_info_path);
    if (!config["firmware_versions"]) {
        log_info("Error: No firmware versions found in YAML file: " + context.firmware_info_path);
        return false;
    }

    // Search for the target firmware version
    for (const auto& version_node : config["firmware_versions"]) {
        int64_t yaml_version = version_node["version"].as<int64_t>();
        if (yaml_version == std::stoi(context.target_version, nullptr, 16)) {
            log_info("Found target firmware version: 0x" + context.target_version);

            if (version_node["clnx"]) {
                context.clnx.location = version_node["clnx"][0]["location"].as<std::string>();
                context.clnx.md5 = version_node["clnx"][0]["md5"].as<std::string>();
                context.clnx.size = version_node["clnx"][0]["size"].as<size_t>();
            }

            context.cpnx.location = version_node["cpnx"][0]["location"].as<std::string>();
            context.cpnx.md5 = version_node["cpnx"][0]["md5"].as<std::string>();
            context.cpnx.size = version_node["cpnx"][0]["size"].as<size_t>();

            return true;
        }
    }

    return false;
}

static bool is_remote(const std::string& location)
{
    return location.find("http") == 0;
}

static size_t write_file_callback(void* contents, size_t size, size_t nmemb, FILE* fp)
{
    return fwrite(contents, size, nmemb, fp);
}

static bool download_firmware_file(hsb_flasher_log_level log_level,
    const std::string& url, const std::string& local_path,
    const std::filesystem::path& allowed_base_dir)
{
    // Reject non-HTTPS URLs
    if (url.rfind("https://", 0) != 0) {
        log_info("Error: only HTTPS URLs are allowed, got: " + url);
        return false;
    }

    // Validate and normalize local_path so it stays inside allowed_base_dir
    std::filesystem::path normalized = std::filesystem::path(local_path).lexically_normal();
    std::filesystem::path base_canonical = std::filesystem::weakly_canonical(allowed_base_dir);
    std::filesystem::path dest_canonical = std::filesystem::weakly_canonical(normalized);

    auto [base_end, dest_it] = std::mismatch(
        base_canonical.begin(), base_canonical.end(), dest_canonical.begin(), dest_canonical.end());
    if (base_end != base_canonical.end()) {
        log_info("Error: download path escapes allowed directory: " + dest_canonical.string());
        return false;
    }

    std::string safe_path = dest_canonical.string();

    if (std::filesystem::exists(safe_path)) {
        log_debug(log_level, "Firmware already exists: " + safe_path);
        return true;
    }

    std::filesystem::path parent = std::filesystem::path(safe_path).parent_path();
    if (!parent.empty() && !std::filesystem::exists(parent)) {
        std::filesystem::create_directories(parent);
    }

    log_info("Downloading firmware: " + url);

    CURL* curl = curl_easy_init();
    if (!curl) {
        log_info("Error: failed to initialize libcurl");
        return false;
    }

    FILE* fp = fopen(safe_path.c_str(), "wb");
    if (!fp) {
        log_info("Error: failed to open output file: " + safe_path);
        curl_easy_cleanup(curl);
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_file_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 5L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);

    // TLS verification
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    const char* ca_bundle = std::getenv("CURL_CA_BUNDLE");
    if (ca_bundle) {
        curl_easy_setopt(curl, CURLOPT_CAINFO, ca_bundle);
    }

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(curl);
    fclose(fp);

    if (res != CURLE_OK) {
        log_info("Error: download failed for " + url + ": "
            + curl_easy_strerror(res)
            + " (HTTP " + std::to_string(http_code) + ")");
        if (std::filesystem::exists(safe_path)) {
            std::filesystem::remove(safe_path);
        }
        return false;
    }

    log_info("Downloaded: " + safe_path);
    return true;
}

static std::string md5_file(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return "";
    }

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        return "";
    }

    if (EVP_DigestInit_ex(ctx, EVP_md5(), nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        return "";
    }

    char buffer[8192];
    while (file.read(buffer, sizeof(buffer))) {
        EVP_DigestUpdate(ctx, buffer, file.gcount());
    }
    if (file.gcount() > 0) {
        EVP_DigestUpdate(ctx, buffer, file.gcount());
    }

    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_len = 0;
    EVP_DigestFinal_ex(ctx, digest, &digest_len);
    EVP_MD_CTX_free(ctx);

    std::ostringstream hex_stream;
    for (unsigned int i = 0; i < digest_len; ++i) {
        hex_stream << std::hex << std::setw(2) << std::setfill('0')
                   << static_cast<int>(digest[i]);
    }
    return hex_stream.str();
}

static bool resolve_and_verify_firmware(hsb_flasher_log_level log_level,
    target_firmware_info& fw, const std::filesystem::path& firmware_dir)
{
    if (is_remote(fw.location)) {
        size_t pos = fw.location.find_last_of('/');
        std::string filename = (pos != std::string::npos) ? fw.location.substr(pos + 1) : fw.location;
        fw.local_location = (firmware_dir / filename).string();

        if (!download_firmware_file(log_level, fw.location, fw.local_location, firmware_dir)) {
            return false;
        }
    } else {
        fw.local_location = fw.location;
    }

    if (!std::filesystem::exists(fw.local_location)) {
        log_info("Error: Firmware not found: " + fw.local_location);
        return false;
    }
    if (std::filesystem::file_size(fw.local_location) != fw.size) {
        log_info("Error: Firmware size mismatch: " + fw.local_location);
        return false;
    }
    if (md5_file(fw.local_location) != fw.md5) {
        log_info("Error: Firmware MD5 mismatch: " + fw.local_location);
        return false;
    }

    return true;
}

bool fetch_target_firmware(hsb_flasher_context& context)
{
    std::filesystem::path exe_dir = get_executable_directory();
    std::filesystem::path firmware_dir = exe_dir / "firmware";

    if (!context.clnx.location.empty()) {
        if (!resolve_and_verify_firmware(context.log_level, context.clnx, firmware_dir)) {
            return false;
        }
    }

    if (!resolve_and_verify_firmware(context.log_level, context.cpnx, firmware_dir)) {
        return false;
    }

    return true;
}

} // namespace hsb_flasher
