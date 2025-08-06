/*
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
 */

#include "programmer.hpp"

#include <algorithm>
#include <cctype>
#include <curl/curl.h>
#include <fstream>
#include <iostream>
#include <openssl/evp.h>
#include <yaml-cpp/yaml.h>

#include <fmt/format.h>
#include <hololink/core/logging_internal.hpp>

#include "clnx_flash.hpp"
#include "cpnx_flash.hpp"

namespace hololink {

std::atomic<int> Programmer::instances { 0 };
constexpr int64_t MINIMUM_HSB_IP_VERSION = 0x2412;

namespace {

    size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::vector<uint8_t>* userp)
    {
        size_t total_size = size * nmemb;
        userp->insert(userp->end(), static_cast<uint8_t*>(contents),
            static_cast<uint8_t*>(contents) + total_size);
        return total_size;
    }

    std::string calculate_md5(const std::vector<uint8_t>& data)
    {
        EVP_MD_CTX* context = EVP_MD_CTX_new();
        if (!context) {
            throw std::runtime_error("Failed to create MD5 context");
        }

        if (EVP_DigestInit_ex(context, EVP_md5(), nullptr) != 1) {
            EVP_MD_CTX_free(context);
            throw std::runtime_error("Failed to initialize MD5 digest");
        }

        if (EVP_DigestUpdate(context, data.data(), data.size()) != 1) {
            EVP_MD_CTX_free(context);
            throw std::runtime_error("Failed to update MD5 digest");
        }

        unsigned char digest[EVP_MAX_MD_SIZE];
        unsigned int digest_len;
        if (EVP_DigestFinal_ex(context, digest, &digest_len) != 1) {
            EVP_MD_CTX_free(context);
            throw std::runtime_error("Failed to finalize MD5 digest");
        }

        EVP_MD_CTX_free(context);

        std::string result;
        for (unsigned int i = 0; i < digest_len; i++) {
            char hex[3];
            snprintf(hex, sizeof(hex), "%02x", digest[i]);
            result += hex;
        }
        return result;
    }

} // anonymous namespace

Programmer::Programmer(const Args& args, const std::string& manifest_filename)
    : args_(args)
    , manifest_filename_(manifest_filename)
    , skip_eula_(false)
{
    // Increment instance counter
    int current_instances = ++instances;

    // Initialize CURL only if this is the first instance
    if (current_instances == 1) {
        auto err = curl_global_init(CURL_GLOBAL_ALL);
        if (err != CURLE_OK) {
            --instances; // Decrement on failure
            throw std::runtime_error(fmt::format("curl_global_init failed: {}", err));
        }
    }
}

Programmer::~Programmer()
{
    // Decrement instance counter
    int remaining_instances = --instances;

    // Cleanup CURL only if this is the last instance
    if (remaining_instances == 0) {
        curl_global_cleanup();
    }
}

void Programmer::fetch_manifest(const std::string& section)
{
    YAML::Node config = YAML::LoadFile(manifest_filename_);
    if (!config[section]) {
        throw std::runtime_error(fmt::format("Section '{}' not found in manifest", section));
    }

    // Store the YAML node directly instead of converting to std::any
    manifest_node_ = config[section];

    if (!manifest_node_["licenses"]) {
        skip_eula_ = true;
    }
}

std::shared_ptr<Hololink> Programmer::hololink(const Metadata& channel_metadata)
{
    auto fpga_uuid = channel_metadata.get<std::string>("fpga_uuid");
    if (!fpga_uuid) {
        throw std::runtime_error("No fpga_uuid in channel metadata");
    }
    if (!check_fpga_uuid(fpga_uuid.value())) {
        auto peer_ip = channel_metadata.get<std::string>("peer_ip");
        if (!peer_ip) {
            throw std::runtime_error("No peer_ip in channel metadata");
        }
        throw std::runtime_error(fmt::format("Sensor bridge ip={} ({}) isn't supported by this manifest file.",
            peer_ip.value(), fpga_uuid.value()));
    }

    auto hsb_ip_version = channel_metadata.get<int64_t>("hsb_ip_version"); // or None
    if (!hsb_ip_version) {
        throw UnsupportedVersion("No 'hsb_ip_version' field found.");
    }
    if (hsb_ip_version.value() < MINIMUM_HSB_IP_VERSION) {
        throw UnsupportedVersion(fmt::format("hsb_ip_version={:#X}; minimum supported version={:#X}.",
            hsb_ip_version.value(), MINIMUM_HSB_IP_VERSION));
    }

    auto hololink = Hololink::from_enumeration_metadata(channel_metadata);
    DataChannel hololink_channel(channel_metadata, hololink);
    hololink->start();
    return hololink;
}

bool Programmer::check_fpga_uuid(const std::string& fpga_uuid)
{
    if (fpga_uuid != HOLOLINK_LITE_UUID) {
        // Sorry for the inconsistent use of failure reporting...
        // fix this only if it actually becomes important.
        throw std::runtime_error(fmt::format("Unexpected fpga_uuid in channel metadata; expected {} but got {}",
            HOLOLINK_LITE_UUID, fpga_uuid));
    }
    auto fpga_uuids = manifest_node_["fpga_uuid"];
    if (fpga_uuids.IsNull()) {
        return false;
    }

    // Check if the fpga_uuid is in the list of supported UUIDs
    if (fpga_uuids.IsSequence()) {
        for (const auto& uuid : fpga_uuids) {
            if (uuid.as<std::string>() == fpga_uuid) {
                return true;
            }
        }
    } else if (fpga_uuids.IsScalar()) {
        // Single UUID case
        return fpga_uuids.as<std::string>() == fpga_uuid;
    }

    return false;
}

bool Programmer::program_and_verify_images(std::shared_ptr<Hololink> hololink)
{
    bool ok = true;
    if (content_.find("clnx") != content_.end()) {
        program_clnx(hololink, CLNX_SPI_BUS, content_["clnx"]);
        if (!verify_clnx(hololink, CLNX_SPI_BUS, content_["clnx"])) {
            ok = false;
        }
    }
    if (content_.find("cpnx") != content_.end()) {
        program_cpnx(hololink, CPNX_SPI_BUS, content_["cpnx"]);
        if (!verify_cpnx(hololink, CPNX_SPI_BUS, content_["cpnx"])) {
            ok = false;
        }
    }
    return ok;
}

void Programmer::program_clnx(std::shared_ptr<Hololink> hololink, uint32_t spi_controller_address,
    const std::vector<uint8_t>& content)
{
    if (args_.skip_program_clnx) {
        HSB_LOG_INFO("Skipping programming CLNX per command-line instructions.");
        return;
    }
    ClnxFlash clnx_flash("CLNX", hololink, spi_controller_address);
    clnx_flash.program(content);
}

bool Programmer::verify_clnx(std::shared_ptr<Hololink> hololink, uint32_t spi_controller_address,
    const std::vector<uint8_t>& content)
{
    if (args_.skip_verify_clnx) {
        HSB_LOG_INFO("Skipping verification CLNX per command-line instructions.");
        return true;
    }
    ClnxFlash clnx_flash("CLNX", hololink, spi_controller_address);
    return clnx_flash.verify(content);
}

void Programmer::program_cpnx(std::shared_ptr<Hololink> hololink, uint32_t spi_controller_address,
    const std::vector<uint8_t>& content)
{
    if (args_.skip_program_cpnx) {
        HSB_LOG_INFO("Skipping programming CPNX per command-line instructions.");
        return;
    }
    CpnxFlash cpnx_flash("CPNX", hololink, spi_controller_address);
    cpnx_flash.program(content);
}

bool Programmer::verify_cpnx(std::shared_ptr<Hololink> hololink, uint32_t spi_controller_address,
    const std::vector<uint8_t>& content)
{
    if (args_.skip_verify_cpnx) {
        HSB_LOG_INFO("Skipping verification CPNX per command-line instructions.");
        return true;
    }
    CpnxFlash cpnx_flash("CPNX", hololink, spi_controller_address);
    return cpnx_flash.verify(content);
}

void Programmer::power_cycle()
{
    std::cout << "You must now physically power cycle the sensor bridge device." << std::endl;
    if (args_.skip_power_cycle) {
        return;
    }
    std::cout << "Press <Enter> to continue: ";
    std::cin.get();
}

std::vector<uint8_t> Programmer::fetch_content(const std::string& content_name)
{
    auto content_metadata = manifest_node_["content"][content_name];
    if (content_metadata.IsNull()) {
        throw std::runtime_error(fmt::format("No content \"{}\" found.", content_name));
    }

    std::string expected_md5 = content_metadata["md5"].as<std::string>();
    size_t expected_size = content_metadata["size"].as<size_t>();

    std::vector<uint8_t> content;

    if (content_metadata["url"]) {
        // Download from URL
        std::string url = content_metadata["url"].as<std::string>();

        CURL* curl = curl_easy_init();
        if (!curl) {
            throw std::runtime_error("Failed to initialize CURL");
        }

        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: binary/octet-stream");

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L); // seconds
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L); // seconds
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &content);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L); // Follow redirects
        curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 5L); // Maximum 5 redirects

        CURLcode res = curl_easy_perform(curl);
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        if (res != CURLE_OK || http_code != 200) {
            throw std::runtime_error(fmt::format("Unable to fetch \"{}\"; status={}", url, http_code));
        }
    } else if (content_metadata["filename"]) {
        // Read from local file
        std::string filename = content_metadata["filename"].as<std::string>();
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error(fmt::format("Unable to open file: {}", filename));
        }

        content = std::vector<uint8_t>((std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>());
    } else {
        throw std::runtime_error(fmt::format("No instructions for where to find {} are provided.", content_name));
    }

    // Verify size
    size_t actual_size = content.size();
    HSB_LOG_DEBUG("expected_size={} actual_size={}", expected_size, actual_size);
    if (actual_size != expected_size) {
        throw std::runtime_error(fmt::format("{} expected_size={} actual_size={}; aborted.",
            content_name, expected_size, actual_size));
    }

    // Verify MD5
    std::string actual_md5 = calculate_md5(content);
    HSB_LOG_DEBUG("expected_md5={} actual_md5={}", expected_md5, actual_md5);
    auto to_lower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(),
            [](unsigned char c) { return std::tolower(c); });
        return s;
    };
    if (to_lower(actual_md5) != to_lower(expected_md5)) {
        throw std::runtime_error(fmt::format("{} expected_md5={} actual_md5={}; aborted.",
            content_name, expected_md5, actual_md5));
    }

    return content;
}

void Programmer::check_eula()
{
    if (skip_eula_) {
        HSB_LOG_TRACE("Accepting EULA is not necessary.");
        return;
    }
    if (args_.accept_eula) {
        HSB_LOG_INFO("All EULAs are accepted via command-line switch.");
        return;
    }

    auto licenses = manifest_node_["licenses"];
    if (!licenses) {
        throw std::runtime_error("No licenses found in manifest");
    }
    if (!licenses.IsSequence() || licenses.size() == 0) {
        throw std::runtime_error("Invalid licenses found in manifest");
    }

    std::cout << "You must accept EULA terms in order to continue." << std::endl;
    std::cout << "For each document, press <Space> to see the next page;" << std::endl;
    std::cout << "At the end of the document, enter <Q> to continue." << std::endl;
    std::cout << "To continue, press <Enter>: ";
    std::cin.get();

    for (const auto& license : licenses) {
        std::string license_name = license.as<std::string>();
        std::vector<uint8_t> license_content = fetch_content(license_name);

        // Convert binary content to string for display
        std::string license_text(license_content.begin(), license_content.end());

        // Display license text using pager (simplified - just print to console)
        std::cout << "\n=== LICENSE: " << license_name << " ===" << std::endl;
        std::cout << license_text << std::endl;
        std::cout << "=== END LICENSE ===" << std::endl;

        std::cout << "Press 'y' or 'Y' to accept this end user license agreement: ";
        std::string answer;
        std::getline(std::cin, answer);
        if (answer.empty() || (answer[0] != 'y' && answer[0] != 'Y')) {
            throw std::runtime_error("Execution of this script requires an agreement with license terms.");
        }
    }
}

void Programmer::check_images()
{
    auto images = manifest_node_["images"];
    if (!images) {
        throw std::runtime_error("No images section found in manifest");
    }

    content_.clear();
    for (const auto& image_metadata : images) {
        std::string context = image_metadata["context"].as<std::string>();
        std::string content_name = image_metadata["content"].as<std::string>();
        HSB_LOG_INFO("context={} content_name={}", context, content_name);

        std::vector<uint8_t> content = fetch_content(content_name);
        content_[context] = content;
    }
}

} // namespace hololink
