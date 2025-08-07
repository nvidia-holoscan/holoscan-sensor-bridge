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

#include <curl/curl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <openssl/evp.h>
#include <sstream>

#include <fmt/format.h>
#include <hololink/core/logging_internal.hpp>

#include "cpnx_flash.hpp"

namespace hololink {

std::atomic<int> Programmer::instances { 0 };
constexpr int64_t MINIMUM_HSB_IP_VERSION = 0x2412;

Programmer::Programmer(const Args& args, const std::string& manifest_filename)
    : args_(args)
    , manifest_filename_(manifest_filename)
    , skip_eula_(false)
{
    if (instances++ == 0) {
        curl_global_init(CURL_GLOBAL_ALL);
    }
}

Programmer::~Programmer()
{
    if (--instances == 0) {
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
        auto ip_opt = channel_metadata.get<std::string>("peer_ip");
        if (!ip_opt) {
            throw std::runtime_error("No peer_ip in channel metadata");
        }
        std::string ip = ip_opt.value();
        throw std::runtime_error(fmt::format("Sensor bridge ip={} ({}) isn't supported by this manifest file.", ip, fpga_uuid.value()));
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
    if (fpga_uuid != LEOPARD_EAGLE_UUID) {
        // Sorry for the inconsistent use of failure reporting...
        // fix this only if it actually becomes important.
        throw std::runtime_error(fmt::format("Unexpected fpga_uuid in channel metadata; expected {} but got {}",
            LEOPARD_EAGLE_UUID, fpga_uuid));
    }
    auto fpga_uuids = manifest_node_["fpga_uuid"];
    for (const auto& uuid : fpga_uuids) {
        if (uuid.as<std::string>() == fpga_uuid) {
            return true;
        }
    }
    return false;
}

bool Programmer::program_and_verify_images(std::shared_ptr<Hololink> hololink)
{
    bool ok = true;
    if (content_.find("cpnx") != content_.end()) {
        program_cpnx(hololink, 0, content_["cpnx"]);
        ok = verify_cpnx(hololink, 0, content_["cpnx"]);
    }
    return ok;
}

void Programmer::program_cpnx(std::shared_ptr<Hololink> hololink, uint32_t spi_controller_address, const std::vector<uint8_t>& content)
{
    if (args_.skip_program_cpnx) {
        HSB_LOG_INFO("Skipping programming CPNX per command-line instructions.");
        return;
    }
    CpnxFlash cpnx_flash("CPNX", hololink, spi_controller_address);
    cpnx_flash.program(content);
}

bool Programmer::verify_cpnx(std::shared_ptr<Hololink> hololink, uint32_t spi_controller_address, const std::vector<uint8_t>& content)
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

size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::vector<uint8_t>* userp)
{
    size_t total_size = size * nmemb;
    const uint8_t* data = static_cast<const uint8_t*>(contents);
    userp->insert(userp->end(), data, data + total_size);
    return total_size;
}

std::vector<uint8_t> Programmer::fetch_content(const std::string& content_name)
{
    auto content_metadata = manifest_node_["content"][content_name];
    if (!content_metadata) {
        throw std::runtime_error(fmt::format("No content \"{}\" found.", content_name));
    }

    std::string expected_md5 = content_metadata["md5"].as<std::string>();
    size_t expected_size = content_metadata["size"].as<size_t>();

    std::vector<uint8_t> content;

    if (content_metadata["url"]) {
        std::string url = content_metadata["url"].as<std::string>();

        CURL* curl = curl_easy_init();
        if (!curl) {
            throw std::runtime_error("Failed to initialize CURL");
        }

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
        curl_easy_setopt(curl, CURLOPT_MAXFILESIZE_LARGE, static_cast<curl_off_t>(expected_size));
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &content);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);

        CURLcode res = curl_easy_perform(curl);
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        curl_easy_cleanup(curl);

        if (res != CURLE_OK || http_code != 200) {
            throw std::runtime_error(fmt::format("Unable to fetch \"{}\"; status={}", url, http_code));
        }
    } else if (content_metadata["filename"]) {
        std::string filename = content_metadata["filename"].as<std::string>();
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error(fmt::format("Could not open file: {}", filename));
        }

        content = std::vector<uint8_t>(
            std::istreambuf_iterator<char>(file),
            std::istreambuf_iterator<char>());
    } else {
        throw std::runtime_error(fmt::format("No instructions for where to find {} are provided.", content_name));
    }

    size_t actual_size = content.size();
    HSB_LOG_DEBUG("expected_size={} actual_size={}", expected_size, actual_size);
    if (actual_size != expected_size) {
        throw std::runtime_error(fmt::format("content_name={} expected_size={} actual_size={}; aborted.", content_name, expected_size, actual_size));
    }

    // Calculate MD5
    unsigned char md5_digest[EVP_MAX_MD_SIZE];
    unsigned int md5_len;

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        throw std::runtime_error("Failed to create MD5 context");
    }

    if (EVP_DigestInit_ex(ctx, EVP_md5(), nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("Failed to initialize MD5");
    }

    if (EVP_DigestUpdate(ctx, content.data(), content.size()) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("Failed to update MD5");
    }

    if (EVP_DigestFinal_ex(ctx, md5_digest, &md5_len) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("Failed to finalize MD5");
    }

    EVP_MD_CTX_free(ctx);

    std::stringstream ss;
    for (unsigned int i = 0; i < md5_len; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(md5_digest[i]);
    }
    std::string actual_md5 = ss.str();

    HSB_LOG_DEBUG("expected_md5={} actual_md5={}", expected_md5, actual_md5);
    if (actual_md5 != expected_md5) {
        throw std::runtime_error(fmt::format("content_name={} expected_md5={} actual_md5={}; aborted.", content_name, expected_md5, actual_md5));
    }

    return content;
}

void Programmer::check_eula()
{
    if (skip_eula_) {
        HSB_LOG_DEBUG("Accepting EULA is not necessary.");
        return;
    }

    if (args_.accept_eula) {
        HSB_LOG_INFO("All EULAs are accepted via command-line switch.");
        return;
    }

    auto licenses = manifest_node_["licenses"];
    std::cout << "You must accept EULA terms in order to continue." << std::endl;

    if (licenses.size() == 0) {
        throw std::runtime_error("Malformed manifest file: no license documents listed.");
    }

    std::cout << "For each document, press <Space> to see the next page;" << std::endl;
    std::cout << "At the end of the document, enter <Q> to continue." << std::endl;
    std::cout << "To continue, press <Enter>: ";
    std::cin.get();

    for (const auto& license : licenses) {
        auto license_content = fetch_content(license.as<std::string>());
        std::string content_str(license_content.begin(), license_content.end());

        // Simple pager implementation
        std::istringstream iss(content_str);
        std::string line;
        int line_count = 0;
        while (std::getline(iss, line)) {
            std::cout << line << std::endl;
            line_count++;
            if (line_count % 20 == 0) {
                std::cout << "Press <Enter> for more, <Q> to quit: ";
                std::string input;
                std::getline(std::cin, input);
                if (input == "Q" || input == "q") {
                    break;
                }
            }
        }

        std::cout << "Press 'y' or 'Y' to accept this end user license agreement: ";
        std::string answer;
        std::getline(std::cin, answer);
        if (!answer.empty() && (answer[0] != 'Y' && answer[0] != 'y')) {
            throw std::runtime_error("Execution of this script requires an agreement with license terms.");
        }
    }
}

void Programmer::check_images()
{
    auto images = manifest_node_["images"];
    for (const auto& image_metadata : images) {
        std::string context = image_metadata["context"].as<std::string>();
        std::string content_name = image_metadata["content"].as<std::string>();
        HSB_LOG_INFO("context={} content_name={}", context, content_name);

        auto content = fetch_content(content_name);
        content_[context] = content;
    }
}

} // namespace hololink
