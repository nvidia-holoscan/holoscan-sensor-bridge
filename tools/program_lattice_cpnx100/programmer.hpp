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

#ifndef PROGRAMMER_HPP
#define PROGRAMMER_HPP

#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <yaml-cpp/yaml.h>

#include <hololink/core/data_channel.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/metadata.hpp>

namespace hololink {

class Programmer {
public:
    struct Args {
        std::string hololink_ip = "192.168.0.2";
        bool force = false;
        hololink::logging::HsbLogLevel log_level = hololink::logging::HSB_LOG_LEVEL_INFO;
        std::string manifest;
        std::string archive;
        bool skip_program_clnx = false;
        bool skip_verify_clnx = false;
        bool skip_program_cpnx = false;
        bool skip_verify_cpnx = false;
        bool accept_eula = false;
        bool skip_power_cycle = false;
    };

    Programmer(const Args& args, const std::string& manifest_filename);
    ~Programmer();

    void fetch_manifest(const std::string& section);
    std::shared_ptr<Hololink> hololink(const Metadata& channel_metadata);
    bool check_fpga_uuid(const std::string& fpga_uuid);
    bool program_and_verify_images(std::shared_ptr<Hololink> hololink);
    void program_clnx(std::shared_ptr<Hololink> hololink, uint32_t spi_controller_address, const std::vector<uint8_t>& content);
    bool verify_clnx(std::shared_ptr<Hololink> hololink, uint32_t spi_controller_address, const std::vector<uint8_t>& content);
    void program_cpnx(std::shared_ptr<Hololink> hololink, uint32_t spi_controller_address, const std::vector<uint8_t>& content);
    bool verify_cpnx(std::shared_ptr<Hololink> hololink, uint32_t spi_controller_address, const std::vector<uint8_t>& content);
    void power_cycle();
    std::vector<uint8_t> fetch_content(const std::string& content_name);
    void check_eula();
    void check_images();

private:
    static std::atomic<int> instances;
    Args args_;
    std::string manifest_filename_;
    bool skip_eula_;
    YAML::Node manifest_node_;
    std::unordered_map<std::string, std::vector<uint8_t>> content_;
};

} // namespace hololink

#endif // PROGRAMMER_HPP
