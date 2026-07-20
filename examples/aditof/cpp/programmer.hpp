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

#include <hololink/core/hololink.hpp>
#include <hololink/core/logging_internal.hpp>

// Forward declaration — avoids pulling the full Adcam header into every TU.
namespace hololink {
namespace sensors {
class Adcam;
}
} // namespace hololink

namespace hololink {

class Programmer {
  public:
    struct Args {
        std::string hololink_ip = "192.168.0.2";
        bool force = false;
        hololink::logging::HsbLogLevel log_level =
            hololink::logging::HSB_LOG_LEVEL_INFO;
        std::string manifest;
        std::string archive;
        bool accept_eula = false;
    };

    Programmer(const Args &args, const std::string &manifest_filename);
    ~Programmer();

    void fetch_manifest(const std::string &section);
    bool program_and_verify_images(
        std::shared_ptr<Hololink> hololink,
        std::shared_ptr<hololink::sensors::Adcam> adcam = nullptr);
    std::vector<uint8_t> fetch_content(const std::string &content_name);
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
