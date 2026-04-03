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

#include "hsb_flasher.hpp"

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace hsb_flasher {

class PythonFlasher : public IFlasher {
public:
    PythonFlasher(const std::string& module_name,
        const std::filesystem::path& strategies_dir,
        const hsb_flasher_context& context)
        : module_name_(module_name)
        , strategies_dir_(strategies_dir)
        , context_(context)
    {
    }

    bool flash(const std::string& clnx_path, const std::string& cpnx_path) override
    {
        std::stringstream python_code;
        python_code << "import sys; "
                    << "sys.path.insert(0, '" << strategies_dir_.string() << "'); "
                    << "import " << module_name_ << " as m; "
                    << "exit(0 if m.do_flash("
                    << "'" << context_.enumeration_metadata.get<std::string>("fpga_uuid").value_or("N/A") << "', "
                    << "0x" << std::hex << context_.enumeration_metadata.get<int64_t>("hsb_ip_version").value_or(0) << ", "
                    << "'" << context_.enumeration_metadata.get<std::string>("mac_id").value_or("N/A") << "', "
                    << "'" << context_.enumeration_metadata.get<std::string>("peer_ip").value_or("N/A") << "', "
                    << "'" << clnx_path << "', "
                    << "'" << cpnx_path << "') else 1)";

        std::string cmd = "python3 -c \"" + python_code.str() + "\"";

        log_info("Executing flasher: " + module_name_);
        log_debug(context_.log_level, "Command: " + cmd);

        int result = std::system(cmd.c_str());
        return result == 0;
    }

private:
    std::string module_name_;
    std::filesystem::path strategies_dir_;
    hsb_flasher_context context_;
};

// Query a Python module to see if it supports the given fpga_uuid and version
static bool query_flasher_support(const std::string& module_name,
    const std::filesystem::path& strategies_dir,
    const std::string& fpga_uuid,
    int64_t version,
    hsb_flasher_log_level log_level)
{
    std::stringstream python_code;
    python_code << "import sys; "
                << "sys.path.insert(0, '" << strategies_dir.string() << "'); "
                << "import " << module_name << " as m; "
                << "exit(0 if m.supports('" << fpga_uuid << "', 0x" << std::hex << version << ") else 1)";

    std::string cmd = "python3 -c \"" + python_code.str() + "\" > /dev/null 2>&1";

    log_debug(log_level, "Querying: " + module_name);

    int result = std::system(cmd.c_str());
    return result == 0;
}

std::unique_ptr<IFlasher> get_flasher(const hsb_flasher_context& context)
{
    namespace fs = std::filesystem;

    fs::path exe_path;
    try {
        exe_path = fs::read_symlink("/proc/self/exe");
    } catch (const fs::filesystem_error& e) {
        log_info("Failed to read /proc/self/exe: " + std::string(e.what()));
        return nullptr;
    }

    fs::path strategies_dir = exe_path.parent_path() / "firmware_flash_strategies";

    if (!fs::exists(strategies_dir)) {
        log_info("firmware_flash_strategies directory not found: " + strategies_dir.string());
        return nullptr;
    }

    log_debug(context.log_level, "Scanning for flash strategies in: " + strategies_dir.string());

    for (const auto& entry : fs::directory_iterator(strategies_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension() != ".py") {
            continue;
        }

        std::string module_name = entry.path().stem().string();

        if (module_name.empty() || module_name[0] == '_') {
            continue;
        }

        if (query_flasher_support(module_name,
                strategies_dir,
                context.enumeration_metadata.get<std::string>("fpga_uuid").value_or("N/A"),
                context.enumeration_metadata.get<int64_t>("hsb_ip_version").value_or(0),
                context.log_level)) {
            log_info("Found matching flasher: " + module_name);
            return std::make_unique<PythonFlasher>(module_name, strategies_dir, context);
        }
    }

    std::stringstream version_hex;
    version_hex << std::hex << context.enumeration_metadata.get<int64_t>("hsb_ip_version").value_or(0);
    log_info("No flasher found for fpga_uuid=" + context.enumeration_metadata.get<std::string>("fpga_uuid").value_or("N/A") + " version=0x" + version_hex.str());
    return nullptr;
}

} // namespace hsb_flasher
