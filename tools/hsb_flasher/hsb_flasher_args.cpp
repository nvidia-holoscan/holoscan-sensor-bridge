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

#include "hsb_flasher_args.hpp"

#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <regex>

namespace hsb_flasher {

void print_usage(const char* program_name)
{
    std::cout
        << "Usage: " << program_name << " -H HOLOLINK_IP -t TARGET_VERSION [OPTIONS]" << std::endl
        << "       " << program_name << " -H HOLOLINK_IP --cpnx PATH [--clnx PATH] [--flash-version VER] [OPTIONS]" << std::endl
        << std::endl
        << "Examples:" << std::endl
        << "  " << program_name << " -H 192.168.0.2 -t 2507" << std::endl
        << "  " << program_name << " -H 192.168.0.2 --cpnx cpnx.bin" << std::endl
        << "  " << program_name << " -H 192.168.0.2 --clnx clnx.bin --cpnx cpnx.bin --flash-version 2507" << std::endl
        << std::endl
        << "Required:" << std::endl
        << "  -H, --hololink IP            IP address of the Hololink device" << std::endl
        << std::endl
        << "Standard mode (discovery + YAML):" << std::endl
        << "  -t, --target-version VER     Target version to flash (Hexadecimal)" << std::endl
        << std::endl
        << "Direct flash mode (skip YAML lookup + firmware fetch):" << std::endl
        << "  --cpnx PATH                  Path to CPNX firmware file" << std::endl
        << "  --clnx PATH                  Path to CLNX firmware file (optional)" << std::endl
        << "  --flash-version VER          Override version for flash strategy selection (optional, Hexadecimal)" << std::endl
        << "                               If omitted, the device's discovered version is used" << std::endl
        << std::endl
        << "Options:" << std::endl
        << "  -u, --uuid UUID              FPGA UUID (for legacy devices that don't report it)" << std::endl
        << "  -e, --enumeration-timeout S  Enumeration timeout in seconds (default: 3)" << std::endl
        << "  -v, --verbose                Enable verbose output" << std::endl
        << "  -h, --help                   Display this information" << std::endl;
}

bool parse_arguments(int argc, char** argv, hsb_flasher_context& context)
{
    enum long_option_ids {
        OPT_CLNX = 256,
        OPT_CPNX,
        OPT_FLASH_VERSION,
    };

    static struct option long_options[] = {
        { "hololink", required_argument, nullptr, 'H' },
        { "target-version", required_argument, nullptr, 't' },
        { "uuid", required_argument, nullptr, 'u' },
        { "enumeration-timeout", required_argument, nullptr, 'e' },
        { "verbose", no_argument, nullptr, 'v' },
        { "help", no_argument, nullptr, 'h' },
        { "clnx", required_argument, nullptr, OPT_CLNX },
        { "cpnx", required_argument, nullptr, OPT_CPNX },
        { "flash-version", required_argument, nullptr, OPT_FLASH_VERSION },
        { nullptr, 0, nullptr, 0 }
    };

    context.log_level = hsb_flasher_log_level::INFO;
    context.timeout = 3.0f;

    int c;
    while ((c = getopt_long(argc, argv, "H:t:u:e:vh", long_options, nullptr)) != -1) {
        switch (c) {
        case 'H':
            context.hololink_ip = optarg;
            break;
        case 't':
            context.target_version = optarg;
            break;
        case 'u':
            context.fpga_uuid_override = optarg;
            break;
        case 'e':
            context.timeout = std::stof(optarg);
            break;
        case 'v':
            context.log_level = hsb_flasher_log_level::DEBUG;
            break;
        case 'h':
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
            break;
        case OPT_CLNX:
            context.direct_flash.clnx_path = optarg;
            break;
        case OPT_CPNX:
            context.direct_flash.cpnx_path = optarg;
            break;
        case OPT_FLASH_VERSION:
            context.direct_flash.flash_version = optarg;
            break;
        default:
            return false;
        }
    }

    if (context.hololink_ip.empty()) {
        log_info("Error: Hololink IP address is required");
        return false;
    } else if (!std::regex_match(context.hololink_ip,
                   std::regex("^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}"
                              "(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"))) {
        log_info("Error: Hololink IP address must be in the format 192.168.0.2");
        return false;
    }

    if (context.direct_flash.has_any() && !context.direct_flash.enabled()) {
        log_info("Error: --cpnx is required for direct flash mode (--clnx and --flash-version are optional)");
        return false;
    }

    if (context.direct_flash.enabled() && !context.direct_flash.flash_version.empty()) {
        if (!std::regex_match(context.direct_flash.flash_version, std::regex("^[0-9A-Fa-f]+$"))) {
            log_info("Error: Flash version must be a hexadecimal number (without 0x prefix)");
            return false;
        }
    }

    if (!context.direct_flash.enabled()) {
        if (context.target_version.empty()) {
            log_info("Error: Target version is required (or use --clnx, --cpnx, --flash-version for direct mode)");
            return false;
        } else if (!std::regex_match(context.target_version, std::regex("^[0-9A-Fa-f]+$"))) {
            log_info("Error: Target version must be a hexadecimal number (without 0x prefix)");
            return false;
        }
    }

    if (context.timeout <= 0.0f) {
        log_info("Error: Timeout must be greater than 0");
        return false;
    }

    return true;
}

} // namespace hsb_flasher
