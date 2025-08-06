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

#include <cstdlib>
#include <getopt.h>
#include <iostream>

#include <hololink/core/enumerator.hpp>
#include <hololink/core/logging_internal.hpp>

#include "programmer.hpp"

namespace {

hololink::Metadata manual_enumeration(const hololink::Programmer::Args& args)
{
    hololink::Metadata metadata;
    metadata["control_port"] = 8192;
    metadata["hsb_ip_version"] = 0x2502;
    metadata["peer_ip"] = args.hololink_ip;
    metadata["sequence_number_checking"] = 0;
    metadata["serial_number"] = "100";
    metadata["fpga_uuid"] = "f1627640-b4dc-48af-a360-c55b09b3d230";

    hololink::DataChannel::use_data_plane_configuration(metadata, 0);
    hololink::DataChannel::use_sensor(metadata, 0);
    return metadata;
}

void print_usage(const char* program_name)
{
    std::cout << "Usage: " << program_name << " [OPTIONS] <manifest>" << std::endl
              << "Options:" << std::endl
              << "  --hololink=IP        IP address of Hololink board (default: 192.168.0.2)" << std::endl
              << "  --force              Don't rely on enumeration data for device connection" << std::endl
              << "  --log-level=LEVEL    Logging level to display" << std::endl
              << "  --skip-program-cpnx  Skip programming CPNX" << std::endl
              << "  --skip-verify-cpnx   Skip verifying CPNX" << std::endl
              << "  --accept-eula        Provide non-interactive EULA acceptance" << std::endl
              << "  --skip-power-cycle   Don't wait for confirmation of power cycle" << std::endl
              << "  -h, --help           Display this information" << std::endl;
}

} // anonymous namespace

int main(int argc, char** argv)
{
    hololink::Programmer::Args args;

    static struct option long_options[] = {
        { "hololink", required_argument, nullptr, 0 },
        { "force", no_argument, nullptr, 0 },
        { "log-level", required_argument, nullptr, 0 },
        { "skip-program-cpnx", no_argument, nullptr, 0 },
        { "skip-verify-cpnx", no_argument, nullptr, 0 },
        { "accept-eula", no_argument, nullptr, 0 },
        { "skip-power-cycle", no_argument, nullptr, 0 },
        { "help", no_argument, nullptr, 'h' },
        { nullptr, 0, nullptr, 0 }
    };

    int option_index = 0;
    int c;

    while ((c = getopt_long(argc, argv, "h", long_options, &option_index)) != -1) {
        if (c == 0) {
            const struct option* cur_option = &long_options[option_index];
            std::string option_name(cur_option->name);

            if (option_name == "hololink") {
                args.hololink_ip = optarg;
            } else if (option_name == "force") {
                args.force = true;
            } else if (option_name == "log-level") {
                args.log_level = static_cast<hololink::logging::HsbLogLevel>(std::stoi(optarg));
            } else if (option_name == "skip-program-cpnx") {
                args.skip_program_cpnx = true;
            } else if (option_name == "skip-verify-cpnx") {
                args.skip_verify_cpnx = true;
            } else if (option_name == "accept-eula") {
                args.accept_eula = true;
            } else if (option_name == "skip-power-cycle") {
                args.skip_power_cycle = true;
            }
        } else if (c == 'h') {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        } else {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    // Check for required manifest argument
    if (optind >= argc) {
        std::cerr << "Error: manifest file is required" << std::endl;
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    args.manifest = argv[optind];

    // Set logging level
    hololink::logging::hsb_log_level = static_cast<hololink::logging::HsbLogLevel>(args.log_level);

    std::cout << "Initializing." << std::endl;

    hololink::Programmer programmer(args, args.manifest);
    programmer.fetch_manifest("hololink");
    programmer.check_eula();
    programmer.check_images();

    hololink::Metadata channel_metadata;
    if (args.force) {
        channel_metadata = manual_enumeration(args);
    } else {
        channel_metadata = hololink::Enumerator::find_channel(args.hololink_ip);
    }

    auto hololink = programmer.hololink(channel_metadata);
    auto ok = programmer.program_and_verify_images(hololink);
    programmer.power_cycle();
    if (ok) {
        return EXIT_SUCCESS;
    }
    return EXIT_FAILURE;
}