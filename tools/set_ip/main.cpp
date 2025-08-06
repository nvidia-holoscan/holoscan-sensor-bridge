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

#include "set_ip.hpp"

#include <algorithm>
#include <getopt.h>
#include <iostream>

static void print_usage(const char* program_name)
{
    std::cout << "Usage: " << program_name << " [OPTIONS] <mac1> <ip1> [<mac2> <ip2>]\n"
              << "  -i, --interface=IFACE  Network interface to use\n"
              << "  -o, --one-time         Exit after setting IP(s)" << std::endl;
}

static std::string to_upper(const std::string& input)
{
    std::string result = input;
    std::transform(result.begin(), result.end(), result.begin(),
        [](unsigned char c) { return std::toupper(c); });
    return result;
}

int main(int argc, char* argv[])
{
    static struct option long_options[] = {
        { "interface", required_argument, nullptr, 'i' },
        { "one-time", no_argument, nullptr, 'o' },
        { "help", no_argument, nullptr, 'h' },
        { nullptr, 0, nullptr, 0 }
    };

    std::string interface;
    bool one_time = false;

    int opt = 0;
    while ((opt = getopt_long(argc, argv, "i:oh", long_options, nullptr)) != -1) {
        switch (opt) {
        case 'i':
            interface = optarg;
            break;
        case 'o':
            one_time = true;
            break;
        case 'h':
            print_usage(argv[0]);
            return 0;
        default:
            print_usage(argv[0]);
            return 1;
        }
    }

    // Remaining arguments are MAC/IP pairs
    int remaining = argc - optind;
    if (remaining == 0 || remaining % 2 != 0) {
        std::cerr << "Error: Expected an even number of MAC/IP arguments.\n";
        print_usage(argv[0]);
        return 1;
    }

    std::unordered_map<std::string, std::string> mac_ip_map;
    for (int i = optind; i < argc; i += 2) {
        std::string mac = to_upper(argv[i]);
        std::string ip = argv[i + 1];
        if (mac_ip_map.find(mac) != mac_ip_map.end()) {
            std::cerr << "Warning: Duplicate MAC address '" << mac << "' detected. Overwriting previous IP.\n";
        }
        mac_ip_map[mac] = ip;
    }

    std::cout << "Setting the following addresses:\n";
    for (const auto& entry : mac_ip_map) {
        std::cout << "  " << entry.first << " to " << entry.second << std::endl;
    }

    hololink::tools::set_ip(mac_ip_map, interface, one_time);

    return 0;
}
