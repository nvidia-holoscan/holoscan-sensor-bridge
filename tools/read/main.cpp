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

#include "read.hpp"

#include <getopt.h>
#include <iostream>

static void print_usage(const char* program_name)
{
    std::cout << "Usage: " << program_name << " [OPTIONS] <address>" << std::endl
              << "  -H, --hololink=IP  Hololink IP address (default 192.168.0.2)" << std::endl;
}

int main(int argc, char* argv[])
{
    static struct option long_options[] = {
        { "hololink", required_argument, nullptr, 'H' },
        { "help", no_argument, nullptr, 'h' },
        { nullptr, 0, nullptr, 0 }
    };

    std::string hololink_ip = "192.168.0.2";

    int opt = 0;
    while ((opt = getopt_long(argc, argv, "H:h", long_options, nullptr)) != -1) {
        switch (opt) {
        case 'H':
            hololink_ip = optarg;
            break;
        case 'h':
            print_usage(argv[0]);
            return 0;
        default:
            print_usage(argv[0]);
            return 1;
        }
    }

    // Remaining argument is read address
    int remaining = argc - optind;
    if (remaining != 1) {
        std::cerr << "Error: Expected single read address.\n";
        print_usage(argv[0]);
        return 1;
    }
    int32_t address = std::stoi(argv[optind]);

    std::cout << "Reading 0x" << std::hex << address << std::dec << " from " << hololink_ip << "..." << std::endl;
    uint32_t value = hololink::tools::read(hololink_ip, address);
    std::cout << "  0x" << std::hex << address << std::dec << "=" << value << std::endl;

    return 0;
}
