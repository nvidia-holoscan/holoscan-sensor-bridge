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

#include "enumerate.hpp"

#include <getopt.h>
#include <iostream>

static void print_usage(const char* program_name)
{
    std::cout << "Usage: " << program_name << " [OPTIONS]" << std::endl
              << "  -i, --interface=IFACE  Network interface to use" << std::endl
              << "  -t, --timeout=SECONDS  Timeout in seconds" << std::endl;
}

int main(int argc, char* argv[])
{
    static struct option long_options[] = {
        { "interface", required_argument, nullptr, 'i' },
        { "timeout", required_argument, nullptr, 't' },
        { "help", no_argument, nullptr, 'h' },
        { nullptr, 0, nullptr, 0 }
    };

    std::string interface;
    int timeout = 0;

    int opt = 0;
    while ((opt = getopt_long(argc, argv, "i:t:h", long_options, nullptr)) != -1) {
        switch (opt) {
        case 'i':
            interface = optarg;
            break;
        case 't':
            timeout = std::stoi(optarg);
            break;
        case 'h':
            print_usage(argv[0]);
            return 0;
        default:
            print_usage(argv[0]);
            return 1;
        }
    }

    hololink::tools::enumerate(interface, timeout);

    return 0;
}
