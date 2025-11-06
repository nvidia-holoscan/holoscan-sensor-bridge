/**
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
 *
 * See README.md for detailed information.
 */

#include <getopt.h>
#include <stdio.h>
#include <string.h>

#include "dlpack/dlpack.h"
#include "emulator_utils.hpp"
#include "hololink/emulation/hsb_config.hpp"
#include "hololink/emulation/linux_data_plane.hpp"

/**
 * @brief Example program to serve frames from a file to a HSB Emulator as a single sensor
 *
 * This program is used to serve frames from a source file of raw bytes from an HSB emulator.
 *
 * The program will read data from a source file, breaking it up into frames of size 'frame_size'
 * and send the entirety of the file 'num_frames' times at a rate of 'frame_rate_per_second'
 * frames per second using raw packets over Linux sockets.
 *
 * Frames will not start being sent until a connection is established with a receiver (a target
 * address and port is detected).
 *
 * default values:
 * - frame_rate_per_second: 60
 * - frame_limit: 0 (infinite)
 * - frame_size: 0 (use the entire size of the source file)
 *
 * for details of other parameters, see the help message below or running
 *
 * /path/to/serve_linux_file --help
 *
 */

#define DEFAULT_FRAME_RATE_PER_SECOND 60
#define DEFAULT_NUM_FRAMES 0
#define DEFAULT_FRAME_SIZE 0

using namespace hololink::emulation;

void print_usage(char const* program_name)
{
    printf("Usage: %s <hololink_ip_address> <filename> [--frame-rate <frame_rate>]\n"
           "    serves raw data in a file 'filename' as if from a single HSB sensor from ip address 'hololink_ip_address'\n\n"
           "    hololink_ip_address: IP address of the HSB Emulator device. "
           "        Note: for roce receivers, this should be on the same subnet "
           "        as the receiver and physically connected\n"
           "    filename: path to the file to serve\n"
           "    -r, --frame-rate FRAME_RATE: FRAME_RATE in frames per second at which to serve the file (default: %d)\n"
           "    -l, --frame-limit NUM_FRAMES: number of frames to serve the file for (default: %d - infinite)\n"
           "    -s, --frame-size FRAME_SIZE: size of each frame to serve (default: %d - use the size of the source file)\n"
           "    -g, --gpu: serve the data from the GPU.\n"
           "    Example: %s 192.168.0.221 imx274_single_frame.txt --frame-rate 30 --frame-limit 10\n",
        program_name, DEFAULT_FRAME_RATE_PER_SECOND, DEFAULT_NUM_FRAMES, DEFAULT_FRAME_SIZE, program_name);
}

void parse_args(int argc, char** argv, struct LoopConfig& loop_config, int64_t& frame_size, bool& gpu)
{
    int c;
    while (1) {
        static struct option long_options[] = {
            { "help", no_argument, NULL, 'h' },
            { "frame-rate", required_argument, NULL, 'r' },
            { "frame-size", required_argument, NULL, 's' },
            { "frame-limit", required_argument, NULL, 'l' },
            { "gpu", no_argument, NULL, 'g' },
            { 0, 0, 0, 0 }
        };
        int option_index = 0;
        c = getopt_long(argc, argv, "hgr:s:l:", long_options, &option_index);
        if (c == -1)
            break;
        switch (c) {
        case 'r': {
            long frame_rate_per_second = atol(optarg);
            if (frame_rate_per_second <= 0 || frame_rate_per_second > UINT_MAX) {
                throw std::invalid_argument("frame-rate must be a positive integer");
            }
            loop_config.frame_rate_per_second = static_cast<unsigned int>(frame_rate_per_second);
            break;
        }
        case 'l': {
            int frame_limit = atoi(optarg);
            if (frame_limit < 0) {
                throw std::invalid_argument("frame-limit must be a non-negative integer");
            }
            loop_config.num_frames = static_cast<unsigned int>(frame_limit);
            break;
        }
        case 'h':
            print_usage(argv[0]);
            exit(EXIT_SUCCESS);
        case 's':
            frame_size = atoll(optarg);
            if (frame_size < 0) {
                throw std::invalid_argument("frame-size must be a non-negative integer");
            }
            break;
        case 'g':
            gpu = true;
            break;
        default:
            fprintf(stderr, "Unknown option: %d\n", c);
            exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char** argv)
{
    char const* program_name = argv[0];
    if (argc < 3) {
        print_usage(program_name);
        exit(EXIT_FAILURE);
    }

    LoopConfig loop_config = {
        .num_frames = DEFAULT_NUM_FRAMES,
        .frame_rate_per_second = DEFAULT_FRAME_RATE_PER_SECOND
    };
    int64_t shape[1] = { DEFAULT_FRAME_SIZE };
    bool gpu = false;
    parse_args(argc, argv, loop_config, shape[0], gpu);
    if (optind >= argc) {
        fprintf(stderr, "ip address not provided\n");
        print_usage(program_name);
        exit(EXIT_FAILURE);
    }
    std::string ip_address = argv[optind];
    if (optind + 1 >= argc) {
        fprintf(stderr, "filename not provided\n");
        print_usage(program_name);
        exit(EXIT_FAILURE);
    }
    std::string filename = argv[optind + 1];

    // allocate a tensor on the stack
    DLTensor tensor = {
        .device = {
            .device_type = gpu ? DLDeviceType::kDLCUDA : DLDeviceType::kDLCPU,
            .device_id = 0 },
        .ndim = 1,
        .dtype = DLDataType { .code = DLDataTypeCode::kDLUInt, .bits = 8, .lanes = 1 },
        .shape = &shape[0]
    };

    // build the emulator and data plane(s)
    HSBEmulator hsb;
    uint8_t data_plane_id = 0;
    uint8_t sensor_id = 0;
    LinuxDataPlane linux_data_plane(
        hsb,
        IPAddress_from_string(ip_address),
        data_plane_id,
        sensor_id);

    // start the emulator
    hsb.start();

    // run your loop/thread/operator
    printf("Running HSB Emulator... Press Ctrl+C to stop\n");
    loop_single_from_file(loop_config, filename, linux_data_plane, tensor);

    // stop the emulator (if cleanup is needed)
    hsb.stop();

    return 0;
}
