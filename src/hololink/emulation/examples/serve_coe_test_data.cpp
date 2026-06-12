/**
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

#include <getopt.h>
#include <stdexcept>
#include <stdio.h>
#include <string.h>

#include "dlpack/dlpack.h"
#include "emulator_utils.hpp"
#include "hololink/emulation/coe_data_plane.hpp"
#include "hololink/emulation/hsb_config.hpp"
#include "hololink/emulation/sensors/test_sensor.hpp"

/**
 * @brief Example program to serve test-pattern frames over CoE using TestSensor.
 *
 * The host configures the emulator over I2C (frame size, pattern, frame rate, streaming).
 * When streaming is enabled, this program generates frames and sends them through COEDataPlane.
 *
 * Default values:
 * - frame-limit: 0 (infinite)
 * - gpu: false (serve data from CPU)
 */

using namespace hololink::emulation;

#define DEFAULT_NUM_FRAMES 0

void print_usage(char const* program_name)
{
    printf("Usage: %s [options] <hololink_ip_address>\n"
           "    serves test-pattern frames over CoE from ip address 'hololink_ip_address'\n\n"
           "    hololink_ip_address: IP address of the HSB Emulator device.\n"
           "    -l, --frame-limit NUM_FRAMES: number of frames to serve (default: %d - infinite)\n"
           "    -g, --gpu: serve the data from the GPU.\n"
           "    -h, --help: print this message.\n"
           "    Example: %s --frame-limit 100 192.168.0.12\n",
        program_name, DEFAULT_NUM_FRAMES, program_name);
}

void parse_args(int argc, char** argv, struct LoopConfig& loop_config, bool& gpu)
{
    int c;
    while (1) {
        static struct option long_options[] = {
            { "help", no_argument, NULL, 'h' },
            { "frame-limit", required_argument, NULL, 'l' },
            { "gpu", no_argument, NULL, 'g' },
            { 0, 0, 0, 0 }
        };
        int option_index = 0;
        c = getopt_long(argc, argv, "hgl:", long_options, &option_index);
        if (c == -1)
            break;
        switch (c) {
        case 'h':
            print_usage(argv[0]);
            exit(EXIT_SUCCESS);
        case 'g':
            gpu = true;
            break;
        case 'l': {
            long frame_limit = atol(optarg);
            if (frame_limit < 0) {
                throw std::invalid_argument("frame-limit must be a non-negative integer");
            }
            loop_config.num_frames = static_cast<unsigned int>(frame_limit);
            break;
        }
        default:
            fprintf(stderr, "Unknown option: %d\n", c);
            exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char** argv)
{
    char const* program_name = argv[0];
    if (argc < 2) {
        print_usage(program_name);
        return -1;
    }

    struct LoopConfig loop_config = {
        .num_frames = DEFAULT_NUM_FRAMES,
        .frame_rate_per_second = 1, // overridden each iteration from TestSensor I2C config
    };
    bool gpu = false;
    parse_args(argc, argv, loop_config, gpu);
    if (optind >= argc) {
        fprintf(stderr, "ip address not provided\n");
        print_usage(program_name);
        exit(EXIT_FAILURE);
    }
    std::string ip_address = argv[optind];

    int64_t shape[1] = { 0 };
    DLTensor tensor = {
        .device = {
            .device_type = gpu ? DLDeviceType::kDLCUDA : DLDeviceType::kDLCPU,
            .device_id = 0 },
        .ndim = 1,
        .dtype = DLDataType { .code = DLDataTypeCode::kDLUInt, .bits = 8, .lanes = 1 },
        .shape = &shape[0]
    };

    HSBEmulator hsb;
    const uint8_t data_plane_id = 0;
    const uint8_t sensor_id = 0;
    COEDataPlane coe_data_plane(
        hsb,
        IPAddress_from_string(ip_address),
        data_plane_id,
        sensor_id);

    sensors::TestSensor test_sensor;
    test_sensor.attach_to_i2c(hsb.get_i2c(hololink::I2C_CTRL), hololink::CAM_I2C_BUS + sensor_id);

    hsb.start();

    printf("Running HSB Emulator with TestSensor... Press Ctrl+C to stop\n");
    loop_single_test(loop_config, test_sensor, coe_data_plane, tensor);

    hsb.stop();

    return 0;
}
