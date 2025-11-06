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
#include "hololink/emulation/coe_data_plane.hpp"
#include "hololink/emulation/hsb_config.hpp"
#include "hololink/emulation/sensors/vb1940_emulator.hpp"

/**
 * @brief Example program to serve frames from a file to a HSB Emulator as if
 * from a single vb1940 sensor (e.g. Leopard Eagle HSB)
 *
 * The program emulates both the HSB and vb1940 sensors and uses their interfaces to
 *   1) generate the frame based on configuration from the host size (frame shape/size)
 *      data can either be in CPU or GPU memory
 *   2) when the vb1940 are set by the host to streaming mode, the program begins to send
 *      the data through the COEDataPlane (Camera-over-Ethernet protocol transmission)
 *   3) The rate at which it will send data defaults to Leopard Eagle values but can be tuned
 *      by program parameters (frame rate). Note that ultimately the tuning of the Emulator
 *      Device's network and CPU stack will determine whether it can reach the target frame rate
 *   4) The program will run until the number of frames specified by the user is reached
 *      (infinite for 0, the default) or the user presses Ctrl+C
 *
 * Default values
 * - frame-rate: 30 (per second)
 * - frame-limit: 0 (infinite)
 * - gpu: false (serve data from the CPU)
 *
 */

using namespace hololink::emulation;

#define DEFAULT_FRAME_RATE_PER_SECOND 30
#define DEFAULT_NUM_FRAMES 0

void print_usage(char const* program_name)
{
    printf("Usage: %s [options] <hololink_ip_address>\n"
           "    serves csi-2 data as if from a single vb1940 sensor from ip address 'hololink_ip_address'\n\n"
           "    hololink_ip_address: IP address of the HSB Emulator device. "
           "        Note: for roce receivers, this should be on the same subnet "
           "        as the receiver and physically connected\n"
           "    -r, --frame-rate FRAME_RATE: FRAME_RATE in frames per second at which to serve the file (default: %d)\n"
           "    -l, --frame-limit NUM_FRAMES: number of frames to serve (default: %d - infinite)\n"
           "    -g, --gpu: serve the data from the GPU.\n"
           "    Example: %s --frame-rate 60 --num-frames 10 192.168.0.12\n",
        program_name, DEFAULT_FRAME_RATE_PER_SECOND, DEFAULT_NUM_FRAMES, program_name);
}

void parse_args(int argc, char** argv, struct LoopConfig& loop_config, bool& gpu)
{
    int c;
    while (1) {
        static struct option long_options[] = {
            { "frame-rate", required_argument, NULL, 'r' },
            { "help", no_argument, NULL, 'h' },
            { "frame-limit", required_argument, NULL, 'l' },
            { "gpu", no_argument, NULL, 'g' },
            { 0, 0, 0, 0 }
        };
        int option_index = 0;
        c = getopt_long(argc, argv, "r:hgl:", long_options, &option_index);
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
        case 'h':
            print_usage(argv[0]);
            exit(EXIT_SUCCESS);
        case 'g':
            gpu = true;
            break;
        case 'l': {
            long frame_limit = atol(optarg);
            if (frame_limit < 0) {
                throw std::invalid_argument("num-frames must be a non-negative integer");
            }
            loop_config.num_frames = static_cast<unsigned int>(frame_limit);
            break;
        }
        default:
            fprintf(stderr, "Unknown option: %d\n", c);
            exit(EXIT_FAILURE);
            break;
        }
    }
}

int main(int argc, char** argv)
{
    char const* program_name = argv[0];
    if (argc < 2) {
        print_usage(program_name);
        return 0;
    }
    struct LoopConfig loop_config = {
        .num_frames = DEFAULT_NUM_FRAMES,
        .frame_rate_per_second = DEFAULT_FRAME_RATE_PER_SECOND
    };
    bool gpu = false;
    parse_args(argc, argv, loop_config, gpu);
    if (optind >= argc) {
        fprintf(stderr, "ip address not provided\n");
        print_usage(program_name);
        exit(EXIT_FAILURE);
    }
    std::string ip_address = argv[optind];

    // allocate space for the frame data
    int64_t shape[1] = { 0 };
    DLTensor tensor = {
        .device = {
            .device_type = gpu ? DLDeviceType::kDLCUDA : DLDeviceType::kDLCPU,
            .device_id = 0 },
        .ndim = 1,
        .dtype = DLDataType { .code = DLDataTypeCode::kDLUInt, .bits = 8, .lanes = 1 },
        .shape = &shape[0]
    };

    // build the emulator and data plane(s)
    // NOTE: this is the configuration of the Leopard Eagle board,
    // which is currently the only one that will work with vb1940
    HSBEmulator hsb(HSB_LEOPARD_EAGLE_CONFIG);
    const uint8_t data_plane_id = 0;
    const uint8_t sensor_id = 0;
    COEDataPlane coe_data_plane(
        hsb,
        IPAddress_from_string(ip_address),
        data_plane_id,
        sensor_id);

    sensors::Vb1940Emulator vb1940;
    // On Leopard Eagle, the i2c bus address is the sensor_id offset from CAM_I2C_BUS
    vb1940.attach_to_i2c(hsb.get_i2c(hololink::I2C_CTRL), hololink::CAM_I2C_BUS + sensor_id);

    // start the emulator
    hsb.start();

    // run your loop/thread/operator
    printf("Running HSB Emulator... Press Ctrl+C to stop\n");
    loop_single_vb1940(loop_config, vb1940, coe_data_plane, tensor);

    // stop the emulator (if cleanup is needed)
    hsb.stop();

    return 0;
}