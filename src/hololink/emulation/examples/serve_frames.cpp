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

#include <cassert>
#include <chrono>
#include <csignal>
#include <cuda_runtime.h>
#include <errno.h>
#include <getopt.h>
#include <stdio.h>
#include <string.h>
#include <thread>

#include "hololink/emulation/dlpack/dlpack.h"
#include "hololink/emulation/linux_data_plane.hpp"

/**
 * @brief Example program to serve frames from a file to a HSB Emulator as a single sensor
 *
 * This program is used to serve frames from a source file of raw bytes from an HSB emulator.
 *
 * The program will read data from a source file, breaking it up into frames of size 'frame_size'
 * and send the entirety of the file 'num_cycles' times at a rate of 'frame_rate_per_second'
 * frames per second.
 *
 * Frames will not start being sent until a connection is established with a receiver (a target
 * address and port is detected).
 *
 * default values:
 * - frame_rate_per_second: 60
 * - num_cycles: 0 (infinite)
 * - frame_size: 0 (use the entire size of the source file)
 *
 * for details of other parameters, see the help message below or running
 *
 * /path/to/serve_frames --help
 *
 */

#define DEFAULT_FRAME_RATE_PER_SECOND 60
#define DEFAULT_SOURCE_PORT 12288
#define DEFAULT_SUBNET_BITS 24
#define DEFAULT_NUM_CYCLES 0
#define DEFAULT_FRAME_SIZE 0

#define MS_PER_SEC (1000.0)
#define US_PER_SEC (1000.0 * MS_PER_SEC)
#define NS_PER_SEC (1000.0 * US_PER_SEC)
#define SEC_PER_NS (1.0 / NS_PER_SEC)

int frame_rate_per_second = DEFAULT_FRAME_RATE_PER_SECOND;
long frame_size = DEFAULT_FRAME_SIZE;
unsigned int num_cycles = DEFAULT_NUM_CYCLES;

void cudaFreeDeleter(void* data)
{
    cudaFree(data);
}

// Load the data from the file into a single block of memory
std::shared_ptr<uint8_t> load_data(char const* filename, int64_t* n_bytes, bool gpu = false)
{
    FILE* file = fopen(filename, "rb");
    if (file == nullptr) {
        fprintf(stderr, "Failed to open file: %s, error: %d - %s", filename, errno, strerror(errno));
        return nullptr;
    }
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    if (file_size < 0) {
        fprintf(stderr, "Failed to get file size: %s, error: %d - %s", filename, errno, strerror(errno));
        return nullptr;
    }
    fseek(file, 0, SEEK_SET);
    uint8_t* data = new uint8_t[file_size];
    long bytes_read = fread(data, 1, file_size, file);
    if (bytes_read != file_size) {
        fprintf(stderr, "Failed to read full file: %s - expected %ld bytes, read %ld bytes, error: %d - %s", filename, file_size, bytes_read, errno, strerror(errno));
        return nullptr;
    }
    fclose(file);
    if (!frame_size) {
        frame_size = file_size;
    }
    *n_bytes = file_size;
    std::shared_ptr<uint8_t> data_ptr = nullptr;
    if (gpu) {
        uint8_t* gpu_data = nullptr;
        if (cudaMalloc(&gpu_data, file_size) == cudaSuccess) {
            if (cudaMemcpy(gpu_data, data, file_size, cudaMemcpyHostToDevice) == cudaSuccess) {
                data_ptr = std::shared_ptr<uint8_t>(gpu_data, cudaFreeDeleter);
            } else {
                fprintf(stderr, "CUDA Memcpy failed\n");
                cudaFree(gpu_data);
                gpu_data = nullptr;
            }
        } else {
            fprintf(stderr, "CUDA allocation failed\n");
            gpu_data = nullptr;
        }
        delete[] data;
    } else {
        data_ptr = std::shared_ptr<uint8_t>(data, std::default_delete<uint8_t[]>());
    }
    return data_ptr;
}

// sleep the thread to try to match target frame rate
void sleep_frame_rate(struct timespec* frame_start_time)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    int64_t delta_sec = now.tv_sec - frame_start_time->tv_sec;
    int64_t delta_nsec = now.tv_nsec - frame_start_time->tv_nsec;
    if (delta_nsec < 0) {
        delta_sec--;
        delta_nsec += NS_PER_SEC;
    }
    int delta_ms = static_cast<int>(MS_PER_SEC / frame_rate_per_second - MS_PER_SEC * (delta_sec + delta_nsec * SEC_PER_NS));
    if (delta_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(delta_ms));
    }
}

// initialize and run data serving loop
void loop(hololink::emulation::DataPlane* channel, char const* filename, bool gpu = false)
{
    int64_t n_bytes = 0;
    std::shared_ptr<uint8_t> data = load_data(filename, &n_bytes, gpu);
    if (!data) {
        fprintf(stderr, "Failed to load data from file: %s\n", filename);
        return;
    }

    int64_t shape[1] = { static_cast<int64_t>(frame_size) };
    DLTensor tensor = {
        .data = data.get(),
        .device = {
            .device_type = gpu ? DLDeviceType::kDLCUDA : DLDeviceType::kDLCPU,
            .device_id = 0 },
        .ndim = 1,
        .dtype = DLDataType { .code = DLDataTypeCode::kDLUInt, .bits = 8, .lanes = 1 },
        .shape = &shape[0]
    };
    assert(frame_size > 0);
    if (frame_size > n_bytes) {
        fprintf(stderr, "frame size is larger than the file size\n");
        return;
    }
    unsigned int num_frames = (unsigned int)(n_bytes / frame_size);
    if (!num_frames) {
        fprintf(stderr, "frame size is larger than the file size\n");
        return;
    }
    unsigned int cycle_count = 0; // unsigned so that it can overflow
    unsigned int frame_count = 0;

    while ((!num_cycles || (cycle_count < num_cycles))) {
        int64_t data_offset = frame_count * frame_size;
        tensor.data = data.get() + data_offset;
        if (data_offset > n_bytes - frame_size) {
            shape[0] = n_bytes - data_offset;
        }
        struct timespec frame_start_time;
        clock_gettime(CLOCK_MONOTONIC, &frame_start_time);
        int64_t sent_bytes = channel->send(tensor);
        if (sent_bytes < 0) {
            fprintf(stderr, "Error sending data: %ld\n", sent_bytes);
        }
        sleep_frame_rate(&frame_start_time);
        if (sent_bytes > 0) { // only increment frame_count if data was sent
            frame_count++;
            if (frame_count >= num_frames) {
                shape[0] = frame_size;
                frame_count = 0;
                cycle_count++;
            }
        }
    }
}

void print_usage(char const* program_name)
{
    printf("Usage: %s <hololink_ip_address> <filename> [--frame-rate <frame_rate>]\n"
           "    serves raw data in a file 'filename' as if from a single HSB sensor from ip address 'hololink_ip_address'\n\n"
           "    hololink_ip_address: IP address of the HSB Emulator device. "
           "        Note: for roce receivers, this should be on the same subnet "
           "        as the receiver and physically connected\n"
           "    filename: path to the file to serve\n"
           "    -b, --subnet-bits SUBNET_BITS: number of bits in subnet mask of the HSB Emulator device. "
           "    -p, --source-port PORT: PORT to serve the file from (default: %d)\n"
           "    -r, --frame-rate FRAME_RATE: FRAME_RATE in frames per second at which to serve the file (default: %d)\n"
           "    -c, --num-cycles NUM_CYCLES: number of cycles to serve the file for (default: %d - infinite)\n"
           "    -s, --frame-size FRAME_SIZE: size of each frame to serve (default: %d - use the size of the source file)\n"
           "    -g, --gpu: serve the data from the GPU.\n"
           "    Example: %s 192.168.0.221 imx274_single_frame.txt --frame-rate 30 --num-cycles 10\n",
        program_name, DEFAULT_SOURCE_PORT, DEFAULT_FRAME_RATE_PER_SECOND, DEFAULT_NUM_CYCLES, DEFAULT_FRAME_SIZE, program_name);
}

int main(int argc, char** argv)
{
    char const* program_name = argv[0];
    if (argc < 3) {
        print_usage(program_name);
        return 0;
    }
    std::string ip_address = argv[1];
    unsigned char subnet_bits = DEFAULT_SUBNET_BITS;
    uint16_t source_port = DEFAULT_SOURCE_PORT;
    bool gpu = false;
    char filename[256];
    size_t filename_len = strlen(argv[2]);
    if (filename_len > sizeof(filename) - 1) {
        fprintf(stderr, "filename is too long\n");
        return 1;
    }
    strncpy(filename, argv[2], sizeof(filename));
    int c;
    while (1) {
        static struct option long_options[] = {
            { "subnet-bits", required_argument, NULL, 'b' },
            { "source-port", required_argument, NULL, 'p' },
            { "frame-rate", required_argument, NULL, 'r' },
            { "help", no_argument, NULL, 'h' },
            { "frame-size", required_argument, NULL, 's' },
            { "num-cycles", required_argument, NULL, 'c' },
            { "gpu", no_argument, NULL, 'g' },
            { 0, 0, 0, 0 }
        };
        int option_index = 0;
        c = getopt_long(argc, argv, "b:p:r:hs:c:g", long_options, &option_index);
        if (c == -1)
            break;
        switch (c) {
        case 'b': {
            int subnet_bits_ = atoi(optarg);
            if (subnet_bits_ < 0 || subnet_bits_ > 32) {
                throw std::invalid_argument("subnet-bits must be between 0 and 32");
            }
            subnet_bits = subnet_bits_;
            break;
        }
        case 'p': {
            int source_port_ = atoi(optarg);
            if (source_port_ < 1024 || source_port_ > 65535) {
                throw std::invalid_argument("source-port must be between 1024 and 65535");
            }
            source_port = (uint16_t)source_port_;
            break;
        }
        case 'r':
            frame_rate_per_second = atoi(optarg);
            if (frame_rate_per_second <= 0) {
                throw std::invalid_argument("frame-rate must be a positive integer");
            }
            break;
        case 'h':
            print_usage(program_name);
            break;
        case 'g':
            gpu = true;
            break;
        case 'c': {
            int num_cycles_ = atoi(optarg);
            if (num_cycles_ < 0) {
                throw std::invalid_argument("num-cycles must be a positive integer");
            }
            num_cycles = num_cycles_;
            break;
        }
        case 's':
            frame_size = atol(optarg);
            if (frame_size <= 0) {
                throw std::invalid_argument("frame-size must be a positive integer");
            }
            break;
        default:
            fprintf(stderr, "Unknown option: %d\n", c);
            return 1;
            break;
        }
    }

    using namespace hololink::emulation;

    // build the emulator and data plane(s)
    HSBEmulator hsb {};
    LinuxDataPlane linux_sensor(
        hsb,
        IPAddress_from_string(ip_address, subnet_bits),
        source_port,
        DataPlaneID::DATA_PLANE_0, SensorID::SENSOR_0);

    // start the emulator
    hsb.start();

    // run your loop/thread/operator
    printf("Running HSB Emulator... Press Ctrl+C to stop\n");
    loop(&linux_sensor, filename, gpu);

    // stop the emulator (if cleanup is needed)
    hsb.stop();

    return 0;
}