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

#include <chrono>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>
#include <errno.h>
#include <memory>
#include <string>
#include <string.h>
#include <thread>
#include <time.h>
 
#include "emulator_utils.hpp"
 
using namespace hololink::emulation;

#include "emulator_kernels.cu"

// returns 0 on success, -1 on failure, required bytes in dest if dest == nullptr
// n_bytes is the number of bytes in the source data. 
// currently assumes src contains 8-bit pixels. adjust type and factor as needed.
// convert 8-bit bayer pattern to 10-bit bayer pattern with T_X2Rc10Rb10Ra10 encoding (3 pixels per 4 bytes)
uint32_t bayer8p_to_T_X2Rc10Rb10Ra10(uint8_t * dest, uint8_t * src, uint16_t pixel_height, uint16_t pixel_width) {
    // every 3 pixels gets packed into 4 bytes
    uint16_t line_bytes = (pixel_width + 2) / 3 * 4;
    // line_bytes gets padded to 64 byte boundary
    line_bytes = ((line_bytes + 63) >> 6) << 6;
    if (!dest) {
        return pixel_height * line_bytes;
    }

    auto threads_per_block = dim3(32, 32);
    auto blocks_per_grid = dim3((pixel_width + 3 * threads_per_block.x - 1) / (threads_per_block.x * 3), (pixel_height + threads_per_block.y - 1) / threads_per_block.y);
    bayer8p_to_T_X2Rc10Rb10Ra10_kernel<<<blocks_per_grid, threads_per_block>>>(dest, line_bytes, src, pixel_height, pixel_width);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("bayer8p_to_T_X2Rc10Rb10Ra10_kernel failed: " + std::string(cudaGetErrorString(error)));
    }
    return 0;
}

// convert 8-bit bayer pattern to 10-bit packed bayer pattern (4 pixels per 5 bytes)
uint32_t bayer8p_to_10p(uint8_t * dest, uint8_t * src, uint16_t pixel_height, uint16_t pixel_width) {
    // every 4 pixels gets packed into 5 bytes
    uint16_t line_bytes = (pixel_width) / 4 * 5;
    // line_bytes gets padded to 8 byte boundary
    line_bytes = ((line_bytes + 7) >> 3) << 3;
    if (!dest) {
        return pixel_height * line_bytes;
    }

    auto threads_per_block = dim3(32, 32);
    auto blocks_per_grid = dim3((pixel_width + 4 * threads_per_block.x - 1) / threads_per_block.x / 4, (pixel_height + threads_per_block.y - 1) / threads_per_block.y);
    bayer8p_to_10p_kernel<<<blocks_per_grid, threads_per_block>>>(dest, line_bytes, src, pixel_height, pixel_width);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("bayer8p_to_10p_kernel failed: " + std::string(cudaGetErrorString(error)));
    }
    cudaDeviceSynchronize();
    return 0;
}

// returns 0 on success, -1 on failure, required bytes in dest if dest == nullptr
// n_bytes is the number of bytes in the source data. It must be 64-byte aligned.
// currently assumes src contains 8-bit pixels. adjust type and factor as needed.
// convert 8-bit bayer pattern to 12-bit bayer pattern with T_R12_PK_ISP encoding (2 pixels per 3 bytes)
uint32_t bayer8p_to_T_R12_PK_ISP(uint8_t * dest, uint8_t * src, uint16_t pixel_height, uint16_t pixel_width) {
    // every 2 pixels gets packed into 3 bytes
    uint16_t line_bytes = (pixel_width + 1) / 2 * 3;
    // line_bytes gets padded to 64 byte boundary
    line_bytes = ((line_bytes + 63) >> 6) << 6;
    if (!dest) {
        return pixel_height * line_bytes;
    }

    auto threads_per_block = dim3(32, 32);
    auto blocks_per_grid = dim3((pixel_width + 2 * threads_per_block.x - 1) / threads_per_block.x / 2, (pixel_height + threads_per_block.y - 1) / threads_per_block.y);
    bayer8p_to_T_R12_PK_ISP_kernel<<<blocks_per_grid, threads_per_block>>>(dest, line_bytes, src, pixel_height, pixel_width);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("bayer8p_to_T_R12_PK_ISP_kernel failed: " + std::string(cudaGetErrorString(error)));
    }
    return 0;
}

// generate 8-bit bayer pattern (GBRG)
void generate_bayerGB8p(uint8_t * data, uint16_t pixel_height, uint16_t pixel_width)
{
    auto threads_per_block = dim3(32, 32);
    auto blocks_per_grid = dim3((pixel_width + threads_per_block.x - 1) / threads_per_block.x, (pixel_height + threads_per_block.y - 1) / threads_per_block.y);
    generate_bayerGB8p_kernel<<<blocks_per_grid, threads_per_block>>>(data, pixel_height, pixel_width);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error("generate_bayerGB8p_kernel failed: " + std::string(cudaGetErrorString(error)));
    }
    cudaDeviceSynchronize();
}

// cuda free deleter
void cudaFreeDeleter(void* data)
{
    cudaFree(data);
}

// Load the data from the file into a single block of memory
std::shared_ptr<uint8_t> load_data(const std::string& filename, int64_t& n_bytes, bool gpu)
{
    FILE* file = fopen(filename.c_str(), "rb");
    if (file == nullptr) {
        fprintf(stderr, "Failed to open file: %s, error: %d - %s", filename.c_str(), errno, strerror(errno));
        return nullptr;
    }
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    if (file_size < 0) {
        fprintf(stderr, "Failed to get file size: %s, error: %d - %s", filename.c_str(), errno, strerror(errno));
        return nullptr;
    }
    fseek(file, 0, SEEK_SET);
    uint8_t* data = new uint8_t[file_size];
    long bytes_read = fread(data, 1, file_size, file);
    if (bytes_read != file_size) {
        fprintf(stderr, "Failed to read full file: %s - expected %ld bytes, read %ld bytes, error: %d - %s", filename.c_str(), file_size, bytes_read, errno, strerror(errno));
        delete[] data;
        return nullptr;
    }
    fclose(file);
    n_bytes = file_size;
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
void sleep_frame_rate(struct timespec* frame_start_time, struct LoopConfig& loop_config)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    int64_t delta_sec = now.tv_sec - frame_start_time->tv_sec;
    int64_t delta_nsec = now.tv_nsec - frame_start_time->tv_nsec;
    if (delta_nsec < 0) {
        delta_sec--;
        delta_nsec += NS_PER_SEC;
    }
    int delta_ms = static_cast<int>(MS_PER_SEC / loop_config.frame_rate_per_second - MS_PER_SEC * (delta_sec + delta_nsec * SEC_PER_NS));
    if (delta_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(delta_ms));
    }
}

// initialize and loop through frames from the file `filename`, loading data into the `tensor`
void loop_single_from_file(struct LoopConfig& loop_config, std::string& filename, DataPlane& channel, DLTensor& tensor)
{
    int64_t n_bytes = 0;
    int64_t& frame_size = tensor.shape[0];
    std::shared_ptr<uint8_t> data = load_data(filename, n_bytes, tensor.device.device_type == DLDeviceType::kDLCUDA);
    if (!data) {
        fprintf(stderr, "Failed to load data from file: %s\n", filename.c_str());
        exit(EXIT_FAILURE);
    }

    if (!frame_size) {
        frame_size = n_bytes;
    }

    if (frame_size > n_bytes) {
        fprintf(stderr, "frame size is larger than the file size: %" PRId64 " > %" PRId64 "\n", frame_size, n_bytes);
        exit(EXIT_FAILURE);
    }

    unsigned int frame_count = 0;
    int64_t data_offset = 0;
    while ((!loop_config.num_frames || (frame_count < loop_config.num_frames))) {
        struct timespec frame_start_time;
        clock_gettime(CLOCK_MONOTONIC, &frame_start_time);
        tensor.data = data.get() + data_offset;
        int64_t sent_bytes = channel.send(tensor);
        if (sent_bytes < 0) {
            fprintf(stderr, "Error sending data: %" PRId64 "\n", sent_bytes);
        }
        if (sent_bytes > 0) {
            frame_count++;
            data_offset += frame_size;
            if (data_offset > n_bytes - frame_size) {
                data_offset = 0;
                ;
            }
        }
        sleep_frame_rate(&frame_start_time, loop_config);
    }
}

// combined loop function that loads data from file but waits for VB1940 streaming control
void loop_single_vb1940_from_file(struct LoopConfig& loop_config, std::string& filename, 
    sensors::Vb1940Emulator& vb1940, DataPlane& data_plane, DLTensor& tensor)
{
    // Load data from file (similar to loop_single_from_file)
    int64_t n_bytes = 0;
    int64_t& frame_size = tensor.shape[0];
    std::shared_ptr<uint8_t> data = load_data(filename, n_bytes, tensor.device.device_type == DLDeviceType::kDLCUDA);
    if (!data) {
        fprintf(stderr, "Failed to load data from file: %s\n", filename.c_str());
        exit(EXIT_FAILURE);
    }

    if (!frame_size) {
        frame_size = n_bytes;
    }
    if (frame_size > n_bytes) {
        fprintf(stderr, "frame size is larger than the file size: %" PRId64 " > %" PRId64 "\n", frame_size, n_bytes);
        exit(EXIT_FAILURE);
    }

    // Loop with VB1940 streaming control (similar to loop_single_vb1940)
    unsigned int frame_count = 0;
    int64_t data_offset = 0;

    while (!loop_config.num_frames || (frame_count < loop_config.num_frames)) {
        struct timespec frame_start_time;
        clock_gettime(CLOCK_MONOTONIC, &frame_start_time);

        // Send data only when streaming
        if (vb1940.is_streaming()) {
            // Set tensor data pointer to current offset in loaded data
            tensor.data = data.get() + data_offset;

            int64_t sent_bytes = data_plane.send(tensor);
            if (sent_bytes < 0) {
                fprintf(stderr, "Error sending data: %" PRId64 "\n", sent_bytes);
            }
            if (sent_bytes > 0) {
                frame_count++;
                data_offset += frame_size;
                // Wrap around when we reach the end of the file
                if (data_offset > n_bytes - frame_size) {
                    data_offset = 0;
                }
            }
        }

        sleep_frame_rate(&frame_start_time, loop_config);
    }
}

// generate a vb1940 csi-2 format frame with optional packetization (query data_plane.packetizer_enabled())
void generate_vb1940_frame(DLTensor &frame_data, sensors::Vb1940Emulator& vb1940, bool packetize)
{
    const bool is_gpu = frame_data.device.device_type == kDLCUDA;
    // clean up any previous frame data
    if (frame_data.data) {
        if (is_gpu) {
            cudaFree(frame_data.data);
        } else {
            delete[] static_cast<uint8_t*>(frame_data.data);
        }
        frame_data.data = nullptr;
    }

    uint32_t (*frame_generator)(uint8_t * frame, uint8_t * gbrg, uint16_t pixel_height, uint16_t pixel_width) = nullptr;
    if (vb1940.get_pixel_bits() == 10) {
        if (packetize) {
            frame_generator = bayer8p_to_T_X2Rc10Rb10Ra10;
        } else {
            frame_generator = bayer8p_to_10p;
        }   
    } // for 8-bit, do nothing, for imx274, there will be a raw12 and associated packetizer

    uint16_t pixel_height = vb1940.get_pixel_height();
    uint16_t pixel_width = vb1940.get_pixel_width();
    uint16_t start_byte = vb1940.get_image_start_byte();
    uint16_t line_bytes = vb1940.get_bytes_per_line();
    uint32_t image_size = pixel_height * line_bytes;
    if (packetize) { // if packetizing, the frame generator will return the modified image size
        image_size = frame_generator(nullptr, nullptr, pixel_height, pixel_width);
        line_bytes = image_size / pixel_height;
    }
    image_size += line_bytes * 3; // 1 line of leading data, 2 lines of trailing data

    cudaError_t error = cudaMalloc(&frame_data.data, image_size);
    if (error != cudaSuccess) {
        frame_data.data = nullptr;
        throw std::runtime_error("CUDA allocation failed for frame data: " + std::string(cudaGetErrorString(error)));
    }
    cudaMemset(frame_data.data, 0, image_size);

    if (frame_generator) {
        uint8_t * gbrg = nullptr;
        error = cudaMalloc(&gbrg, pixel_height * pixel_width);
        if (error != cudaSuccess) {
            frame_data.data = nullptr;
            throw std::runtime_error("CUDA allocation failed for gbrg: " + std::string(cudaGetErrorString(error)));
        }
        generate_bayerGB8p(gbrg, pixel_height, pixel_width);
        frame_generator(((uint8_t*)frame_data.data) + start_byte, gbrg, pixel_height, pixel_width);
        cudaFree(gbrg);
        frame_data.shape[0] = image_size;
    } else { // generate_bayerGB8p is enough for raw8
        generate_bayerGB8p(((uint8_t*)frame_data.data) + start_byte, pixel_height, pixel_width);
        frame_data.shape[0] = image_size;
    } // add raw12 option for imx274

    if (!is_gpu) { // if example wants to exercise sending host memory
        uint8_t * host_data = new uint8_t[image_size];
        error = cudaMemcpy(host_data, frame_data.data, image_size, cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            throw std::runtime_error("CUDA Memcpy failed: " + std::string(cudaGetErrorString(error)));
        }
        cudaFree(frame_data.data);
        frame_data.data = host_data;
    }
}

// the thread loop for serving a frame (configured on host side) in vb1940 csi-2 format as if from a single vb1940 sensor
void loop_single_vb1940(struct LoopConfig& loop_config, sensors::Vb1940Emulator& vb1940, DataPlane& data_plane, DLTensor& frame_data)
{
    bool streaming = false;
    unsigned int frame_count = 0;
    while (!loop_config.num_frames || (frame_count < loop_config.num_frames)) {
        struct timespec frame_start_time;
        clock_gettime(CLOCK_MONOTONIC, &frame_start_time);
        if (!streaming && vb1940.is_streaming()) {
            streaming = true;
            generate_vb1940_frame(frame_data, vb1940, data_plane.packetizer_enabled());
        } else if (!vb1940.is_streaming()) {
            streaming = false;
        }
        if (streaming) {
            int64_t sent_bytes = data_plane.send(frame_data);
            if (sent_bytes < 0) {
                throw std::runtime_error("Error sending data: " + std::to_string(errno) + " " + std::string(strerror(errno)));
            }
            if (sent_bytes > 0) { // only increment frame_count if data was sent
                frame_count++;
            }
        }
        sleep_frame_rate(&frame_start_time, loop_config);
    }
}

// the thread loop for serving two frames simultaneously (configured on host side) in vb1940 csi-2 format as if from stereo vb1940 sensors
void loop_stereo_vb1940(struct LoopConfig& loop_config, sensors::Vb1940Emulator& vb1940_0, DataPlane& data_plane_0, DLTensor& tensor_0, sensors::Vb1940Emulator& vb1940_1, DataPlane& data_plane_1, DLTensor& tensor_1)
{
    bool streaming = false;
    unsigned int frame_count = 0;
    while (!loop_config.num_frames || (frame_count < loop_config.num_frames)) {
        struct timespec frame_start_time;
        clock_gettime(CLOCK_MONOTONIC, &frame_start_time);
        if (!streaming && (vb1940_0.is_streaming() && vb1940_1.is_streaming())) {
            streaming = true;
            generate_vb1940_frame(tensor_0, vb1940_0, data_plane_0.packetizer_enabled());
            generate_vb1940_frame(tensor_1, vb1940_1, data_plane_1.packetizer_enabled());
        } else if (!vb1940_0.is_streaming() && !vb1940_1.is_streaming()) {
            streaming = false;
        }
        if (streaming) {
            int64_t sent_bytes_0 = 0, sent_bytes_1 = 0;
            if (vb1940_0.is_streaming()) {
                sent_bytes_0 = data_plane_0.send(tensor_0);
                if (sent_bytes_0 < 0) {
                    throw std::runtime_error("Error sending data on channel 0 {" + std::to_string(sent_bytes_0) + "/" + std::to_string(tensor_0.shape[0]) + "}: " + std::to_string(errno) + " " + std::string(strerror(errno)));
                }
            }
            if (vb1940_1.is_streaming()) {
                sent_bytes_1 = data_plane_1.send(tensor_1);
                if (sent_bytes_1 < 0) {
                    throw std::runtime_error("Error sending data on channel 1 {" + std::to_string(sent_bytes_1) + "/" + std::to_string(tensor_1.shape[0]) + "}: " + std::to_string(errno) + " " + std::string(strerror(errno)));
                }
            }
            if (sent_bytes_0 + sent_bytes_1 > 0) {
                frame_count++;
            }
        }
        
        sleep_frame_rate(&frame_start_time, loop_config);
    }
}
