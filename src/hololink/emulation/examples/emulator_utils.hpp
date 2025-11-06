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

#ifndef EXAMPLES_EMULATOR_UTILS_HPP
#define EXAMPLES_EMULATOR_UTILS_HPP

#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <memory>
#include <string>

#include "dlpack/dlpack.h"
#include "hololink/emulation/data_plane.hpp"
#include "hololink/emulation/hsb_config.hpp"
#include "hololink/emulation/sensors/vb1940_emulator.hpp"

#define MS_PER_SEC (1000.0)
#define US_PER_SEC (1000.0 * MS_PER_SEC)
#define NS_PER_SEC (1000.0 * US_PER_SEC)
#define SEC_PER_NS (1.0 / NS_PER_SEC)

/////// loop configuration and operation elements //////
struct LoopConfig {
    unsigned int num_frames;
    unsigned int frame_rate_per_second;
};

// sleep the thread to try to match target frame rate
void sleep_frame_rate(struct timespec* frame_start_time, struct LoopConfig& loop_config);

/////// data generation elements //////
// Load the data from the file into a single block of memory
std::shared_ptr<uint8_t> load_data(const std::string& filename, int64_t& n_bytes, bool gpu = false);

// generate a vb1940 csi-2 format frame.
// to generate data on GPU, set frame_data.device.device_type = kDLCUDA. 0 or other values will generate data on host device using new[].
// cleanup of the buffer in frame_data is left to the caller. Note that the examples leak this memory
// if frame_data.device.device_type == kDLCUDA, use cudaFree(frame_data.data), otherwise use delete[] static_cast<uint8_t*>(frame_data.data)
void generate_vb1940_frame(DLTensor& frame_data, hololink::emulation::sensors::Vb1940Emulator& vb1940, bool packetize = false);

/////// loop execution methods //////
// initialize and loop through frames from the file `filename`, loading data into the `tensor`
void loop_single_from_file(struct LoopConfig& loop_config, std::string& filename, hololink::emulation::DataPlane& channel, DLTensor& tensor);

// Combined loop function that loads data from file but waits for VB1940 streaming control
void loop_single_vb1940_from_file(struct LoopConfig& loop_config, std::string& filename,
    hololink::emulation::sensors::Vb1940Emulator& vb1940, hololink::emulation::DataPlane& data_plane, DLTensor& tensor);

// the thread loop for serving a frame (configured on host side) in vb1940 csi-2 format as if from a single vb1940 sensor
void loop_single_vb1940(struct LoopConfig& loop_config, hololink::emulation::sensors::Vb1940Emulator& vb1940, hololink::emulation::DataPlane& data_plane, DLTensor& frame_data);

// the thread loop for serving two frames simultaneously (configured on host side) in vb1940 csi-2 format as if from stereo vb1940 sensors
void loop_stereo_vb1940(struct LoopConfig& loop_config, hololink::emulation::sensors::Vb1940Emulator& vb1940_0, hololink::emulation::DataPlane& data_plane_0, DLTensor& tensor_0, hololink::emulation::sensors::Vb1940Emulator& vb1940_1, hololink::emulation::DataPlane& data_plane_1, DLTensor& tensor_1);

#endif // EXAMPLES_EMULATOR_UTILS_HPP
