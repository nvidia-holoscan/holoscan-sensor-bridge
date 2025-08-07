/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SRC_HOLOLINK_OPERATORS_ARGUS_ISP_ARGUS_IMPL
#define SRC_HOLOLINK_OPERATORS_ARGUS_ISP_ARGUS_IMPL

#include <memory>

#include <hololink/common/cuda_helper.hpp>
#include <holoscan/core/operator.hpp>

#include <Argus/Argus.h>
#undef Success // Error propagated in X.h
#include "Argus/CaptureMetadata.h"
#include "EGLStream/EGLStream.h"
#include "cuda.h"
#include "cudaEGL.h"
#include "cuda_runtime_api.h"

#include "camera_provider.hpp"

namespace hololink::operators {

typedef enum {
    OUTPUT_PIXEL_FORMAT_YUV420 = 0,
    OUTPUT_PIXEL_FORMAT_YUV444,
} OutputPixelFormat;

class ArgusImpl {
public:
    // Member functions
    std::vector<Argus::CameraDevice*> camera_devices_;
    uint32_t camera_index_ = 0;
    std::vector<Argus::SensorMode*> sensor_modes_;

    Argus::ICameraProvider* i_camera_provider_ = nullptr;
    Argus::UniqueObj<Argus::CaptureSession> capture_session_;
    Argus::ISensorMode* i_sensor_mode_ = nullptr;
    Argus::IReprocessInfo* i_reprocess_info_ = nullptr;

    Argus::UniqueObj<Argus::Request> request_;
    Argus::UniqueObj<Argus::OutputStream> out_stream_;
    Argus::UniqueObj<Argus::OutputStreamSettings> out_stream_settings_;

    Argus::UniqueObj<Argus::InputStream> in_stream_;
    Argus::UniqueObj<Argus::InputStreamSettings> in_stream_settings_;

    // Interface for interacting with output EGL streams
    Argus::IEGLOutputStream* i_egl_output_stream_ = nullptr;
    Argus::IEGLInputStream* i_egl_input_stream_ = nullptr;

    // CUDA Interop data
    // for input connection
    CUeglStreamConnection cuda_egl_o_connection_ = nullptr;
    // for output connection
    CUeglStreamConnection cuda_egl_i_connection_ = nullptr;

    // TODO: Parameter from the user
    OutputPixelFormat output_pixel_format_ = OUTPUT_PIXEL_FORMAT_YUV420;

    // Constructor
    ArgusImpl(std::shared_ptr<Argus::CameraProvider> cameraProvider);

    // Helper functions
    void setup_camera_devices(uint32_t cameraIndex);
    void set_sensor_mode_info(uint32_t sensorModeIndex);
    void set_reprocess_info(int bayerFormat, int pixelBitDepth);
    void setup_output_streams(const uint8_t sensorModeIndex);
    void setup_input_streams();
    void setup_capture_request(float analog_gain, float exposureTimeMs);
    void stop();
};
} // namespace hololink::operators

#endif // SRC_HOLOLINK_OPERATORS_ARGUS_ISP_ARGUS_IMPL
