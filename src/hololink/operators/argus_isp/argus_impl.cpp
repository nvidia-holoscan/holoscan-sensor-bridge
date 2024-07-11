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

#include "argus_impl.hpp"

namespace hololink::operators {

ArgusImpl::ArgusImpl(std::shared_ptr<Argus::CameraProvider> cameraProvider)
{
    // Get the ICameraProvider interface from the global CameraProvider.
    i_camera_provider_ = Argus::interface_cast<Argus::ICameraProvider>(cameraProvider.get());
    if (!i_camera_provider_)
        throw std::runtime_error("Failed to get ICameraProvider interface");
}

void ArgusImpl::setup_camera_devices()
{
    Argus::Status status = Argus::STATUS_OK;
    // Get the camera devices.
    if (!i_camera_provider_) {
        throw std::runtime_error("camera provider is empty\n");
    }
    status = i_camera_provider_->getCameraDevices(&camera_devices_);
    if (status != Argus::STATUS_OK) {
        throw std::runtime_error(fmt::format(
            "Error while listing camera devices: "
            "Unable to get camera devices list from camera provider interface "
            "(Argus Status: {})",
            status));
    }

    if (camera_devices_.size() == 0) {
        HOLOSCAN_LOG_WARN("no camera devices are available");
    } else if (camera_devices_.size() > 0) {
        for (uint32_t i = 0; i < camera_devices_.size(); i++) {
            Argus::ICameraProperties* i_camera_properties = Argus::interface_cast<Argus::ICameraProperties>(camera_devices_[i]);
            if (!i_camera_properties) {
                throw std::runtime_error("Error while listing camera devices: "
                                         "Failed to get ICameraProperties interface");
            }
        }
    }
}

void ArgusImpl::set_sensor_mode_info(uint32_t sensorModeIndex)
{
    Argus::ICameraProperties* i_camera_properties = Argus::interface_cast<Argus::ICameraProperties>(camera_devices_[0]);
    if (!i_camera_properties) {
        throw std::runtime_error("Failed to get ICameraProperties interface\n");
    }

    Argus::Status status = i_camera_properties->getAllSensorModes(&sensor_modes_);
    if (status != Argus::STATUS_OK) {
        throw std::runtime_error("Failed to get the sensor modes from the device.\n");
    }
    if (sensor_modes_.size() == 0) {
        throw std::runtime_error("No sensor modes are available.\n");
    }
    if (sensor_modes_.size() <= sensorModeIndex) {
        throw std::runtime_error(fmt::format("Sensor mode {} was requested but onlt {} {} is available.\n", sensorModeIndex, static_cast<unsigned long>(sensor_modes_.size()), sensor_modes_.size() == 1 ? "is" : "are"));
    }

    i_sensor_mode_ = Argus::interface_cast<Argus::ISensorMode>(sensor_modes_[sensorModeIndex]);
    if (!i_sensor_mode_) {
        throw std::runtime_error("Failed to get sensor mode interface");
    }

    HOLOSCAN_LOG_INFO(fmt::format("Capturing from mono device using sensor mode {} ({}x{})\n",
        sensorModeIndex,
        i_sensor_mode_->getResolution().width(),
        i_sensor_mode_->getResolution().height()));
    return;
}

void ArgusImpl::set_reprocess_info(int bayerFormat, int pixelBitDepth)
{
    i_reprocess_info_ = Argus::interface_cast<Argus::IReprocessInfo>(camera_devices_[0]);
    if (!i_reprocess_info_) {
        throw std::runtime_error("Failed to get the reprocessInfo interface");
    }
    i_reprocess_info_->setReprocessingEnable(true);
    const Argus::Size2D<uint32_t>& resolution = { i_sensor_mode_->getResolution().width(), i_sensor_mode_->getResolution().height() };
    Argus::BayerPhase phase = Argus::BAYER_PHASE_UNKNOWN;
    switch (bayerFormat) {
    case 1: // hololink::operators::ArgusIspOp::BayerFormat::RGGB
        phase = Argus::BAYER_PHASE_RGGB;
        break;
    case 2: // hololink::operators::ArgusIspOp::BayerFormat::GBRG
        phase = Argus::BAYER_PHASE_GBRG;
        break;
    default:
        throw std::runtime_error("Invalid bayer format");
    }
    i_reprocess_info_->setReprocessingModeColorFormat(phase);
    i_reprocess_info_->setReprocessingModeResolution(resolution);
    i_reprocess_info_->setReprocessingModePixelBitDepth(pixelBitDepth);
    i_reprocess_info_->setReprocessingModeDynamicPixelBitDepth(pixelBitDepth);

    // Create capture session before ICaptureSession.
    capture_session_.reset(i_camera_provider_->createCaptureSession(camera_devices_[0]));
}

void ArgusImpl::setup_capture_request(float analogGain, float exporeTimeMs)
{

    // Create CaptureSession
    Argus::ICaptureSession* i_capture_session = Argus::interface_cast<Argus::ICaptureSession>(capture_session_);
    if (!i_capture_session) {
        throw std::runtime_error("Failed to create CaptureSession in setup capture request");
    }

    request_.reset(i_capture_session->createRequest(Argus::CAPTURE_INTENT_PREVIEW));
    Argus::IRequest* i_request = Argus::interface_cast<Argus::IRequest>(request_);
    if (!i_request) {
        throw std::runtime_error("Failed to create capture request");
    }

    Argus::IAutoControlSettings* ac = Argus::interface_cast<Argus::IAutoControlSettings>(i_request->getAutoControlSettings());
    if (!ac) {
        throw std::runtime_error("Failed to get autocontrol settings interface");
    }

    // Set the digital gain range with minimum 1.0f
    if (ac->setIspDigitalGainRange(Argus::Range<float>(1.0f)) != Argus::STATUS_OK) {
        throw std::runtime_error("Unable to set Isp Digital Gain");
    }

    // Configure source settings
    Argus::ISourceSettings* i_source_settings = Argus::interface_cast<Argus::ISourceSettings>(request_);
    if (!i_source_settings) {
        throw std::runtime_error("Failed to get ISourceSettings interface");
    }
    i_source_settings->setSensorMode(sensor_modes_[0]);

    // setExposureTimeRange takes the input in nanoseconds
    if (i_source_settings->setExposureTimeRange(
            Argus::Range<uint64_t>(exporeTimeMs * 1000000))
        != Argus::STATUS_OK) {
        throw std::runtime_error("Unable to set the Source Settings Exposure Time Range");
    }

    // Set analog gain range
    if (i_source_settings->setGainRange(Argus::Range<float>(analogGain)) != Argus::STATUS_OK) {
        throw std::runtime_error("Unable to set the Source Settings Gain Range");
    }

    // Enable the output streams to be used for the capture request
    i_request->enableOutputStream(out_stream_.get());
}

void ArgusImpl::setup_output_streams(const uint8_t sensorModeIndex)
{
    Argus::Status status = Argus::STATUS_OK;
    Argus::ICaptureSession* i_capture_session = Argus::interface_cast<Argus::ICaptureSession>(capture_session_);
    if (!i_capture_session) {
        throw std::runtime_error("Failed to create CaptureSession in output stream");
    }

    // Create output egl stream
    out_stream_settings_.reset(
        i_capture_session->createOutputStreamSettings(Argus::STREAM_TYPE_EGL, &status));
    Argus::IOutputStreamSettings* i_stream_settings = Argus::interface_cast<Argus::IOutputStreamSettings>(out_stream_settings_);
    if (!i_stream_settings) {
        throw std::runtime_error("Failed to get IOutputStreamSettings interface for stream 0");
    }
    i_stream_settings->setCameraDevice(camera_devices_[0]);
    Argus::IEGLOutputStreamSettings* i_egl_stream_settings = Argus::interface_cast<Argus::IEGLOutputStreamSettings>(out_stream_settings_);
    if (!i_egl_stream_settings) {
        throw std::runtime_error("Failed to get IEGLOutputStreamSettings interface for stream 0");
    }
    switch (output_pixel_format_) {
    case OUTPUT_PIXEL_FORMAT_YUV420:
        i_egl_stream_settings->setPixelFormat(Argus::PIXEL_FMT_YCbCr_420_888);
        break;
    default:
        throw std::runtime_error("Output format not supported");
    }
    i_egl_stream_settings->setResolution(i_sensor_mode_->getResolution());
    i_egl_stream_settings->setMode(Argus::EGL_STREAM_MODE_FIFO);
    i_egl_stream_settings->setMetadataEnable(true);

    out_stream_.reset(
        i_capture_session->createOutputStream(out_stream_settings_.get()));
    i_egl_output_stream_ = Argus::interface_cast<Argus::IEGLOutputStream>(out_stream_);
}

void ArgusImpl::setup_input_streams()
{
    Argus::ICaptureSession* i_capture_session = Argus::interface_cast<Argus::ICaptureSession>(capture_session_);
    if (!i_capture_session) {
        throw std::runtime_error("Failed to create CaptureSession in setup capture request");
    }

    in_stream_settings_.reset(i_capture_session->createInputStreamSettings(Argus::STREAM_TYPE_EGL));
    Argus::IEGLInputStreamSettings* i_egl_input_stream_settings = Argus::interface_cast<Argus::IEGLInputStreamSettings>(in_stream_settings_);
    if (!i_egl_input_stream_settings) {
        throw std::runtime_error("Cannot get IEGLInputStreamSettings Interface");
    }
    i_egl_input_stream_settings->setPixelFormat(Argus::PIXEL_FMT_RAW16);
    i_egl_input_stream_settings->setResolution(i_sensor_mode_->getResolution());

    in_stream_.reset(i_capture_session->createInputStream(in_stream_settings_.get()));
    if (!in_stream_) {
        throw std::runtime_error("Failed to create EGLInputStream");
    }
    i_egl_input_stream_ = Argus::interface_cast<Argus::IEGLInputStream>(in_stream_);

    Argus::IRequest* i_request = Argus::interface_cast<Argus::IRequest>(request_);
    if (!i_request) {
        throw std::runtime_error("Failed to create capture request");
    }

    Argus::Status status = i_request->enableInputStream(in_stream_.get(), in_stream_settings_.get());
    if (status != Argus::STATUS_OK) {
        throw std::runtime_error("Failed to enable stream in capture request");
    }

    status = i_request->setReprocessingEnable(true);
    if (status != Argus::STATUS_OK) {
        throw std::runtime_error("Failed to set Reprocessing enable in request");
    }

    // create and connect the input stream consumer
    status = i_capture_session->connectAllRequestInputStreams(request_.get(), 1);
    if (status != Argus::STATUS_OK) {
        throw std::runtime_error("Failed to connect input stream");
    }
}

void ArgusImpl::stop()
{
    in_stream_.reset();
    in_stream_settings_.reset();

    out_stream_.reset();
    out_stream_settings_.reset();

    capture_session_.reset();
    request_.reset();

    if (cuda_egl_i_connection_) {
        CudaCheck(cuEGLStreamProducerDisconnect(&cuda_egl_i_connection_));
    }

    if (cuda_egl_o_connection_) {
        CudaCheck(cuEGLStreamConsumerDisconnect(&cuda_egl_o_connection_));
    }

    return;
}

} // namespace hololink::operators
