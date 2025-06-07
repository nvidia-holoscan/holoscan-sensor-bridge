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

#include <gxf/std/tensor.hpp>
#include <holoscan/holoscan.hpp>
#include <npp.h>

#include "argus_isp.hpp"

// Complete type inclusion
// ensure this includes the full definition of ArgusImpl
#include <hololink/operators/argus_isp/argus_impl.hpp>
// ensure this includes the full definition of CameraProvider
#include <hololink/operators/argus_isp/camera_provider.hpp>

namespace hololink::operators {

void ArgusIspOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input<holoscan::gxf::Entity>("input");
    spec.output<holoscan::gxf::Entity>("output");

    spec.param(bayer_format_, "bayer_format", "BayerFormat", "Bayer format (one of hololink::operators::ArgusIspOp::BayerFormat)");
    spec.param(exposure_time_ms_, "exposure_time_ms", "ExposureTimeMs", "Exposure set for the camera sensor in milliseconds");
    spec.param(analog_gain_, "analog_gain", "AnalogGain", "Min analog gain");
    spec.param(pixel_bit_depth_, "pixel_bit_depth", "PixelBitDepth", "Pixel bit depth of the Bayer frame");
    spec.param(pool_, "pool", "Pool", "Pool to allocate the output message.");
    spec.param(out_tensor_name_, "out_tensor_name", "OutputTensorName", "Name of the output tensor", std::string(""));
    spec.param(camera_index_, "camera_index", "CameraIndex", "Argus camera index number");

    cuda_stream_handler_.define_params(spec);
}

void ArgusIspOp::start()
{
    camera_provider_ = std::make_unique<CameraProvider>();
    if (!camera_provider_) {
        throw std::runtime_error("Failed to initialize CameraProvider");
    }

    argus_impl_ = std::make_unique<ArgusImpl>(camera_provider_->get_camera_provider());
    if (!argus_impl_) {
        throw std::runtime_error("Failed to initialize ArgusImpl");
    }

    argus_impl_->output_pixel_format_ = OUTPUT_PIXEL_FORMAT_YUV420;

    // Get the camera devices.
    argus_impl_->setup_camera_devices(camera_index_);

    // set sensormode.
    uint32_t sensor_mode_index = 0;
    argus_impl_->set_sensor_mode_info(sensor_mode_index);

    // set reprocessing info
    argus_impl_->set_reprocess_info(bayer_format_.get(), pixel_bit_depth_.get());

    // set output stream
    argus_impl_->setup_output_streams(sensor_mode_index);

    // Pipe the eglStreams for camera images to CUDA.
    CudaCheck(cuInit(0));
    CUdevice cudaDevice;
    CudaCheck(cuDeviceGet(&cuda_device_, 0));
    CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));

    int integrated = 0;
    CudaCheck(cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, cuda_device_));
    is_integrated_ = (integrated != 0);

    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);

    CudaCheck(cuEGLStreamConsumerConnect(&argus_impl_->cuda_egl_o_connection_,
        argus_impl_->i_egl_output_stream_->getEGLStream()));

    argus_impl_->setup_capture_request(analog_gain_.get(), exposure_time_ms_.get());
    argus_impl_->setup_input_streams();

    CudaCheck(cuEGLStreamProducerConnect(&argus_impl_->cuda_egl_i_connection_,
        argus_impl_->i_egl_input_stream_->getEGLStream(),
        argus_impl_->i_sensor_mode_->getResolution().width(),
        argus_impl_->i_sensor_mode_->getResolution().height()));

    const uint32_t height = argus_impl_->i_sensor_mode_->getResolution().width();
    const uint32_t width = argus_impl_->i_sensor_mode_->getResolution().height();
    size_t nv12Size = (width * height) + (width * (height / 2));
    if (cuMemAlloc(&device_ptr_nv12_, nv12Size) != CUDA_SUCCESS) {
        throw std::runtime_error("Memory allocation to the device_ptr_nv12_ failed\n");
    }
}

void ArgusIspOp::stop()
{
    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);

    Argus::ICaptureSession* i_capture_session = Argus::interface_cast<Argus::ICaptureSession>(argus_impl_->capture_session_);
    i_capture_session->stopRepeat();

    argus_impl_->stop();
    argus_impl_.reset();

    camera_provider_.reset();

    for (int i = 0; i < MaxNumberOfFrames; i++) {
        tensor_pointers_[i].reset();
    }

    cuMemFree(device_ptr_nv12_);
    CudaCheck(cuDevicePrimaryCtxRelease(cuda_device_));
    cuda_context_ = nullptr;
}

void ArgusIspOp::compute(holoscan::InputContext& input,
    holoscan::OutputContext& output,
    holoscan::ExecutionContext& context)
{
    auto maybe_entity = input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) {
        throw std::runtime_error("Failed to receive input");
    }

    auto& entity = maybe_entity.value();

    // get the CUDA stream from the input message
    gxf_result_t stream_handler_result = cuda_stream_handler_.from_message(context.context(), entity);
    if (stream_handler_result != GXF_SUCCESS) {
        throw std::runtime_error(fmt::format("Failed to get the CUDA stream from incoming messages: {}", GxfResultStr(stream_handler_result)));
    }

    const auto input_tensor = entity.get<holoscan::Tensor>();
    if (!input_tensor) {
        throw std::runtime_error("Tensor not found in message");
    }

    DLDevice input_device = input_tensor->device();

    if (input_device.device_type == kDLCUDAHost) {
        throw std::runtime_error("The tensor is on the host");
    } else if (!is_integrated_) {
        throw std::runtime_error("The tensor is not on the iGPU");
    }

    if (input_tensor->ndim() != 3) {
        throw std::runtime_error("Tensor must be an image");
    }

    DLDataType dtype = input_tensor->dtype();
    if (dtype.code != kDLUInt || dtype.bits != 16) {
        throw std::runtime_error(fmt::format("Unexpected image data type '(code: {}, bits: {})',"
                                             "expected '(code: {}, bits: {})'",
            static_cast<int>(dtype.code), dtype.bits, static_cast<int>(kDLUInt), 16));
    }

    const auto input_shape = input_tensor->shape();
    const uint32_t height = input_shape[0];
    const uint32_t width = input_shape[1];
    const uint32_t components = input_shape[2];
    if (components != 1) {
        throw std::runtime_error(fmt::format("Unexpected component count {}, expected '1'", components));
    }

    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);
    cudaStream_t cuda_stream = cuda_stream_handler_.get_cuda_stream(context.context());

    // Map the cuda pointer to the EGL frame
    size_t frame_index = frame_number_ % MaxNumberOfFrames;
    producerEglFrame[frame_index].width = argus_impl_->i_sensor_mode_->getResolution().width();
    producerEglFrame[frame_index].height = argus_impl_->i_sensor_mode_->getResolution().height();
    producerEglFrame[frame_index].pitch = 0;
    producerEglFrame[frame_index].planeCount = 1; // Bayer image
    producerEglFrame[frame_index].depth = 1;
    producerEglFrame[frame_index].numChannels = 1;
    producerEglFrame[frame_index].frameType = CU_EGL_FRAME_TYPE_PITCH;
    producerEglFrame[frame_index].eglColorFormat = CU_EGL_COLOR_FORMAT_BAYER_RGGB;
    producerEglFrame[frame_index].cuFormat = CU_AD_FORMAT_UNSIGNED_INT16;

    tensor_pointers_[frame_index] = input_tensor->dl_ctx();
    producerEglFrame[frame_index].frame.pPitch[0] = reinterpret_cast<void*>(input_tensor->data());

    CUresult r = cuEGLStreamProducerPresentFrame(&argus_impl_->cuda_egl_i_connection_,
        producerEglFrame[frame_index],
        &cuda_stream);
    if (r != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to present the frame to the cuEGLStream\n");
    }

    if (frame_number_ > 0) {
        CUresult res;
        CUeglFrame frame;
        do {
            res = cuEGLStreamProducerReturnFrame(&argus_impl_->cuda_egl_i_connection_,
                &frame,
                &cuda_stream);
        } while (res != CUDA_SUCCESS);

        tensor_pointers_[(frame_number_ - 1) % MaxNumberOfFrames].reset();
    }

    Argus::ICaptureSession* i_capture_session = Argus::interface_cast<Argus::ICaptureSession>(argus_impl_->capture_session_);
    if (!i_capture_session) {
        throw std::runtime_error("Failed to create CaptureSession in setup capture request");
    }

    if (i_capture_session->capture(argus_impl_->request_.get()) == 0)
        throw std::runtime_error("Failed to submit capture request");

    CUgraphicsResource cuda_resource = { 0 };
    if (cuEGLStreamConsumerAcquireFrame(&argus_impl_->cuda_egl_o_connection_,
            &cuda_resource, &cuda_stream, CUDA_EGL_INFINITE_TIMEOUT)
        != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to acquire frame from EGLStream");
    }

    CUeglFrame cuda_egl_frame;
    if (cuGraphicsResourceGetMappedEglFrame(&cuda_egl_frame, cuda_resource, 0, 0) != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to map output eglframe");
    }

    // get handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        fragment()->executor().context(), pool_->gxf_cid());

    nvidia::gxf::Expected<nvidia::gxf::Entity> out_message
        = CreateTensorMap(context.context(),
            allocator.value(),
            { { out_tensor_name_.get(),
                nvidia::gxf::MemoryStorageType::kDevice,
                nvidia::gxf::Shape { static_cast<int>(height),
                    static_cast<int>(width),
                    3 },
                nvidia::gxf::PrimitiveType::kUnsigned8,
                0,
                nvidia::gxf::ComputeTrivialStrides(
                    nvidia::gxf::Shape { static_cast<int>(height),
                        static_cast<int>(width),
                        3 },
                    nvidia::gxf::PrimitiveTypeSize(nvidia::gxf::PrimitiveType::kUnsigned8)) } },
            false);

    if (!out_message) {
        throw std::runtime_error("failed to create out_message\n");
    }

    const auto tensor = out_message.value().get<nvidia::gxf::Tensor>(out_tensor_name_.get().c_str());
    if (!tensor) {
        throw std::runtime_error(
            fmt::format("failed to create out_tensor with name \"{}\"", out_tensor_name_.get()));
    }

    void* output_data_ptr = tensor.value()->pointer();
    if (!output_data_ptr) {
        throw std::runtime_error("output data_ptr is invalid\n");
    }
    // add the npp function
    NppStreamContext npp_stream_ctx_ {};
    npp_stream_ctx_.hStream = cuda_stream_handler_.get_cuda_stream(context.context());

    CUDA_MEMCPY2D pCopyParams;
    pCopyParams.srcY = 0u;
    pCopyParams.srcXInBytes = 0u;
    pCopyParams.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    pCopyParams.srcArray = cuda_egl_frame.frame.pArray[0];
    pCopyParams.dstY = 0u;
    pCopyParams.dstXInBytes = 0u;
    pCopyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    pCopyParams.dstDevice = device_ptr_nv12_;
    pCopyParams.dstPitch = static_cast<unsigned int>(cuda_egl_frame.width);
    pCopyParams.WidthInBytes = static_cast<unsigned int>(cuda_egl_frame.width);
    pCopyParams.Height = static_cast<unsigned int>(cuda_egl_frame.height);
    if (cuMemcpy2DAsync(&pCopyParams, cuda_stream) != CUDA_SUCCESS) {
        throw std::runtime_error("Could not copy from Y array to device\n");
    }
    pCopyParams.srcArray = cuda_egl_frame.frame.pArray[1];
    pCopyParams.dstY = static_cast<unsigned int>(cuda_egl_frame.height);
    pCopyParams.Height = static_cast<unsigned int>(cuda_egl_frame.height) / 2;
    if (cuMemcpy2DAsync(&pCopyParams, cuda_stream) != CUDA_SUCCESS) {
        throw std::runtime_error("Could not copy from UV array to device\n");
    }

    const auto in_y_ptr = reinterpret_cast<const uint8_t*>(device_ptr_nv12_);
    const auto in_uv_ptr = in_y_ptr + width * height;
    const uint8_t* in_y_uv_ptrs[2] = { in_y_ptr, in_uv_ptr };

    NppiSize oSizeROI;
    oSizeROI.width = width;
    oSizeROI.height = height;
    NppStatus status = nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx(
        in_y_uv_ptrs,
        width * sizeof(Npp8u),
        static_cast<Npp8u*>(output_data_ptr),
        width * sizeof(Npp8u) * 3,
        oSizeROI,
        npp_stream_ctx_);
    if (status != NPP_SUCCESS) {
        throw std::runtime_error(fmt::format("Failed with \"{}\" to convert NV12 to RGB\n", static_cast<int>(status)));
    }
    // pass the CUDA stream to the output message
    stream_handler_result = cuda_stream_handler_.to_message(out_message);
    if (stream_handler_result != GXF_SUCCESS) {
        throw std::runtime_error("Failed to add the CUDA stream to the outgoing messages");
    }
    // Emit the tensor
    auto result = holoscan::gxf::Entity(std::move(out_message.value()));

    if (cuEGLStreamConsumerReleaseFrame(&argus_impl_->cuda_egl_o_connection_, cuda_resource, &cuda_stream) != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to release the frame");
    }

    frame_number_++;

    output.emit(result, "output");
}

} // namespace hololink::operators
